"""Audit and batch-read a ZIP locally. Findings do not approve a training snapshot."""
import argparse
import collections
import csv
import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image
from furniture.data import dhash, rgb
from new_data.zip_dataset import loader_from_config


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    archive = Path(os.path.expandvars(cfg['archive_path'])).expanduser().resolve()
    prefix = cfg['archive_prefix']
    root = Path(os.path.expandvars(cfg['v2_root'])).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    if digest.hexdigest() != cfg['expected_archive_sha256']:
        raise ValueError('Archive SHA-256 does not match the verified source')
    errors, entries, hashes = [], [], []
    with zipfile.ZipFile(archive) as z:
        mapping = json.loads(z.read(prefix+'labels.json'))
        label_maps = {'type':mapping['type2id'], 'style':mapping['style2id']}
        with z.open(prefix+'train.csv') as f:
            source = list(csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline='')))
        ids, paths = set(), set()
        for row in source:
            path = row['path']
            ident = 'train-'+row['src_index'].zfill(7)
            issue = []
            parts = PurePosixPath(path).parts
            if len(parts) != 5 or parts[:2] != ('images', 'train') or parts[2] != row['type'] or parts[3] != row['style'].replace(' ', '_'):
                issue.append('path does not agree with type/style')
            if ident in ids or path in paths:
                issue.append('duplicate source ID or path')
            ids.add(ident); paths.add(path)
            for task in ('type','style'):
                if row[task] not in label_maps[task] or str(label_maps[task].get(row[task])) != row[task+'_id']:
                    issue.append(task+' class ID disagrees with labels.json')
            image_dhash = None
            try:
                with z.open(prefix+path) as stream, Image.open(stream) as im:
                    im = rgb(im)
                    im.load()
                    image_dhash = dhash(im)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
                issue.append('image read: '+str(error))
            if issue:
                errors.append({'image_id':ident, 'image_path':path, 'type':row['type'],
                               'style':row['style'], 'reason':'; '.join(issue)})
            if image_dhash is not None:
                entries.append({'image_id':ident, 'image_path':path, 'type':row['type'], 'style':row['style']})
                hashes.append(int(image_dhash, 16))
    write_csv(out/'errors.csv', ['image_id','image_path','type','style','reason'], errors)
    write_csv(out/'train-manifest.csv', ['image_id','image_path','type','style'], entries)
    (out/'mapping.json').write_text(json.dumps(label_maps, indent=2), encoding='utf-8')
    if len(set(r['type'] for r in source)) != len(label_maps['type']) or len(set(r['style'] for r in source)) != len(label_maps['style']):
        raise ValueError('Training labels do not cover the supplied mapping')
    reference = []
    for split in ('validation','synthetic_test','real_source_test'):
        with (root/(split+'.csv')).open(encoding='utf-8', newline='') as f:
            reference.extend((split, r) for r in csv.DictReader(f))
    reference_hashes = np.array([int(row['dhash'], 16) for _,row in reference], dtype=np.uint64)
    bitcounts = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    candidates = []
    for entry, value in zip(entries, hashes):
        xor = np.bitwise_xor(reference_hashes, np.uint64(value)).view(np.uint8).reshape(-1,8)
        distances = bitcounts[xor].sum(axis=1)
        for index in np.flatnonzero(distances <= cfg.get('dhash_threshold', 4)):
            split, ref = reference[int(index)]
            candidates.append({'train_id':entry['image_id'], 'train_path':entry['image_path'],
                               'holdout_split':split, 'holdout_id':ref['image_id'],
                               'distance':int(distances[index]), 'decision':'unreviewed'})
    write_csv(out/'holdout-dhash-candidates.csv',
              ['train_id','train_path','holdout_split','holdout_id','distance','decision'], candidates)
    counts = collections.Counter(row['holdout_split'] for row in candidates)
    run_config = {'storage_mode':'zip','archive_path':str(archive),'archive_prefix':prefix,
                  'manifest_path':str(out/'train-manifest.csv'),'snapshot_id':'inspection-only-not-approved',
                  'corrupt_image_policy':'error','batch_size':cfg['batch_size'],'num_workers':cfg['num_workers'],
                  'prefetch_factor':2,'persistent_workers':cfg['num_workers']>0,'seed':cfg['seed'],
                  'shuffle':False,'image_size':cfg['image_size'],'drop_last':False,'augment':False}
    (out/'loader-config.json').write_text(json.dumps(run_config, indent=2),encoding='utf-8')
    result = {'archive':str(archive),'archive_sha256':digest.hexdigest(), 'train_rows':len(source),
              'decoded':len(entries),'errors':len(errors),'candidate_pairs_by_split':dict(counts),
              'dhash_threshold':cfg.get('dhash_threshold',4),
              'candidate_note':'dHash candidates are not identity decisions; JPEG recompression can change hashes',
              'label_note':'CSV label, class ID and folder agreement only; not human semantic validation',
              'training_approved':False, 'training_performed':False}
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if errors:
        print(json.dumps(result,indent=2)); raise SystemExit('Audit errors; batch trial stopped; see errors.csv')
    torch.set_num_threads(1)
    loader = loader_from_config(out/'loader-config.json', label_maps)
    start = time.perf_counter(); offset = batches = 0
    for images, labels in loader:
        count = len(images)
        for task in ('type','style'):
            expected = [label_maps[task][row[task]] for row in entries[offset:offset+count]]
            if labels[task].tolist() != expected:
                raise ValueError(f'Batch labels disagree with manifest at row {offset}: {task}')
        if not torch.isfinite(images).all():
            raise ValueError(f'Nonfinite image tensor at row {offset}')
        offset += count; batches += 1
    if offset != len(entries):
        raise ValueError(f'Expected {len(entries)} images, read {offset}')
    result['batch_trial'] = {'images':offset,'batches':batches,'seconds':time.perf_counter()-start,
                             'batch_size':cfg['batch_size'],'workers':cfg['num_workers'],
                             'drop_last':False,'shuffle':False,'device':'CPU',
                             'cache':'uncontrolled OS cache; not a training benchmark'}
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
