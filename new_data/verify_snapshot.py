"""Independently check a ZIP snapshot's membership, group boundaries and hashes."""
import argparse
import collections
import csv
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from PIL import Image


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',required=True); parser.add_argument('--config',required=True)
    parser.add_argument('--output',default='verification.json')
    args=parser.parse_args()
    snap=Path(args.snapshot); cfg=json.loads(Path(args.config).read_text(encoding='utf8'))
    archive=Path(os.path.expandvars(cfg['archive_path'])); root=Path(cfg['v2_root'])
    audit=json.loads((snap/'audit.json').read_text(encoding='utf8'))
    for name,expected in audit['manifest_sha256'].items():
        if hashlib.sha256((snap/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Snapshot file changed: {name}')
    for split,expected in audit['holdout_manifest_sha256'].items():
        if hashlib.sha256((root/(split+'.csv')).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Holdout changed: {split}')
    digest=hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): digest.update(block)
    if digest.hexdigest()!=audit['archive_sha256'] or digest.hexdigest()!=cfg['expected_archive_sha256']:
        raise ValueError('ZIP hash changed')
    train=read(snap/'train.csv'); excluded=read(snap/'excluded.csv')
    ids=[r['image_id'] for r in train]; excluded_ids=[r['image_id'] for r in excluded]
    retained=set(ids)
    if len(ids)!=len(retained) or retained&set(excluded_ids) or len(ids)+len(excluded_ids)!=audit['input_images']:
        raise ValueError('Invalid input partition')
    graph=collections.defaultdict(set)
    for edge in read(snap/'group-evidence.csv'):
        graph[edge['id_a']].add(edge['id_b']); graph[edge['id_b']].add(edge['id_a'])
    blocked={r['image_id'] for name in ('validation','synthetic_test','real_source_test') for r in read(root/(name+'.csv'))}
    blocked.update(r['image_id'] for r in read(root/'quarantine.csv'))
    todo=collections.deque(blocked); visited=set(blocked)
    while todo:
        for neighbor in graph[todo.popleft()]:
            if neighbor not in visited: visited.add(neighbor); todo.append(neighbor)
    if retained&visited:
        raise ValueError('Retained group touches holdout or quarantine')
    for candidate in read(cfg['holdout_candidates']):
        if candidate['holdout_id'] not in graph[candidate['train_id']]:
            raise ValueError('A holdout candidate is missing from the group evidence')
        if candidate['train_id'] in retained:
            raise ValueError('A retained sample is a holdout candidate')
    component_members=collections.defaultdict(set)
    for row in read(root/'group_audit.csv'):
        component_members[row['candidate_component_id']].add(row['image_id'])
    for members in component_members.values():
        if members & retained and members & blocked:
            raise ValueError('A retained sample shares a v2 candidate component with a blocked sample')
    old_train={r['image_id']:r for r in read(root/'train.csv')}
    if not retained<=set(old_train): raise ValueError('Snapshot includes an ID outside v2 train')
    mapping=json.loads((snap/'class-mapping.json').read_text(encoding='utf8'))
    for task in ('type','style'):
        expected={x:i for i,x in enumerate(sorted({r[task] for r in train}))}
        if expected!=mapping[task]: raise ValueError('Mapping is not derived from retained train')
    with zipfile.ZipFile(archive) as z:
        for row in train:
            original=old_train[row['image_id']]
            if (row['type'],row['style'])!=(original['type'],original['style']):
                raise ValueError('Label differs from v2 source: '+row['image_id'])
            content=z.read(cfg['archive_prefix']+row['image_path'])
            with Image.open(root/original['image_path']) as im:
                output=io.BytesIO(); im.convert('RGB').resize((256,256),Image.Resampling.LANCZOS).save(output,'JPEG',quality=95)
            if hashlib.sha256(content).hexdigest()!=row['zip_file_sha256'] or content!=output.getvalue():
                raise ValueError('ZIP sample does not match verified v2 source: '+row['image_id'])
    result={'status':'pass_conservative_group_checks','images_checked':len(train),
            'components_touching_holdout_or_quarantine_in_train':0,'reencoded_jpeg_exact':len(train),
            'train_only_mapping':True, 'novel_images_relative_to_v2_train':0,
            'label_review':'AI visual sample only; style annotations not independently verified',
            'test_predictions_computed':False}
    dest=snap/args.output
    with dest.open('x',encoding='utf8') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
