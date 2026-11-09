"""Build a conservative ZIP snapshot: dHash edges quarantine groups, not identities."""
import argparse
import collections
import csv
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from furniture.data import dhash, rgb


class Groups:
    def __init__(self):
        self.parents = {}

    def find(self, item):
        self.parents.setdefault(item, item)
        if self.parents[item] != item:
            self.parents[item] = self.find(self.parents[item])
        return self.parents[item]

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parents[max(a,b)] = min(a,b)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_csv(path, fields, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def jpeg_sha(image_path):
    with Image.open(image_path) as image:
        encoded = io.BytesIO()
        image.convert('RGB').resize((256,256), Image.Resampling.LANCZOS).save(encoded,'JPEG',quality=95)
        return sha(encoded.getvalue())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    archive = Path(os.path.expandvars(cfg['archive_path'])).resolve()
    root = Path(cfg['v2_root']).resolve()
    target = Path(args.output).resolve()
    target.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''):
            digest.update(block)
    if digest.hexdigest() != cfg['expected_archive_sha256']:
        raise ValueError('ZIP hash differs from audited archive')
    splits = {name:read_csv(root/(name+'.csv')) for name in ('train','validation','synthetic_test','real_source_test')}
    reference = {r['image_id']:(s,r) for s, rows in splits.items() for r in rows}
    groups = Groups(); edges = []
    def link(a,b,reason,distance=''):
        groups.union(a,b)
        edges.append({'id_a':a,'id_b':b,'evidence':reason,'distance':distance})
    old_groups = collections.defaultdict(list)
    for row in read_csv(root/'group_audit.csv'):
        old_groups[row['candidate_component_id']].append(row['image_id'])
    for component, members in old_groups.items():
        for member in members[1:]: link(members[0],member,'v2_candidate_component:'+component)
    quarantined = {r['image_id'] for r in read_csv(root/'quarantine.csv')}
    with zipfile.ZipFile(archive) as z:
        prefix = cfg['archive_prefix']
        with z.open(prefix+'train.csv') as f:
            raw = list(csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',newline='')))
        rows, hvalues, zip_file_hashes, matches = [], [], collections.defaultdict(list), {}
        for record in raw:
            ident = 'train-'+record['src_index'].zfill(7)
            if ident in matches: raise ValueError('Repeated source index in ZIP train')
            content = z.read(prefix+record['path'])
            encoded_hash = sha(content); zip_file_hashes[encoded_hash].append(ident)
            with Image.open(io.BytesIO(content)) as im:
                im = rgb(im); im.load(); hvalues.append(int(dhash(im),16))
            matched = False
            if ident in reference:
                _, ref = reference[ident]
                matched = encoded_hash == jpeg_sha(root/ref['image_path'])
            matches[ident] = matched
            row = {'image_id':ident,'image_path':record['path'], 'type':record['type'],
                   'style':record['style'], 'style_raw':record['style_raw'].strip().lower(),
                   'zip_file_sha256':encoded_hash, 'source_id_jpeg_exact':str(matched)}
            rows.append(row); groups.find(ident)
    for values in zip_file_hashes.values():
        for other in values[1:]: link(values[0],other,'zip_member_sha256')
    # Preserve original source identity, but never trust a source index as proof by itself.
    holdouts = [(split,r) for split in ('validation','synthetic_test','real_source_test') for r in splits[split]]
    for split, row in holdouts:
        ident = row['image_id']; groups.find(ident)
        for incoming in zip_file_hashes.get(jpeg_sha(root/row['image_path']), []):
            link(incoming, ident, 'v2_reencoded_jpeg_sha256:'+split)
    values = np.array(hvalues,dtype=np.uint64)
    lookup = np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
    threshold = int(cfg['dhash_threshold'])
    if not 0 <= threshold <= 64:
        raise ValueError('dHash threshold must be between 0 and 64')
    for i, row in enumerate(rows):
        xor = np.bitwise_xor(values[i+1:], values[i]).view(np.uint8).reshape(-1,8)
        for position in np.flatnonzero(lookup[xor].sum(axis=1) <= threshold):
            j = i+1+int(position)
            link(row['image_id'],rows[j]['image_id'],'zip_train_dhash_candidate',int(lookup[xor[position]].sum()))
    for candidate in read_csv(cfg['holdout_candidates']):
        link(candidate['train_id'],candidate['holdout_id'],
             'zip_holdout_dhash_candidate:'+candidate['holdout_split'],candidate['distance'])
    holdout_ids = {r['image_id'] for _,r in holdouts}
    for ident in holdout_ids | quarantined: groups.find(ident)
    blocked_holdout = {groups.find(ident) for ident in holdout_ids}
    blocked_quarantine = {groups.find(ident) for ident in quarantined}
    blocked_unverified = {groups.find(row['image_id']) for row in rows if not matches[row['image_id']]}
    selected, excluded = [], []
    for row in rows:
        ident = row['image_id']; group = groups.find(ident)
        reasons = []
        if group in blocked_holdout: reasons.append('candidate_component_touches_holdout')
        if group in blocked_quarantine: reasons.append('candidate_component_touches_v2_quarantine')
        if group in blocked_unverified: reasons.append('component_contains_unverified_source_identity')
        if ident not in reference or reference[ident][0] != 'train': reasons.append('not_in_v2_retained_train')
        if ident in reference and reference[ident][0] == 'train':
            ref = reference[ident][1]
            if (row['type'],row['style_raw']) != (ref['type'],ref['style']):
                reasons.append('raw_label_differs_from_v2')
        if reasons:
            excluded.append({**row,'group_id':group,'reasons':'|'.join(reasons)})
        else:
            selected.append({'image_id':ident,'image_path':row['image_path'],
                             'type':ref['type'],'style':ref['style'],'group_id':group,
                             'zip_file_sha256':row['zip_file_sha256']})
    # A label mismatch is local evidence of uncertainty: exclude its whole group too.
    label_block = {r['group_id'] for r in excluded if 'raw_label_differs_from_v2' in r['reasons']}
    kept = []
    for row in selected:
        if row['group_id'] in label_block:
            raw_row = next(r for r in rows if r['image_id']==row['image_id'])
            excluded.append({**raw_row,'group_id':row['group_id'],'reasons':'component_contains_label_mismatch'})
        else: kept.append(row)
    selected = kept
    save_csv(target/'train.csv', ['image_id','image_path','type','style','group_id','zip_file_sha256'],selected)
    save_csv(target/'excluded.csv', ['image_id','image_path','type','style','style_raw','zip_file_sha256','source_id_jpeg_exact','group_id','reasons'],excluded)
    save_csv(target/'group-evidence.csv',['id_a','id_b','evidence','distance'],edges)
    mapping = {key:{value:i for i,value in enumerate(sorted({r[key] for r in selected}))} for key in ('type','style')}
    (target/'class-mapping.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    counts = collections.Counter((r['type'],r['style']) for r in selected)
    before = collections.Counter((r['type'],r['style_raw']) for r in rows)
    combinations = sorted(before.keys() | counts.keys())
    save_csv(target/'class-counts.csv',['type','style','before','after'],
             [{'type':a,'style':b,'before':before[(a,b)],'after':counts[(a,b)]} for a,b in combinations])
    # Exact names only; ambiguous legacy aliases are recorded, never silently collapsed.
    v2map = json.loads((root/'audit.json').read_text(encoding='utf-8'))['class_mapping']
    with zipfile.ZipFile(archive) as z: zmap = json.loads(z.read(cfg['archive_prefix']+'labels.json'))
    aliases = zmap.get('style_aliases',{})
    comparisons = []
    for task in ('type','style'):
        zipnames = set(zmap[task+'2id'])
        for name in sorted(zipnames | set(v2map[task])):
            comparisons.append({'task':task,'name':name,'in_zip':name in zipnames,'in_v2':name in v2map[task],
              'zip_alias_target':aliases.get(name,'') if task=='style' else '',
              'status':'exact_name' if name in zipnames and name in v2map[task] else 'not_exact_name'})
    save_csv(target/'mapping-comparison.csv',['task','name','in_zip','in_v2','zip_alias_target','status'],comparisons)
    report = {'archive_sha256':digest.hexdigest(),'input_images':len(rows),'retained_images':len(selected),
              'retained_groups':len({r['group_id'] for r in selected}),'excluded_images':len(excluded),
              'excluded_groups':len({r['group_id'] for r in excluded}),
              'source_index_exact_reencoded_jpeg':sum(matches.values()),
              'group_edges':len(edges),'group_policy':'transitive candidate links are conservative quarantine, not identity decisions',
              'augmentations':'none; no offline copies used', 'real_source_review':'AI preliminary only; all candidate components quarantined',
              'unseen_validation':{task:sorted({r[task] for r in splits['validation']}-set(mapping[task])) for task in mapping},
              'manifest_sha256':{name:sha((target/name).read_bytes()) for name in ('train.csv','excluded.csv','group-evidence.csv','class-mapping.json')},
              'holdout_manifest_sha256':{name:sha((root/(name+'.csv')).read_bytes()) for name in ('validation','synthetic_test','real_source_test')},
              'status':'requires post-build group audit and label review before pilot',
              'novel_data_claim':'none: verified selected samples are also present in v2 train'}
    (target/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
