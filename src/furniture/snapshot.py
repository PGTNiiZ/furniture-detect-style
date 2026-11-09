"""Build a separate, reproducible closed-set snapshot from an audited intake.

The conservative candidate graph is used only for quarantine, never as an
identity decision. Confirmed same/related decisions alone create group IDs.
"""
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from .data import read_rows, write_rows, validate_splits, image_hash

PROTOCOL = 'furniture-dataset-2.0'
ALIASES = {'train': 'train', 'val': 'validation', 'test': 'synthetic_test'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Groups:
    def __init__(self, ids):
        self.parents = {i: i for i in ids}

    def find(self, i):
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]
            i = self.parents[i]
        return i

    def union(self, a, b):
        self.parents[self.find(a)] = self.find(b)

    def components(self):
        result = defaultdict(list)
        for i in self.parents:
            result[self.find(i)].append(i)
        return list(result.values())


def build_snapshot(source_root, destination, candidate_csv, audit_json, *, source_dataset_id, source_revision, extra_external=None, decisions_csv=None):
    """Quarantine cross-split candidate components; preserve source and review evidence.

    extra_external is a directory with rows.csv and image files at image_path.
    Source-test labels, including missing and unseen labels, are never used to fit
    the training mapping or validate the closed-set training partitions.
    """
    source_root, destination = Path(source_root), Path(destination)
    candidate_csv, audit_json = Path(candidate_csv), Path(audit_json)
    if destination.exists():
        raise ValueError('Snapshot destination exists; choose a new directory')
    if destination.resolve().is_relative_to(source_root.resolve()):
        raise ValueError('Snapshot destination must be outside the source root')
    if not isinstance(source_dataset_id, str) or not source_dataset_id.strip():
        raise ValueError('Source dataset ID must be explicit')
    if not isinstance(source_revision, str) or len(source_revision) != 40 or any(c not in '0123456789abcdef' for c in source_revision.lower()):
        raise ValueError('Source revision must be a pinned 40-character SHA')
    audit = json.loads(audit_json.read_text(encoding='utf-8'))
    if audit.get('truncated') is not False or audit.get('threshold') != 4:
        raise ValueError('An exhaustive threshold-4 candidate audit is required')
    for split in ALIASES:
        if audit['manifest_sha256'][split] != sha(source_root / f'{split}.csv'):
            raise ValueError(f'Source manifest changed: {split}')
    source = {s: read_rows(source_root / f'{s}.csv') for s in ALIASES}
    train_rows = [(ALIASES[s], row) for s, rows in source.items() for row in rows if row['source_split'] == 'train']
    external = [row for rows in source.values() for row in rows if row['source_split'] == 'test']
    if sum(map(len, source.values())) != len(train_rows) + len(external):
        raise ValueError('Unexpected source split')
    all_rows = {r['image_id']: (s, r) for s, r in train_rows}
    if len(all_rows) != len(train_rows):
        raise ValueError('Duplicate source-train ID')
    if len(train_rows) != audit['rows'] or audit['examined_pairs'] != len(train_rows)*(len(train_rows)-1)//2:
        raise ValueError('Candidate audit does not cover the source train')
    candidates = read_rows(candidate_csv)
    if len(candidates) != audit['candidate_pairs']:
        raise ValueError('Candidate CSV count differs from audit')
    conservative = Groups(all_rows)
    confirmed = Groups(all_rows)
    keys = {}
    for _, r in train_rows:
        for key in [('pixel', r['pixel_hash']), ('group', r.get('group_id'))]:
            if key[1]:
                if key in keys:
                    conservative.union(r['image_id'], keys[key])
                    confirmed.union(r['image_id'], keys[key])
                keys[key] = r['image_id']
    decisions = {}
    if decisions_csv is not None:
        for row in read_rows(decisions_csv):
            key = tuple(sorted((row['image_id_a'], row['image_id_b'])))
            decision = row.get('decision', '').strip().lower()
            if decision == 'unreviewed':
                decision = ''
            if key in decisions or decision not in ('', 'same', 'related', 'different', 'uncertain'):
                raise ValueError('Invalid or duplicate review decision')
            decisions[key] = decision
    seen = set()
    for pair in candidates:
        a, b = pair['image_id_a'], pair['image_id_b']
        if a not in all_rows or b not in all_rows or a == b or tuple(sorted((a,b))) in seen:
            raise ValueError('Invalid or duplicate candidate pair')
        seen.add(tuple(sorted((a,b))))
        a_split, a_row = all_rows[a]
        b_split, b_row = all_rows[b]
        if ALIASES.get(pair['split_a']) != a_split or ALIASES.get(pair['split_b']) != b_split:
            raise ValueError('Candidate split does not match source manifest')
        distance = (int(a_row['dhash'], 16) ^ int(b_row['dhash'], 16)).bit_count()
        if distance > 4 or distance != int(pair['distance']):
            raise ValueError('Candidate distance does not match source hashes')
        decision = decisions.get(tuple(sorted((a, b))), '')
        if decision != 'different':
            conservative.union(a, b)
        if decision in ('same', 'related'):
            confirmed.union(a, b)
    if set(decisions) - seen:
        raise ValueError('Review decision refers to a non-candidate pair')
    for (a, b), decision in decisions.items():
        if decision == 'different' and confirmed.find(a) == confirmed.find(b):
            raise ValueError('Different decision contradicts a confirmed transitive group')
    quarantine = {i for group in conservative.components()
                  if len({all_rows[i][0] for i in group}) > 1 for i in group}
    cross_before = [p for p in candidates if all_rows[p['image_id_a']][0] != all_rows[p['image_id_b']][0]]
    if 'cross_split_pairs' in audit and len(cross_before) != audit['cross_split_pairs']:
        raise ValueError('Candidate CSV cross-split count differs from audit')
    remaining_cross = [p for p in cross_before if p['image_id_a'] not in quarantine and p['image_id_b'] not in quarantine
                       and decisions.get(tuple(sorted((p['image_id_a'], p['image_id_b']))), '') != 'different']
    if remaining_cross:
        raise ValueError('Unresolved cross-split candidates remain after quarantine')
    splits = {s: [] for s in ('train', 'validation', 'synthetic_test', 'real_source_test')}
    for group in confirmed.components():
        if len(group) > 1:
            group_id = 'confirmed-' + hashlib.sha256('\n'.join(sorted(group)).encode()).hexdigest()[:20]
            for i in group:
                all_rows[i][1]['group_id'] = group_id
    for split, row in train_rows:
        if row['image_id'] not in quarantine:
            splits[split].append(row)
    external_files = [(row, source_root) for row in external]
    if extra_external is not None:
        extra_external = Path(extra_external)
        external_files += [(row, extra_external) for row in read_rows(extra_external/'rows.csv')]
    ids = set(all_rows)
    for row, _ in external_files:
        if row['source_split'] != 'test' or row['image_id'] in ids:
            raise ValueError('Invalid or duplicate external sample ID')
        ids.add(row['image_id'])
        splits['real_source_test'].append(row)
    # Validate source paths before opening files or creating the destination.
    # Resolve also detects junctions/symlinks that leave their source root.
    files = [(row, source_root) for split in ('train', 'validation', 'synthetic_test') for row in splits[split]] + external_files
    paths = set()
    def checked_path(origin, relative):
        path = Path(relative)
        if not relative or path.is_absolute() or path.drive:
            raise ValueError('Image path must be relative to its source root')
        resolved = (origin / path).resolve()
        try:
            resolved.relative_to(origin.resolve())
        except ValueError:
            raise ValueError('Image path escapes its source root') from None
        return resolved
    for row, origin in files:
        checked_path(origin, row['image_path'])
        path_key = str((destination / row['image_path']).resolve()).casefold()
        if path_key in paths:
            raise ValueError('Image paths collide between samples')
        paths.add(path_key)
    internal_hashes = {r['pixel_hash'] for _, r in train_rows}
    internal_groups = {r['group_id'] for _, r in train_rows if r.get('group_id')}
    external_overlaps = {'exact_pixel_ids': [r['image_id'] for r in splits['real_source_test'] if r['pixel_hash'] in internal_hashes],
                         'group_ids': [r['image_id'] for r in splits['real_source_test'] if r.get('group_id') and r['group_id'] in internal_groups]}
    if any(external_overlaps.values()):
        raise ValueError('External reference overlaps a training snapshot by exact pixels or group')
    for row, origin in external_files:
        with Image.open(checked_path(origin, row['image_path'])) as image:
            if image_hash(image) != row['pixel_hash']:
                raise ValueError(f'Changed external image: {row["image_id"]}')
    mapping = {k: {v: i for i, v in enumerate(sorted({r[k] for r in splits['train']}))} for k in ('type', 'style')}
    for split in ('validation', 'synthetic_test'):
        for row in splits[split]:
            if any(row[k] not in mapping[k] for k in mapping):
                raise ValueError(f'Closed-set violation in {split}: {row["image_id"]}')
    # Refuse to overwrite even a partially written snapshot on a subsequent invocation.
    destination.mkdir(parents=True)
    for split in ('train', 'validation', 'synthetic_test'):
        for row in splits[split]:
            target = destination / row['image_path']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checked_path(source_root, row['image_path']), target)
    for row, origin in external_files:
        target = destination / row['image_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError('External image path collides with another sample')
        shutil.copy2(checked_path(origin, row['image_path']), target)
    for split, rows in splits.items():
        write_rows(destination / f'{split}.csv', rows)
    for old, new in ALIASES.items():
        if old != new:
            shutil.copyfile(destination/f'{new}.csv', destination/f'{old}.csv')
    validated, checked_mapping = validate_splits(destination)
    if mapping != checked_mapping:
        raise ValueError('Training class mapping does not match validation')
    import_exclusions = json.loads((source_root/'excluded_samples.json').read_text(encoding='utf-8'))
    prior_excluded = import_exclusions.get('excluded', [])
    recovered = {r['image_id'] for r, _ in external_files} & {r['image_id'] for r in prior_excluded}
    group_details = {}
    for component in conservative.components():
        group_id = 'candidate-' + hashlib.sha256('\n'.join(sorted(component)).encode()).hexdigest()[:20]
        component_splits = sorted({all_rows[i][0] for i in component})
        for i in component:
            group_details[i] = {'candidate_component_id': group_id, 'candidate_component_splits': '|'.join(component_splits),
                                'confirmed_group_id': all_rows[i][1].get('group_id', ''), 'quarantined': i in quarantine}
    with (destination/'group_audit.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = ['image_id','split','candidate_component_id','candidate_component_splits','confirmed_group_id','quarantined']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in sorted(group_details):
            writer.writerow({'image_id':i, 'split':all_rows[i][0], **group_details[i]})
    quarantine_rows = [{'image_id': i, 'split': all_rows[i][0], 'type': all_rows[i][1]['type'],
                        'style': all_rows[i][1]['style'], 'reason': 'unresolved_cross_split_candidate_component'}
                       for i in sorted(quarantine)]
    with (destination/'quarantine.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image_id','split','type','style','reason'])
        writer.writeheader()
        writer.writerows(quarantine_rows)
    shutil.copyfile(candidate_csv, destination/'candidate_evidence.csv')
    shutil.copyfile(audit_json, destination/'candidate_audit.json')
    if decisions_csv is not None:
        shutil.copyfile(decisions_csv, destination/'submitted_review_decisions.csv')
    with (destination/'near_duplicate_review.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image_id_a','split_a','image_id_b','split_b','distance','decision','quarantined_a','quarantined_b'])
        writer.writeheader()
        for pair in cross_before:
            a, b = pair['image_id_a'], pair['image_id_b']
            writer.writerow({**{k: pair[k] for k in ('image_id_a','split_a','image_id_b','split_b','distance')},
                             'decision': decisions.get(tuple(sorted((a,b))), '') or 'unreviewed',
                             'quarantined_a': a in quarantine, 'quarantined_b': b in quarantine})
    shutil.copyfile(source_root/'excluded_samples.json', destination/'source_exclusions.json')
    if extra_external is not None and (extra_external/'source_record.json').is_file():
        shutil.copyfile(extra_external/'source_record.json', destination/'recovered_external_source_record.json')
    metadata = {'dataset_protocol_version': PROTOCOL,
                'source': {'dataset_id': source_dataset_id, 'revision': source_revision},
                'policy': 'Conservative quarantine of every dHash candidate-connected component crossing training splits; no candidates declared identical.',
                'closed_set': 'mapping from train only; validation and synthetic_test must be known',
                'external_policy': 'reference only, never training or model selection; missing and unseen labels allowed',
                'candidate_decisions': {key: sum(decisions.get(tuple(sorted((p['image_id_a'], p['image_id_b']))), '') == key for p in candidates)
                                        for key in ('same', 'related', 'different', 'uncertain', '')},
                'training_audit': {'status': 'pass_conservative_quarantine', 'closed_set': True,
                                   'group_overlap_count': 0, 'exact_overlap_count': 0,
                                   'remaining_cross_split_candidate_count': len(remaining_cross),
                                   'unresolved_candidate_pairs_before_quarantine': sum(
                                       decisions.get(tuple(sorted((p['image_id_a'], p['image_id_b']))), '') in ('', 'uncertain') for p in cross_before),
                                   'human_review_complete': all(decisions.get(tuple(sorted((p['image_id_a'], p['image_id_b']))), '') in ('same', 'related', 'different') for p in cross_before)},
                'source_exclusions_recovered_as_external_reference': sorted(recovered),
                'counts': {s:len(rows) for s,rows in splits.items()},
                'class_counts': {s:{k:dict(sorted(Counter(r[k] for r in rows).items())) for k in mapping} for s,rows in splits.items()},
                'quarantine_counts': dict(Counter(r['split'] for r in quarantine_rows)),
                'class_mapping': mapping,
                'manifest_sha256': {s:sha(destination/f'{s}.csv') for s in splits},
                'split_hashes': {s:sha(destination/f'{s}.csv') for s in validated},
                'source_manifest_sha256': audit['manifest_sha256'],
                'candidate_evidence_sha256': sha(candidate_csv), 'candidate_audit_sha256': sha(audit_json),
                'review_decisions_sha256': sha(decisions_csv) if decisions_csv is not None else None,
                'quarantine_sha256': sha(destination/'quarantine.csv'),
                'near_duplicate_review_sha256': sha(destination/'near_duplicate_review.csv'),
                'group_audit_sha256': sha(destination/'group_audit.csv'),
                'training_group_and_exact_overlap': 'none; validate_splits recomputes image hashes',
                'external_exact_or_group_overlaps': external_overlaps}
    (destination/'dataset_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    (destination/'audit.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    return metadata


def validate_audited_snapshot(root):
    """Recheck frozen manifests and audit evidence before a real-data workflow."""
    root = Path(root)
    metadata = json.loads((root/'dataset_metadata.json').read_text(encoding='utf-8'))
    audit = metadata.get('training_audit', {})
    if (metadata.get('dataset_protocol_version') != PROTOCOL
            or audit.get('status') != 'pass_conservative_quarantine'
            or audit.get('closed_set') is not True
            or any(audit.get(k) != 0 for k in ('group_overlap_count','exact_overlap_count',
                                                'remaining_cross_split_candidate_count'))):
        raise ValueError('Snapshot training audit has not passed')
    for name in ('train','validation','synthetic_test','real_source_test'):
        if sha(root/f'{name}.csv') != metadata['manifest_sha256'][name]:
            raise ValueError(f'Snapshot manifest hash changed: {name}')
    for alias, canonical in (('val','validation'), ('test','synthetic_test')):
        if sha(root/f'{alias}.csv') != metadata['manifest_sha256'][canonical]:
            raise ValueError(f'Snapshot alias differs from {canonical}: {alias}')
    for filename, key in (('candidate_evidence.csv','candidate_evidence_sha256'),
                          ('candidate_audit.json','candidate_audit_sha256'),
                          ('quarantine.csv','quarantine_sha256'),
                          ('near_duplicate_review.csv','near_duplicate_review_sha256'),
                          ('group_audit.csv','group_audit_sha256')):
        if sha(root/filename) != metadata[key]:
            raise ValueError(f'Snapshot evidence hash changed: {filename}')
    if metadata.get('review_decisions_sha256') and sha(root/'submitted_review_decisions.csv') != metadata['review_decisions_sha256']:
        raise ValueError('Snapshot submitted decisions changed')
    splits, mapping = validate_splits(root)
    if any(row['source_split'] != 'train' for rows in splits.values() for row in rows):
        raise ValueError('Training snapshot contains an external source row')
    if any(row['source_split'] != 'test' for row in read_rows(root/'real_source_test.csv')):
        raise ValueError('External reference contains a non-test source row')
    if mapping != metadata['class_mapping']:
        raise ValueError('Training class mapping changed')
    if any(sha(root/f'{name}.csv') != expected for name, expected in metadata['split_hashes'].items()):
        raise ValueError('Training split hash changed')
    return splits, mapping, metadata
