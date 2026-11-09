import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from furniture.data import dhash, image_hash, write_rows
from furniture.snapshot import build_snapshot as _build_snapshot, sha


def build_snapshot(*args, **kwargs):
    return _build_snapshot(*args, source_dataset_id='test/source', source_revision='a'*40, **kwargs)


def fixture_snapshot(tmp_path):
    root = tmp_path / 'source'
    (root / 'images').mkdir(parents=True)
    all_splits = {'train': [], 'val': [], 'test': []}
    for idx, (identifier, split) in enumerate([('a','train'),('b','train'),('c','train'),('d','val'),
                                                ('x','train'),('y','val'),('z','test'),('external','test')]):
        image = Image.new('RGB', (2,2), (idx * 20, 0, 0))
        image.save(root / 'images' / f'{identifier}.png')
        row = {'image_id':identifier, 'image_path':f'images/{identifier}.png', 'type':'chair', 'style':'modern',
               'source_split':'test' if identifier == 'external' else 'train',
               'pixel_hash':image_hash(image), 'dhash':f'{idx:016x}', 'group_id':''}
        if identifier == 'external':
            row['style'] = 'unknown-source-style'
        all_splits[split].append(row)
    for split, rows in all_splits.items():
        write_rows(root/f'{split}.csv', rows)
    (root/'excluded_samples.json').write_text(json.dumps({'excluded': []}), encoding='utf-8')
    candidates = tmp_path/'candidates.csv'
    with candidates.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image_id_a','split_a','image_id_b','split_b','distance'])
        w.writeheader()
        w.writerows([{'image_id_a':'a','split_a':'train','image_id_b':'b','split_b':'train','distance':1},
                     {'image_id_a':'b','split_a':'train','image_id_b':'c','split_b':'train','distance':2},
                     {'image_id_a':'c','split_a':'train','image_id_b':'d','split_b':'val','distance':1}])
    audit = tmp_path/'audit.json'
    audit.write_text(json.dumps({'truncated':False,'threshold':4,'rows':7,'examined_pairs':21,
                                 'candidate_pairs':3, 'manifest_sha256':{s:sha(root/f'{s}.csv') for s in all_splits}}))
    return root, candidates, audit


def test_conservative_quarantine_and_external_unknown(tmp_path):
    root, candidates, audit = fixture_snapshot(tmp_path)
    out = tmp_path/'snapshot'
    result = build_snapshot(root, out, candidates, audit)
    assert result['counts'] == {'train':1,'validation':1,'synthetic_test':1,'real_source_test':1}
    assert result['class_mapping']['style'] == {'modern':0}
    assert result['quarantine_counts'] == {'train':3,'validation':1}
    assert result['training_audit']['unresolved_candidate_pairs_before_quarantine'] == 1
    assert result['training_audit']['remaining_cross_split_candidate_count'] == 0
    assert result['training_audit']['human_review_complete'] is False
    assert (out/'val.csv').read_bytes() == (out/'validation.csv').read_bytes()
    assert (out/'test.csv').read_bytes() == (out/'synthetic_test.csv').read_bytes()
    assert 'unknown-source-style' in (out/'real_source_test.csv').read_text()
    with pytest.raises(ValueError, match='destination exists'):
        build_snapshot(root, out, candidates, audit)


def test_confirmed_groups_transitive_and_different_can_stay_split(tmp_path):
    root, candidates, audit = fixture_snapshot(tmp_path)
    decisions = tmp_path/'decisions.csv'
    decisions.write_text('image_id_a,image_id_b,decision\na,b,same\nb,c,related\nc,d,different\n', encoding='utf-8')
    out = tmp_path/'snapshot'
    result = build_snapshot(root, out, candidates, audit, decisions_csv=decisions)
    rows = list(csv.DictReader((out/'train.csv').open()))
    groups = {r['group_id'] for r in rows if r['image_id'] in ('a','b','c')}
    assert len(groups) == 1 and '' not in groups
    assert result['training_audit']['human_review_complete']
    assert result['training_audit']['unresolved_candidate_pairs_before_quarantine'] == 0
    assert result['quarantine_counts'] == {}


def test_changed_source_or_invalid_decisions_are_rejected(tmp_path):
    root, candidates, audit = fixture_snapshot(tmp_path)
    decisions = tmp_path/'decisions.csv'
    decisions.write_text('image_id_a,image_id_b,decision\na,b,yes\n', encoding='utf-8')
    with pytest.raises(ValueError, match='review decision'):
        build_snapshot(root, tmp_path/'one', candidates, audit, decisions_csv=decisions)
    (root/'train.csv').write_text('changed')
    with pytest.raises(ValueError, match='manifest changed'):
        build_snapshot(root, tmp_path/'two', candidates, audit)


def test_missing_external_row_kept_without_entering_mapping(tmp_path):
    root, candidates, audit = fixture_snapshot(tmp_path)
    extra = tmp_path/'extra'
    (extra/'images').mkdir(parents=True)
    image = Image.new('RGB', (2,2), (1,2,3))
    image.save(extra/'images/missing.png')
    row = {'image_id':'missing', 'image_path':'images/missing.png', 'type':'', 'style':'',
           'source_split':'test', 'pixel_hash':image_hash(image), 'dhash':dhash(image), 'group_id':''}
    write_rows(extra/'rows.csv', [row])
    (root/'excluded_samples.json').write_text(json.dumps({'excluded':[{'image_id':'missing'}]}))
    out = tmp_path/'snapshot'
    result = build_snapshot(root, out, candidates, audit, extra_external=extra)
    assert result['counts']['real_source_test'] == 2
    assert result['source_exclusions_recovered_as_external_reference'] == ['missing']
    assert '' not in result['class_mapping']['type']
    assert list(csv.DictReader((out/'real_source_test.csv').open()))[-1]['type'] == ''


def test_external_exact_overlap_fails_before_creating_snapshot(tmp_path):
    root, candidates, audit = fixture_snapshot(tmp_path)
    extra = tmp_path/'extra'
    (extra/'images').mkdir(parents=True)
    image = Image.open(root/'images/x.png')
    image.save(extra/'images/duplicate.png')
    row = {'image_id':'external-duplicate', 'image_path':'images/duplicate.png', 'type':'', 'style':'',
           'source_split':'test', 'pixel_hash':image_hash(image), 'dhash':dhash(image), 'group_id':''}
    write_rows(extra/'rows.csv', [row])
    out = tmp_path/'snapshot'
    with pytest.raises(ValueError, match='External reference overlaps'):
        build_snapshot(root, out, candidates, audit, extra_external=extra)
    assert not out.exists()


def test_inconsistent_decisions_are_rejected(tmp_path):
    root, candidates, audit = fixture_snapshot(tmp_path)
    with candidates.open('a', encoding='utf-8', newline='') as stream:
        csv.DictWriter(stream, fieldnames=['image_id_a','split_a','image_id_b','split_b','distance']).writerow(
            {'image_id_a':'a','split_a':'train','image_id_b':'c','split_b':'train','distance':1})
    report = json.loads(audit.read_text())
    report['candidate_pairs'] = 4
    audit.write_text(json.dumps(report))
    decisions = tmp_path/'decisions.csv'
    decisions.write_text('image_id_a,image_id_b,decision\na,b,same\nb,c,related\na,c,different\n')
    with pytest.raises(ValueError, match='contradicts'):
        build_snapshot(root, tmp_path/'snapshot', candidates, audit, decisions_csv=decisions)


def test_unreviewed_export_is_accepted_and_path_escape_rejected(tmp_path):
    root, candidates, audit = fixture_snapshot(tmp_path)
    decisions = tmp_path/'decisions.csv'
    decisions.write_text('image_id_a,image_id_b,decision\nc,d,unreviewed\n')
    result = build_snapshot(root, tmp_path/'ok', candidates, audit, decisions_csv=decisions)
    assert result['training_audit']['unresolved_candidate_pairs_before_quarantine'] == 1
    extra = tmp_path/'external'
    extra.mkdir()
    row = {'image_id':'evil', 'image_path':'../outside.png', 'type':'', 'style':'',
           'source_split':'test', 'pixel_hash':'bad', 'dhash':'0'*16, 'group_id':''}
    write_rows(extra/'rows.csv', [row])
    with pytest.raises(ValueError, match='escapes'):
        build_snapshot(root, tmp_path/'reject', candidates, audit, extra_external=extra)
    assert not (tmp_path/'reject').exists()


def test_source_revision_and_image_path_collision_rejected(tmp_path):
    root, candidates, audit = fixture_snapshot(tmp_path)
    with pytest.raises(ValueError, match='revision'):
        _build_snapshot(root, tmp_path/'bad', candidates, audit, source_dataset_id='source', source_revision='not-pinned')
    extra = tmp_path/'external'
    extra.mkdir()
    row = {'image_id':'duplicate-path', 'image_path':'images/x.png', 'type':'', 'style':'',
           'source_split':'test', 'pixel_hash':'bad', 'dhash':'0'*16, 'group_id':''}
    write_rows(extra/'rows.csv', [row])
    with pytest.raises(ValueError, match='collide'):
        build_snapshot(root, tmp_path/'reject', candidates, audit, extra_external=extra)


def test_audited_snapshot_detects_tampering(tmp_path):
    from furniture.snapshot import validate_audited_snapshot
    root, candidates, audit = fixture_snapshot(tmp_path)
    out = tmp_path/'snapshot'
    build_snapshot(root, out, candidates, audit)
    splits, mapping, metadata = validate_audited_snapshot(out)
    assert set(splits) == {'train','val','test'}
    assert metadata['training_audit']['human_review_complete'] is False
    (out/'candidate_evidence.csv').write_text('modified')
    with pytest.raises(ValueError, match='evidence hash changed'):
        validate_audited_snapshot(out)


def test_audited_snapshot_checks_alias_and_source_provenance(tmp_path):
    from furniture.snapshot import validate_audited_snapshot
    root, candidates, audit = fixture_snapshot(tmp_path)
    out = tmp_path/'snapshot'
    build_snapshot(root, out, candidates, audit)
    (out/'test.csv').write_text('modified')
    with pytest.raises(ValueError, match='alias'):
        validate_audited_snapshot(out)
