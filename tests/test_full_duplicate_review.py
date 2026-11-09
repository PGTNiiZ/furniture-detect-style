import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image


def test_gallery_covers_all_cross_splits_and_quarantine_is_only_a_proposal(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'create_full_duplicate_review.py'
    spec = importlib.util.spec_from_file_location('full_review', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / 'data'
    (root / 'images').mkdir(parents=True)
    hashes = {}
    for split, ids in {'train': ['a', 'd'], 'val': ['b'], 'test': ['c', 'external']}.items():
        with (root / f'{split}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, ['image_id', 'image_path', 'type', 'style', 'source_split', 'pixel_hash', 'group_id'])
            writer.writeheader()
            for image_id in ids:
                Image.new('RGB', (5, 5)).save(root / 'images' / f'{image_id}.png')
                writer.writerow({'image_id': image_id, 'image_path': f'images/{image_id}.png', 'type': 'chair', 'style': 'modern',
                                 'source_split': 'test' if image_id == 'external' else 'train', 'pixel_hash': image_id})
        hashes[split] = hashlib.sha256((root / f'{split}.csv').read_bytes()).hexdigest()
    candidate = tmp_path / 'pairs.csv'
    with candidate.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, ['image_id_a', 'split_a', 'image_id_b', 'split_b', 'distance', 'exact_pixels'])
        writer.writeheader()
        for a, sa, b, sb in [('a', 'train', 'b', 'val'), ('a', 'train', 'c', 'test'),
                              ('b', 'val', 'c', 'test'), ('a', 'train', 'd', 'train')]:
            writer.writerow({'image_id_a': a, 'split_a': sa, 'image_id_b': b, 'split_b': sb, 'distance': 2, 'exact_pixels': 'False'})
    candidate.with_suffix('.json').write_text(json.dumps({'source_split': 'train', 'truncated': False,
        'manifest_sha256': hashes, 'candidate_pairs': 4, 'cross_split_pairs': 3}), encoding='utf-8')
    output = tmp_path / 'review'
    report = module.create(root, candidate, output)
    assert report['candidate_count'] == 3
    assert report['quarantine_images'] == 4  # d is connected through a same-split candidate
    assert report['status'] == 'proposal_only_not_applied'
    page = (output / 'index.html').read_text(encoding='utf-8')
    assert 'external' not in page and 'value="related"' in page and 'Download decisions CSV' in page
    with (output / 'candidates-unreviewed.csv').open(encoding='utf-8', newline='') as stream:
        pairs = list(csv.DictReader(stream))
    assert {p['decision'] for p in pairs} == {'unreviewed'}
    assert {p['split_b'] for p in pairs} == {'validation', 'synthetic_test'}
    assert len(list((output / 'images').glob('*.jpg'))) == 3
    with pytest.raises(ValueError, match='exists'):
        module.create(root, candidate, output)
    invalid = tmp_path / 'invalid.csv'
    invalid.write_text(candidate.read_text(encoding='utf-8').replace('a,train,b,val', 'external,train,b,val', 1), encoding='utf-8')
    invalid.with_suffix('.json').write_text(candidate.with_suffix('.json').read_text(encoding='utf-8'), encoding='utf-8')
    with pytest.raises(ValueError, match='source-train'):
        module.create(root, invalid, tmp_path / 'invalid-review')
    (root / 'val.csv').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='differs'):
        module.create(root, candidate, tmp_path / 'stale')
