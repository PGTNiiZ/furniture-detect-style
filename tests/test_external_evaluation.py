import csv
from copy import deepcopy
import pytest
import torch
from PIL import Image

from furniture.data import image_hash, write_rows
from furniture.external_evaluation import _check_exposure, create_external_protocol, evaluate_external


def sample(root, name, source, type_label, style, color):
    image = Image.new('RGB', (8, 8), color)
    image.save(root / (name + '.png'))
    return {'image_id': name, 'image_path': name + '.png', 'source_split': source,
            'type': type_label, 'style': style, 'pixel_hash': image_hash(image), 'dhash': '', 'group_id': ''}


def test_closed_set_external_protocol_and_metrics(tmp_path, monkeypatch):
    train = [sample(tmp_path, 'tr1', 'train', 'chair', 'modern', 'red'),
             sample(tmp_path, 'tr2', 'train', 'table', 'classic', 'blue')]
    external = [sample(tmp_path, 'a', 'test', 'chair', 'modern', 'black'),
                sample(tmp_path, 'b', 'test', 'unknown', 'modern', 'green'),
                sample(tmp_path, 'c', 'test', 'table', '', 'white'),
                sample(tmp_path, 'd', 'test', '', 'alien', 'yellow')]
    train_path, external_path = tmp_path / 'train.csv', tmp_path / 'external.csv'
    write_rows(train_path, train)
    write_rows(external_path, external)
    protocol_path = tmp_path / 'protocol.json'
    protocol = create_external_protocol(train_path, external_path, protocol_path)
    assert protocol['annotation_inspection']['counts'] == {
        'type': {'known': 2, 'missing': 1, 'unseen': 1},
        'style': {'known': 2, 'missing': 1, 'unseen': 1}}
    assert protocol['annotation_inspection']['known_both'] == 1
    assert not (tmp_path / 'result').exists()
    with pytest.raises(FileExistsError):
        create_external_protocol(train_path, external_path, protocol_path)

    class Model:
        def __call__(self, tensor):
            return {'type': torch.tensor([[1., 0.]]), 'style': torch.tensor([[0., 1.]])}

    import hashlib
    payload = {'mapping': protocol['class_mapping'], 'config': {'image_size': 8},
               'split_hashes': {'train': hashlib.sha256(train_path.read_bytes()).hexdigest()},
               'initialization': {'mode': 'scratch'},
               'exposure_membership': {s: {'pixel_hashes': [], 'group_ids': []} for s in ('train', 'val', 'test')}}
    payload['exposure_membership']['train']['pixel_hashes'] = [row['pixel_hash'] for row in train]
    payload['data_membership'] = deepcopy(payload['exposure_membership'])
    monkeypatch.setattr('furniture.external_evaluation.load_model', lambda *args: (Model(), payload))
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'mock checkpoint')
    payload['split_hashes']['train'] = 'different'
    with pytest.raises(ValueError, match='not bound'):
        evaluate_external(protocol_path, train_path, external_path, tmp_path, checkpoint, tmp_path / 'bad', 'cpu')
    payload['split_hashes']['train'] = hashlib.sha256(train_path.read_bytes()).hexdigest()
    report = evaluate_external(protocol_path, train_path, external_path, tmp_path, checkpoint, tmp_path / 'result', 'cpu')
    assert report['tasks']['type']['eligible'] == 2
    assert report['tasks']['style']['coverage'] == .5
    assert report['tasks']['type']['metrics']['accuracy'] == .5
    assert report['joint'] == {'eligible': 1, 'total': 4, 'coverage': .25, 'correct': 1, 'accuracy': 1.0}
    with (tmp_path / 'result' / 'missing_or_unseen.csv').open(encoding='utf-8', newline='') as stream:
        assert {row['image_id'] for row in csv.DictReader(stream)} == {'b', 'c', 'd'}
    with pytest.raises(ValueError, match='Output already exists'):
        evaluate_external(protocol_path, train_path, external_path, tmp_path, checkpoint, tmp_path / 'result', 'cpu')
    external[0]['type'] = 'other'
    write_rows(external_path, external)
    with pytest.raises(ValueError, match='locked inspection'):
        evaluate_external(protocol_path, train_path, external_path, tmp_path, checkpoint, tmp_path / 'new', 'cpu')


def test_external_protocol_rejects_leakage_and_wrong_source(tmp_path):
    row = sample(tmp_path, 'tr', 'train', 'chair', 'modern', 'red')
    train, external = tmp_path / 'train.csv', tmp_path / 'external.csv'
    write_rows(train, [row])
    other = {**row, 'image_id': 'test', 'source_split': 'test'}
    write_rows(external, [other])
    with pytest.raises(ValueError, match='overlap in pixel_hash'):
        create_external_protocol(train, external, tmp_path / 'protocol.json')
    other['pixel_hash'] = 'different'
    other['source_split'] = 'train'
    write_rows(external, [other])
    with pytest.raises(ValueError, match='source test'):
        create_external_protocol(train, external, tmp_path / 'protocol.json')


def test_exposure_guard_fails_closed_and_checks_ancestors(tmp_path):
    rows = [{'image_id': 'external', 'pixel_hash': 'external-hash', 'group_id': 'external-group'}]
    exposure = {s: {'pixel_hashes': [], 'group_ids': []} for s in ('train', 'val', 'test')}
    payload = {'exposure_membership': exposure, 'data_membership': deepcopy(exposure), 'initialization': {'mode': 'scratch'}}
    _check_exposure(payload, rows)
    for split, field, value in [('train', 'pixel_hashes', 'external-hash'),
                                 ('val', 'group_ids', 'external-group'),
                                 ('test', 'image_ids', 'external')]:
        exposure[split][field] = [value]
        with pytest.raises(ValueError, match='overlaps checkpoint'):
            _check_exposure(payload, rows)
        exposure[split][field] = []
    with pytest.raises(ValueError, match='exposure membership'):
        _check_exposure({'initialization': {'mode': 'scratch'}}, rows)
    with pytest.raises(ValueError, match='initialization provenance'):
        _check_exposure({'exposure_membership': exposure, 'data_membership': deepcopy(exposure)}, rows)
    parent = tmp_path / 'parent.pt'
    torch.save(payload, parent)
    import hashlib
    child = {'exposure_membership': exposure, 'data_membership': deepcopy(exposure), 'initialization': {'mode': 'warm_start',
             'parent_checkpoint': str(parent), 'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest()}}
    _check_exposure(child, rows)
    torch.save({'initialization': {'mode': 'scratch'}}, parent)
    with pytest.raises(ValueError, match='hash differs'):
        _check_exposure(child, rows)
    child['initialization']['parent_sha256'] = hashlib.sha256(parent.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='exposure membership'):
        _check_exposure(child, rows)
    parent.unlink()
    with pytest.raises(ValueError, match='unavailable'):
        _check_exposure(child, rows)
