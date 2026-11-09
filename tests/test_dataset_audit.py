import sys
import copy
import json
from types import SimpleNamespace

from PIL import Image
import pytest

from furniture.config import load_config, resolve_path
from furniture.data import prepare, read_rows, validate_splits, write_rows
from furniture.prepare import near_duplicate_candidates, prepare_data, audit_data


def test_dhash_candidates_are_bounded_and_not_automatic_groups():
    def row(name, value):
        return {'image_id':name, 'image_path':name+'.png', 'dhash':f'{value:016x}',
                'pixel_hash':name, 'group_id':''}
    splits = {'train':[row('a', 0),row('far', (1<<64)-1)], 'test':[row('b', 1)]}
    result = near_duplicate_candidates(splits, threshold=1)
    assert [(p['image_id_a'],p['image_id_b'],p['distance']) for p in result['pairs']] == [('a','b',1)]
    assert result['pairs'][0]['split_a'] == 'train' and result['pairs'][0]['split_b'] == 'test'
    assert not result['truncated']
    assert near_duplicate_candidates(splits, threshold=1, max_pairs=1)['truncated']
    with pytest.raises(ValueError, match='threshold'):
        near_duplicate_candidates(splits, threshold=9)


def test_huggingface_test_split_never_moves_to_train(tmp_path, monkeypatch):
    cfg = load_config('configs/project.yaml')
    cfg['project_root'] = str(tmp_path)
    cfg['dataset']['source']['revision'] = 'a'*40
    samples = lambda offset,n: [{'image':Image.new('RGB',(20,20),(i*5+offset,30,80)),
                                 'type':'chair','style':'modern'} for i in range(n)]
    monkeypatch.setitem(sys.modules,'datasets', SimpleNamespace(
        load_dataset=lambda *a,**k:{'train':samples(0,12), 'test':samples(150,3)}))
    report = prepare_data(cfg)
    root = resolve_path(cfg,cfg['dataset']['root'])
    assert all(row['source_split'] != 'test' for split in ('train','val') for row in read_rows(root/f'{split}.csv'))
    assert sum(row['source_split'] == 'test' for row in read_rows(root/'test.csv')) == 3
    assert report['split_hashes'] and (root/'near_duplicate_candidates.json').exists()
    assert (root/'dataset_metadata.json').exists()
    old = read_rows(root/'train.csv')
    old[0]['source_split'] = 'test'
    write_rows(root/'train.csv', old)
    with pytest.raises(ValueError, match='Source test row'):
        validate_splits(root)


def test_legacy_huggingface_prepare_preserves_test(tmp_path, monkeypatch):
    samples = lambda offset,n: [{'image':Image.new('RGB',(20,20),(i*5+offset,30,80)),
                                 'type':'chair','style':'modern'} for i in range(n)]
    monkeypatch.setitem(sys.modules,'datasets', SimpleNamespace(
        load_dataset=lambda *a,**k:{'train':samples(0,12), 'test':samples(150,3)}))
    root = tmp_path/'legacy'
    prepare(root, 'example', 'a'*40)
    assert all(row['source_split'] != 'test' for split in ('train','val') for row in read_rows(root/f'{split}.csv'))
    assert sum(row['source_split'] == 'test' for row in read_rows(root/'test.csv')) == 3


def test_incomplete_labels_require_opt_in_and_are_audited(tmp_path, monkeypatch):
    cfg = load_config('configs/project.yaml')
    cfg['project_root'] = str(tmp_path)
    cfg['dataset']['source']['revision'] = 'a'*40
    samples = lambda offset,n: [{'image':Image.new('RGB',(20,20),(i*5+offset,30,80)),
                                 'type':'chair','style':'modern'} for i in range(n)]
    train, test = samples(0,12), samples(150,3)
    test[1]['style'] = None
    monkeypatch.setitem(sys.modules,'datasets', SimpleNamespace(load_dataset=lambda *a,**k:{'train':train, 'test':test}))
    with pytest.raises(ValueError,match='import errors'):
        prepare_data(cfg)
    failed = resolve_path(cfg,cfg['dataset']['root'])
    errors = json.loads((failed/'import_errors.json').read_text(encoding='utf-8'))
    assert errors == [{'image_id':'test-0000001', 'source_split':'test', 'error':'Empty type/style label: style'}]
    assert not (failed/'train.csv').exists()

    cfg['dataset']['id'] = 'complete-labels-v2'
    cfg['dataset']['root'] = 'data/complete-labels-v2'
    report = prepare_data(cfg, exclude_incomplete_labels=True)
    root = resolve_path(cfg,cfg['dataset']['root'])
    exclusions = json.loads((root/'excluded_samples.json').read_text(encoding='utf-8'))
    assert exclusions['policy'] == 'exclude_incomplete_labels'
    assert exclusions['source_counts'] == {'train':12,'test':3}
    assert exclusions['excluded_count'] == 1
    assert exclusions['excluded_by_source_split'] == {'test':1}
    assert exclusions['excluded_by_label'] == {'style':1}
    assert exclusions['excluded'] == [{'image_id':'test-0000001','source_split':'test',
                                      'reason':'missing_required_label','missing_labels':['style']}]
    assert report['rows'] == 14 and sum(report['counts'].values()) == 14
    assert report['import_exclusions'] == exclusions
    assert json.loads((root/'dataset_metadata.json').read_text(encoding='utf-8'))['import_exclusions'] == exclusions
    assert audit_data(cfg)['import_exclusions'] == exclusions
    assert all(row['source_split'] != 'test' for split in ('train','val') for row in read_rows(root/f'{split}.csv'))
    assert sum(row['source_split'] == 'test' for row in read_rows(root/'test.csv')) == 2
    assert (failed/'import_errors.json').exists() and not (failed/'train.csv').exists()


@pytest.mark.parametrize('problem', ['invalid_image', 'missing_column', 'invalid_label_type', 'holdout_override'])
def test_opt_in_never_suppresses_other_import_errors(tmp_path, monkeypatch, problem):
    cfg = load_config('configs/project.yaml')
    cfg['project_root'] = str(tmp_path)
    cfg['dataset']['source']['revision'] = 'a'*40
    cfg['dataset']['source']['columns']['split'] = 'assigned_split'
    samples = lambda offset,n: [{'image':Image.new('RGB',(20,20),(i*5+offset,30,80)),
                                 'type':'chair','style':'modern','assigned_split':''} for i in range(n)]
    train, test = samples(0,12), samples(150,3)
    bad = test[0]
    if problem == 'invalid_image':
        bad['image'],bad['style'] = 'not-an-image',None
    elif problem == 'missing_column':
        del bad['style']
    elif problem == 'invalid_label_type':
        bad['style'] = 42
    else:
        bad['style'],bad['assigned_split'] = '', 'train'
    monkeypatch.setitem(sys.modules,'datasets', SimpleNamespace(load_dataset=lambda *a,**k:{'train':train, 'test':test}))
    with pytest.raises(ValueError,match='import errors'):
        prepare_data(cfg,exclude_incomplete_labels=True)
    root = resolve_path(cfg,cfg['dataset']['root'])
    errors = json.loads((root/'import_errors.json').read_text(encoding='utf-8'))
    assert len(errors) == 1 and errors[0]['image_id'] == 'test-0000000'
    assert not (root/'train.csv').exists()
