import csv
import json
from pathlib import Path

import pytest
import torch
import yaml

from furniture.reports import generate_report


def _create_run(root, seed, val_score, test_score):
    root.mkdir()
    mapping = {'type':{'chair':0, 'table':1}, 'style':{'modern':0, 'rustic':1}}
    cfg = {'dataset':{'id':'fixture-v1'}, 'model':{'backbone':'resnet18'},
           'training':{'seed':seed, 'epochs':2, 'freeze_epochs':0},
           'augmentation':{'preset':'none'}, 'experiment':{'id':'fixture'}}
    metrics = lambda score: {key:{'accuracy':score, 'macro_f1':score, 'weighted_f1':score,
                                 'confusion_matrix':[[1, 0], [1, 0]]} for key in mapping}
    validation, test = metrics(val_score), metrics(test_score)
    checkpoint = {'mapping':mapping, 'split_hashes':{'train':'x', 'val':'y', 'test':'z'},
                  'val':validation, 'epoch':1, 'shared_backbone_update_epochs':1}
    torch.save(checkpoint, root/'best.pt')
    (root/'status.json').write_text(json.dumps({'status':'completed', 'augmentation_protocol':'post-p0-fix-v1'}))
    (root/'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
    (root/'validation.json').write_text(json.dumps(validation))
    (root/'test.json').write_text(json.dumps(test))
    for split in ('validation', 'test'):
        (root/f'{split}_predictions.csv').write_text(
            'image_id,image_path,true_type,pred_type,true_style,pred_style\n'
            'one,one.png,chair,chair,modern,rustic\n', encoding='utf-8')
    history = [{'epoch':i, 'train':{'loss':1/i, **validation}, 'val':{'loss':2/i, **validation}} for i in (1, 2)]
    (root/'history.json').write_text(json.dumps(history))
    return root


def test_report_uses_recorded_split_and_independent_seeds(tmp_path):
    pytest.importorskip('matplotlib')
    run_a = _create_run(tmp_path/'a', 42, .2, .3)
    run_b = _create_run(tmp_path/'b', 123, .6, .5)
    result = generate_report([run_a, run_b], tmp_path/'val-report', vary=('seed',))
    assert result['split'] == 'validation'
    assert result['repeated_seed_summary'][0]['macro_f1_mean'] == pytest.approx(.4)
    assert result['repeated_seed_summary'][0]['macro_f1_sample_std'] == pytest.approx(.2828427125)
    assert (tmp_path/'val-report/run-001-curves.png').is_file()
    assert (tmp_path/'val-report/run-001-type-confusion.png').is_file()
    with (tmp_path/'val-report/results.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and all(row['split'] == 'validation' for row in rows)
    with (tmp_path/'val-report/examples.csv').open(encoding='utf-8', newline='') as stream:
        examples = list(csv.DictReader(stream))
    assert len(examples) == 4 and {row['correct'] for row in examples} == {'True', 'False'}
    test_result = generate_report([run_a, run_b], tmp_path/'test-report', split='test', vary=('seed',))
    assert test_result['repeated_seed_summary'][0]['macro_f1_mean'] == pytest.approx(.4)
    assert {row['accuracy'] for row in test_result['rows']} == {.3, .5}
    with pytest.raises(ValueError, match='different'):
        generate_report([run_a, run_b], tmp_path/'uncontrolled')
    (run_b/'status.json').write_text(json.dumps({'status':'failed'}))
    with pytest.raises(ValueError, match='completed'):
        generate_report([run_a, run_b], tmp_path/'failed', vary=('seed',))


def test_report_does_not_summarize_duplicate_seeds_or_missing_test(tmp_path):
    pytest.importorskip('matplotlib')
    run_a = _create_run(tmp_path/'a', 42, .2, .3)
    run_b = _create_run(tmp_path/'b', 42, .6, .5)
    result = generate_report([run_a, run_b], tmp_path/'report')
    assert result['repeated_seed_summary'] == []
    assert 'Duplicate seeds' in result['skipped_summaries'][0]['reason']
    (run_b/'test.json').unlink()
    with pytest.raises(ValueError, match='test evaluation is missing'):
        generate_report([run_a, run_b], tmp_path/'missing', split='test')
    assert not (tmp_path/'missing').exists()
