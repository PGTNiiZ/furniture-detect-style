import csv
import json
from pathlib import Path

import pytest
import torch

from scripts import release_evidence as evidence
from scripts.record_combined_tests import bind, capture


@pytest.mark.parametrize('stage', ['synthetic-test', 'external-reference', 'synthetic-test-report', 'validation-report'])
def test_interrupted_report_is_recovered_without_deleting_attempt(tmp_path, stage):
    target = tmp_path/stage
    context = {'checkpoint_sha256':'checkpoint-a', 'split_hashes':{'test':'split-a'}}
    def interrupted(path):
        (path/'part.txt').write_text('partial')
        raise RuntimeError('simulated crash')
    with pytest.raises(RuntimeError, match='simulated crash'):
        evidence.bundle(target, context, ['result.txt'], interrupted)
    assert not target.exists()
    attempts = list((tmp_path/'_attempts').iterdir())
    assert len(attempts) == 1 and (attempts[0]/'part.txt').read_text() == 'partial'
    evidence.bundle(target, context, ['result.txt'], lambda path:(path/'result.txt').write_text('complete'))
    assert (target/'result.txt').read_text() == 'complete'
    assert (attempts[0]/'part.txt').is_file()
    evidence.bundle(target, context, ['result.txt'], lambda path:pytest.fail('completed report must not rerun'))
    actions = {json.loads(p.read_text(encoding='utf-8'))['action'] for p in (tmp_path/'recovery-events').glob('*.json')}
    assert {'attempt_interrupted','recovery_started','published','reused_verified'} <= actions


@pytest.mark.parametrize('changed', [
    {'checkpoint_sha256':'checkpoint-b', 'split_hashes':{'test':'split-a'}},
    {'checkpoint_sha256':'checkpoint-a', 'split_hashes':{'test':'split-b'}},
])
def test_rejects_reuse_for_different_checkpoint_or_split(tmp_path, changed):
    target = tmp_path/'evaluation'
    context = {'checkpoint_sha256':'checkpoint-a', 'split_hashes':{'test':'split-a'}}
    evidence.bundle(target, context, ['metrics.json'], lambda p:(p/'metrics.json').write_text('{}'))
    with pytest.raises(ValueError, match='checkpoint, split or inputs differ'):
        evidence.bundle(target, changed, ['metrics.json'], lambda p:None)
    assert (target/'metrics.json').read_text() == '{}'


def test_rejects_changed_output_even_if_manifest_exists(tmp_path):
    target = tmp_path/'evaluation'
    evidence.bundle(target, {'selected':'one'}, ['metrics.json'], lambda p:(p/'metrics.json').write_text('{}'))
    (target/'metrics.json').write_text('{"altered":true}')
    with pytest.raises(ValueError, match='missing or changed'):
        evidence.validate_bundle(target, {'selected':'one'}, ['metrics.json'])


def test_run_contract_rejects_checkpoint_split_and_mapping_mismatch(tmp_path, monkeypatch):
    run = tmp_path/'run'
    run.mkdir()
    (run/'best.pt').write_bytes(b'checkpoint')
    metric = {'accuracy':.4,'macro_precision':.4,'macro_recall':.4,'macro_f1':.4,'weighted_f1':.4}
    (run/'validation.json').write_text(json.dumps({'type':metric}))
    item = {'config':{'training':{'tasks':['type']}}, 'config_sha256':'config', 'attempts':[{'run_dir':str(run)}]}
    snapshot = {'split_hashes':{'train':'a','val':'b','test':'c'}, 'class_mapping':{'type':{'chair':0}}}
    payload = {'mapping':{'type':{'chair':0}}, 'split_hashes':dict(snapshot['split_hashes']), 'val':{'type':metric}, 'config':{'seed':42}}
    monkeypatch.setattr(evidence, 'training_settings', lambda cfg:{'seed':42})
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs:payload)
    evidence.validate_run(item, snapshot)
    payload['split_hashes']['test'] = 'other'
    with pytest.raises(ValueError, match='split hashes'):
        evidence.validate_run(item, snapshot)
    payload['split_hashes']['test'] = 'c'
    payload['mapping']['type'] = {'sofa':0}
    with pytest.raises(ValueError, match='class mapping'):
        evidence.validate_run(item, snapshot)


def test_repeated_seed_summary_uses_all_three_and_rejects_missing(tmp_path):
    items = []
    for condition in ('C1','C2','C3','C5'):
        for seed in (42,123,2026):
            run = tmp_path/f'{condition}-{seed}'
            run.mkdir()
            value = {42:.3,123:.6,2026:.9}[seed]
            (run/'validation.json').write_text(json.dumps({t:{'macro_f1':value} for t in ('type','style')}))
            cfg = {'experiment':{'id':f'repeated-resnet18-{condition}-{seed}-type-style'},
                   'model':{'backbone':'resnet18'}, 'training':{'seed':seed,'epochs':30},
                   'augmentation':{'condition':condition}, 'dataset':{'id':'same'}}
            items.append({'config':cfg, 'attempts':[{'run_dir':str(run)}]})
    output = tmp_path/'repeated.csv'
    evidence.repeated_summary(items, 'resnet18', output)
    with output.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert all(row['seeds'] == '42,123,2026' and row['n_seeds'] == '3' for row in rows)
    assert float(rows[0]['macro_f1_mean']) == pytest.approx(.6)
    assert float(rows[0]['macro_f1_sample_std']) == pytest.approx(.3)
    with pytest.raises(ValueError, match='Incomplete repeated seeds'):
        evidence.repeated_summary(items[:-1], 'resnet18', tmp_path/'incomplete.csv')


def test_controlled_contrasts_check_actual_protocol_controls(tmp_path):
    from furniture.suite import plan_suite
    controlled_plans = plan_suite('configs/project_v2_conservative.yaml', 'controlled', ['resnet18'])
    ablation_plans = plan_suite('configs/project_v2_conservative.yaml', 'ablation', ['resnet18'])
    items = []
    for index, plan in enumerate(controlled_plans + ablation_plans):
        run = tmp_path/str(index)
        run.mkdir()
        (run/'validation.json').write_text(json.dumps({'type':{'macro_f1':.5}, 'style':{'macro_f1':.25}}))
        items.append({'config':plan['config'], 'attempts':[{'run_dir':str(run)}]})
    path = tmp_path/'contrasts.csv'
    evidence.controlled_contrasts(items[:8], items[8:], 'resnet18', path)
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16
    assert {row['axis'] for row in rows} == {'training_mode','augmentation','occlusion','tasks'}
    items[4]['config']['training']['batch_size'] += 1
    with pytest.raises(ValueError, match='controls differ'):
        evidence.controlled_contrasts(items[:8], items[8:], 'resnet18', tmp_path/'mismatch.csv')


@pytest.mark.parametrize('encoding', ['utf-8', 'utf-16'])
def test_combined_tests_bound_to_same_code_before_and_after(tmp_path, encoding):
    for name in ('src/furniture','scripts','tests','configs','notebooks'):
        (tmp_path/name).mkdir(parents=True)
    source = tmp_path/'scripts/model.py'
    source.write_text('version one')
    (tmp_path/'pyproject.toml').write_text('[build-system]')
    before, log, binding = tmp_path/'before.json', tmp_path/'pytest.txt', tmp_path/'binding.json'
    capture(tmp_path, before)
    log.write_text('5 passed, 2 warnings in 1.25s', encoding=encoding)
    bind(tmp_path, before, log, binding, 0)
    assert evidence.validate_test_binding(tmp_path, binding)['passed'] == 5
    source.write_text('version two')
    with pytest.raises(ValueError, match='not bound'):
        evidence.validate_test_binding(tmp_path, binding)
    with pytest.raises(ValueError, match='changed while'):
        bind(tmp_path, before, log, tmp_path/'different.json', 0)
