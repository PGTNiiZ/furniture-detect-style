import json
import hashlib
from pathlib import Path

import pytest
import yaml

from furniture.config import digest, load_config
from furniture import suite


PROJECT = Path(__file__).resolve().parents[1]/'configs'/'project_controlled.yaml'


def plans_in(tmp_path):
    plans = suite.plan_suite(PROJECT, 'ablation', ['resnet18'])
    for index, plan in enumerate(plans):
        plan['config']['project_root'] = str(tmp_path)
        plan['config']['output_root'] = 'runs'
        plan['config']['dataset']['root'] = 'data/suite-fixture'
        plan['run_dir'] = str(tmp_path/'runs'/f'run-{index}')
    dataset = tmp_path/'data'/'suite-fixture'
    dataset.mkdir(parents=True, exist_ok=True)
    for split in ('train','val','test'):
        (dataset/f'{split}.csv').write_text(f'image_id\n{split}\n',encoding='utf-8')
    return plans


def test_screening_rejects_a_snapshot_without_completed_audit(tmp_path, monkeypatch):
    plans = plans_in(tmp_path)[:1]
    plans[0]['config']['experiment']['id'] = 'screening-resnet18-C3-42-type-style'
    monkeypatch.setattr(suite, 'train_run', lambda *args: pytest.fail('training must not start'))
    with pytest.raises(ValueError, match='requires an audited training snapshot'):
        suite.run_suite(plans, tmp_path/'screening.json')


def write_result(path, cfg, state='completed', checkpoint=False):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path/'status.json').write_text(json.dumps({'status':state,'augmentation_protocol':'post-p0-fix-v1'}))
    (path/'resolved_config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    (path/'provenance.json').write_text(json.dumps({'config_sha256':digest(cfg)}), encoding='utf-8')
    if state == 'completed':
        (path/'best.pt').write_bytes(b'checkpoint')
        (path/'checkpoint.sha256').write_text(hashlib.sha256(b'checkpoint').hexdigest())
        (path/'validation.json').write_text('{}')
    if checkpoint:
        (path/'last.pt').write_bytes(b'last-checkpoint')


def test_continue_and_retry_only_failed_run_with_fresh_attempt(tmp_path, monkeypatch):
    plans = plans_in(tmp_path)
    manifest = tmp_path/'suite.json'
    calls = []
    first_id = plans[0]['config']['experiment']['id']
    def fake_train(cfg, run):
        calls.append((cfg['experiment']['id'], str(run)))
        if cfg['experiment']['id'] == first_id and sum(x[0] == first_id for x in calls) == 1:
            write_result(run,cfg,'failed')
            raise RuntimeError('temporary failure')
        write_result(run,cfg)
    monkeypatch.setattr(suite, 'train_run', fake_train)
    result = suite.run_suite(plans, manifest)
    assert result['counts']['failed'] == 1 and result['counts']['completed'] == 1
    assert len(calls) == 2
    assert suite.run_suite(manifest_path=manifest)['counts']['failed'] == 1
    assert len(calls) == 2
    result = suite.run_suite(manifest_path=manifest, retry_failed=True)
    assert result['counts']['completed'] == 2 and len(calls) == 3
    assert calls[-1][0] == first_id and calls[-1][1] != calls[0][1]
    recorded = json.loads(manifest.read_text(encoding='utf-8'))['runs'][0]
    assert len(recorded['attempts']) == 2 and recorded['attempts'][0]['status'] == 'failed'
    suite.run_suite(manifest_path=manifest, retry_failed=True)
    assert len(calls) == 3


def test_fail_fast_preserves_unstarted_runs_for_later(tmp_path, monkeypatch):
    plans = plans_in(tmp_path)
    manifest = tmp_path/'suite.json'
    def fail(cfg, run):
        write_result(run,cfg,'failed')
        raise RuntimeError('stop here')
    monkeypatch.setattr(suite, 'train_run', fail)
    with pytest.raises(RuntimeError, match='stop here'):
        suite.run_suite(plans, manifest, fail_fast=True)
    saved = json.loads(manifest.read_text(encoding='utf-8'))
    assert [r['status'] for r in saved['runs']] == ['failed','planned']
    monkeypatch.setattr(suite, 'train_run', lambda cfg,run:write_result(run,cfg))
    result = suite.run_suite(manifest_path=manifest, retry_failed=True)
    assert result['counts']['completed'] == 2


def test_checkpoint_retry_and_explicit_running_recovery(tmp_path, monkeypatch):
    import furniture.pipeline as pipeline
    plans = plans_in(tmp_path)[:1]
    manifest = tmp_path/'suite.json'
    cfg = plans[0]['config']
    def fail(cfg,run):
        write_result(run,cfg,'failed',checkpoint=True)
        raise RuntimeError('after epoch')
    monkeypatch.setattr(suite, 'train_run', fail)
    suite.run_suite(plans, manifest)
    resumed = []
    def resume(run, recover_running=False):
        resumed.append((str(run), recover_running))
        write_result(run,cfg)
    monkeypatch.setattr(pipeline, 'resume_run', resume)
    result = suite.run_suite(manifest_path=manifest, retry_failed=True)
    assert result['counts']['completed'] == 1
    assert resumed == [(plans[0]['run_dir'], False)]
    assert result['runs'][0]['attempts'] == 1

    # A process that may still be live is never restarted without explicit recovery.
    other = tmp_path/'other'
    other.mkdir()
    plans = plans_in(other)[:1]
    other_manifest = other/'suite.json'
    write_result(plans[0]['run_dir'],plans[0]['config'],'running',checkpoint=True)
    assert suite.run_suite(plans, other_manifest)['counts']['running'] == 1
    assert len(resumed) == 1
    cfg = plans[0]['config']
    monkeypatch.setattr(pipeline, 'resume_run', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('owner active')))
    assert suite.run_suite(manifest_path=other_manifest,recover_running=True)['counts']['running'] == 1
    monkeypatch.setattr(pipeline, 'resume_run', resume)
    assert suite.run_suite(manifest_path=other_manifest,recover_running=True)['counts']['completed'] == 1
    assert resumed[-1] == (plans[0]['run_dir'], True)


def test_tampered_manifest_and_simultaneous_execution_are_rejected(tmp_path, monkeypatch):
    plans = plans_in(tmp_path)
    manifest = tmp_path/'suite.json'
    monkeypatch.setattr(suite, 'train_run', lambda cfg,run:write_result(run,cfg))
    suite.run_suite(plans, manifest)
    data = json.loads(manifest.read_text(encoding='utf-8'))
    data['runs'][0]['config']['training']['seed'] = 999
    manifest.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError,match='digest mismatch'):
        suite.run_suite(manifest_path=manifest)
    with suite._suite_lock(tmp_path/'locked.json.lock'):
        with pytest.raises(RuntimeError,match='already locked'):
            suite.run_suite(plans, tmp_path/'locked.json')


def test_frozen_dataset_stops_suite_before_next_run_and_before_continue(tmp_path, monkeypatch):
    plans = plans_in(tmp_path)
    manifest = tmp_path/'suite.json'
    train_manifest = tmp_path/'data'/'suite-fixture'/'train.csv'
    original = train_manifest.read_bytes()
    calls = []
    def train_then_change(cfg, run):
        calls.append(str(run))
        write_result(run, cfg)
        if len(calls) == 1:
            train_manifest.write_bytes(original + b'new-row\n')
    monkeypatch.setattr(suite, 'train_run', train_then_change)
    with pytest.raises(ValueError, match='split hashes changed'):
        suite.run_suite(plans, manifest)
    data = json.loads(manifest.read_text(encoding='utf-8'))
    assert len(data['dataset']['split_hashes']) == 3
    assert [item['status'] for item in data['runs']] == ['completed','planned']
    with pytest.raises(ValueError, match='split hashes changed'):
        suite.run_suite(manifest_path=manifest)
    assert len(calls) == 1
    train_manifest.write_bytes(original)
    assert suite.run_suite(manifest_path=manifest)['counts']['completed'] == 2
    assert len(calls) == 2


def test_restart_failed_keeps_unusable_checkpoint_and_does_not_resume(tmp_path, monkeypatch):
    import furniture.pipeline as pipeline
    plans = plans_in(tmp_path)[:1]
    manifest = tmp_path/'suite.json'
    calls = []
    def train(cfg,run):
        calls.append(str(run))
        if len(calls) == 1:
            write_result(run,cfg,'failed',checkpoint=True)
            raise RuntimeError('broken checkpoint')
        write_result(run,cfg)
    monkeypatch.setattr(suite, 'train_run', train)
    monkeypatch.setattr(pipeline, 'resume_run', lambda *args,**kwargs: pytest.fail('must not resume'))
    assert suite.run_suite(plans,manifest)['counts']['failed'] == 1
    old = Path(calls[0])
    evidence = (old/'last.pt').read_bytes(), (old/'status.json').read_bytes()
    result = suite.run_suite(manifest_path=manifest,restart_failed=True)
    assert result['counts']['completed'] == 1 and result['runs'][0]['attempts'] == 2
    assert calls[0] != calls[1]
    assert evidence == ((old/'last.pt').read_bytes(), (old/'status.json').read_bytes())


def test_orphan_without_checkpoint_requires_abandon_guard_before_new_attempt(tmp_path, monkeypatch):
    import furniture.pipeline as pipeline
    plans = plans_in(tmp_path)[:1]
    manifest = tmp_path/'suite.json'
    original = Path(plans[0]['run_dir'])
    write_result(original, plans[0]['config'], 'running')
    suite.run_suite(plans,manifest)
    abandoned, trained = [], []
    def abandon(run, expected_config_hash):
        abandoned.append((str(run), expected_config_hash))
        assert not (Path(run)/'last.pt').exists()
        write_result(run,plans[0]['config'],'interrupted')
    def train(cfg,run):
        trained.append(str(run))
        write_result(run,cfg)
    monkeypatch.setattr(pipeline, 'abandon_running', abandon)
    monkeypatch.setattr(suite, 'train_run', train)
    assert suite.run_suite(manifest_path=manifest,retry_failed=True)['counts']['running'] == 1
    assert suite.run_suite(manifest_path=manifest,restart_failed=True)['counts']['running'] == 1
    assert not abandoned and not trained
    result = suite.run_suite(manifest_path=manifest,retry_failed=True,recover_running=True)
    assert result['counts']['completed'] == 1 and result['runs'][0]['attempts'] == 2
    assert abandoned == [(str(original),digest(plans[0]['config']))]
    assert trained[0] != str(original) and (original/'status.json').exists()


def test_real_smoke_run_is_not_retrained_on_continue(tmp_path):
    import torch
    from furniture.smoke import make_smoke
    torch.set_num_threads(1)
    make_smoke(tmp_path/'data'/'smoke')
    cfg = load_config(PROJECT.parent/'project_smoke.yaml')
    cfg['project_root'] = str(tmp_path)
    cfg['output_root'] = 'runs'
    run = tmp_path/'runs'/'actual-run'
    manifest = tmp_path/'suite.json'
    initial = suite.run_suite([{'run_dir':str(run), 'config':cfg}], manifest)
    assert initial['counts']['completed'] == 1
    checkpoint = run/'best.pt'
    before = (checkpoint.stat().st_mtime_ns, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    result = suite.run_suite(manifest_path=manifest, retry_failed=True)
    after = (checkpoint.stat().st_mtime_ns, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    assert result['counts']['completed'] == 1 and result['runs'][0]['attempts'] == 1
    assert before == after
    checkpoint.write_bytes(b'corrupt')
    rejected = suite.run_suite(manifest_path=manifest, retry_failed=True)
    assert rejected['counts']['failed'] == 1 and checkpoint.read_bytes() == b'corrupt'
