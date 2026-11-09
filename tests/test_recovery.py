import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from furniture.pipeline import _run_lock, abandon_running, resume_run
from furniture.smoke import make_smoke


PROJECT = Path(__file__).resolve().parents[1]


def test_orphaned_run_recovery_checks_owner_lock_checkpoint_and_provenance(tmp_path):
    torch.set_num_threads(1)
    make_smoke(tmp_path/'data'/'smoke')
    run = tmp_path/'orphan'
    code = r'''
import os, sys, torch
sys.path.insert(0,sys.argv[4])
import furniture.train as train
from furniture.config import load_config
from furniture.pipeline import train_run
torch.set_num_threads(1)
cfg=load_config(sys.argv[1])
cfg['project_root']=sys.argv[2]
cfg['training']['epochs']=2
cfg['training']['patience']=2
original=train.run_epoch
calls=0
def crash_after_first_epoch(*args, **kwargs):
    global calls
    calls += 1
    if calls == 3:
        os._exit(37)
    return original(*args, **kwargs)
train.run_epoch=crash_after_first_epoch
train_run(cfg,sys.argv[3])
'''
    env = {**os.environ, 'OMP_NUM_THREADS':'1'}
    process = subprocess.run([sys.executable, '-c', code, str(PROJECT/'configs/project_smoke.yaml'),
                              str(tmp_path), str(run), str(PROJECT/'src')], env=env, timeout=180, capture_output=True, text=True)
    assert process.returncode == 37, process.stderr
    status_path = run/'status.json'
    original = json.loads(status_path.read_text(encoding='utf-8'))
    assert original['status'] == 'running'
    assert original['owner']['pid'] != os.getpid()
    assert (run/'last.pt').exists() and not (run/'recovery-events.jsonl').exists()

    with pytest.raises(ValueError, match='explicit recovery'):
        resume_run(run)
    with _run_lock(run):
        with pytest.raises(ValueError, match='locked'):
            resume_run(run, recover_running=True)

    active = {**original, 'owner':{**original['owner'], 'pid':os.getpid()}}
    status_path.write_text(json.dumps(active), encoding='utf-8')
    with pytest.raises(ValueError, match='still active'):
        resume_run(run, recover_running=True)
    legacy = {**original, 'owner':None}
    status_path.write_text(json.dumps(legacy), encoding='utf-8')
    with pytest.raises(ValueError, match='legacy running'):
        resume_run(run, recover_running=True)
    remote = {**original, 'owner':{**original['owner'], 'hostname':'another-host'}}
    status_path.write_text(json.dumps(remote), encoding='utf-8')
    with pytest.raises(ValueError, match='different host'):
        resume_run(run, recover_running=True)
    status_path.write_text(json.dumps(original), encoding='utf-8')

    cfg_path = run/'resolved_config.yaml'
    config = cfg_path.read_bytes()
    cfg_path.write_bytes(config+b'\noutput_root: changed\n')
    with pytest.raises(ValueError, match='config changed'):
        resume_run(run, recover_running=True)
    cfg_path.write_bytes(config)

    last = run/'last.pt'
    backup = run/'last-backup.pt'
    last.replace(backup)
    last.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError, match='checkpoint could not be read'):
        resume_run(run, recover_running=True)
    last.unlink()
    backup.replace(last)
    assert json.loads(status_path.read_text(encoding='utf-8')) == original
    assert not (run/'recovery-events.jsonl').exists()

    previous_hash = hashlib.sha256(last.read_bytes()).hexdigest()
    result = resume_run(run, recover_running=True)
    assert result['epochs'] == 2
    status = json.loads(status_path.read_text(encoding='utf-8'))
    assert status['status'] == 'completed' and status['owner']['pid'] == os.getpid()
    events = [json.loads(line) for line in (run/'recovery-events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [event['event'] for event in events] == ['recovery_started','recovery_completed']
    assert events[0]['previous_owner'] == original['owner'] and events[0]['epoch'] == 1
    assert events[0]['last_checkpoint_sha256'] == previous_hash
    assert events[0]['event_id'] == events[1]['event_id'] == status['recovery_event_id']


def test_recover_after_final_epoch_finishes_validation_without_retraining(tmp_path):
    torch.set_num_threads(1)
    make_smoke(tmp_path/'data'/'smoke')
    run = tmp_path/'orphan-final'
    code = r'''
import os, sys, torch
sys.path.insert(0,sys.argv[4])
import furniture.train as train
from furniture.config import load_config
from furniture.pipeline import train_run
torch.set_num_threads(1)
cfg=load_config(sys.argv[1])
cfg['project_root']=sys.argv[2]
cfg['training']['epochs']=1
original=train.atomic_torch_save
def crash_after_checkpoint(payload,path):
    original(payload,path)
    if path.name == 'last.pt':
        os._exit(38)
train.atomic_torch_save=crash_after_checkpoint
train_run(cfg,sys.argv[3])
'''
    env = {**os.environ, 'OMP_NUM_THREADS':'1'}
    process = subprocess.run([sys.executable, '-c', code, str(PROJECT/'configs/project_smoke.yaml'),
                              str(tmp_path), str(run), str(PROJECT/'src')], env=env, timeout=180, capture_output=True, text=True)
    assert process.returncode == 38, process.stderr
    checkpoint_before = (run/'last.pt').read_bytes()
    assert not (run/'validation.json').exists()
    result = resume_run(run, recover_running=True)
    assert result['epochs'] == 1
    assert (run/'last.pt').read_bytes() == checkpoint_before
    assert (run/'validation.json').is_file() and (run/'checkpoint.sha256').is_file()
    assert json.loads((run/'status.json').read_text(encoding='utf-8'))['status'] == 'completed'


def test_abandon_orphan_before_first_checkpoint_checks_lock_owner_and_provenance(tmp_path):
    run = tmp_path/'orphan-before-checkpoint'
    code = r'''
import os, sys
sys.path.insert(0,sys.argv[4])
import furniture.train as train
from furniture.config import load_config
from furniture.pipeline import train_run
cfg=load_config(sys.argv[1])
cfg['project_root']=sys.argv[2]
train.fit=lambda *args, **kwargs: os._exit(39)
train_run(cfg,sys.argv[3])
'''
    env = {**os.environ, 'OMP_NUM_THREADS':'1'}
    process = subprocess.run([sys.executable, '-c', code, str(PROJECT/'configs/project_smoke.yaml'),
                              str(tmp_path), str(run), str(PROJECT/'src')], env=env, timeout=180, capture_output=True, text=True)
    assert process.returncode == 39, process.stderr
    status_path = run/'status.json'
    original = json.loads(status_path.read_text(encoding='utf-8'))
    provenance = (run/'provenance.json').read_bytes()
    expected = json.loads(provenance)['config_sha256']
    assert original['status'] == 'running' and not (run/'last.pt').exists()
    with _run_lock(run):
        with pytest.raises(ValueError, match='locked'):
            abandon_running(run, expected)
    active = {**original, 'owner':{**original['owner'], 'pid':os.getpid()}}
    status_path.write_text(json.dumps(active), encoding='utf-8')
    with pytest.raises(ValueError, match='still active'):
        abandon_running(run, expected)
    status_path.write_text(json.dumps(original), encoding='utf-8')
    with pytest.raises(ValueError, match='suite manifest'):
        abandon_running(run, '0'*64)
    last = run/'last.pt'
    last.write_bytes(b'preserve')
    with pytest.raises(ValueError, match='Checkpoint exists'):
        abandon_running(run, expected)
    assert last.read_bytes() == b'preserve'
    last.unlink()
    assert json.loads(status_path.read_text(encoding='utf-8')) == original
    assert not (run/'recovery-events.jsonl').exists()
    result = abandon_running(run, expected)
    status = json.loads(status_path.read_text(encoding='utf-8'))
    assert result['status'] == status['status'] == 'interrupted'
    assert (run/'provenance.json').read_bytes() == provenance
    event = json.loads((run/'recovery-events.jsonl').read_text(encoding='utf-8'))
    assert event['event'] == 'recovery_abandoned_no_checkpoint'
    assert event['config_sha256'] == expected and event['previous_owner'] == original['owner']
    assert event['event_id'] == status['recovery_event_id']
