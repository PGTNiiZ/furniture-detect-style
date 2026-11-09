"""Thin orchestration: each stage calls one testable module."""
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
import csv
import hashlib
import json
import os
import socket
import uuid
import yaml
from .config import load_config, resolve_path, training_settings, digest


@contextmanager
def _run_lock(run):
    """Hold an OS lock across all writes; the kernel releases it if the process dies."""
    with (run/'.run.lock').open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            raise ValueError('Run is locked by an active process') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _owner():
    return {'hostname':socket.gethostname(), 'pid':os.getpid(), 'session_id':uuid.uuid4().hex}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _process_alive(pid):
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError('Run owner has an invalid PID')
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:  # ERROR_INVALID_PARAMETER: no process with this PID
                return False
            raise ValueError(f'Cannot verify owner process {pid}: Windows error {error}')
        try:
            result = kernel.WaitForSingleObject(handle, 0)
            if result == 0:
                return False
            if result == 258:
                return True
            raise ValueError(f'Cannot verify owner process {pid}: wait result {result}')
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError as error:
        raise ValueError(f'Cannot verify owner process {pid}: permission denied') from error


def new_run_path(cfg):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dataset_id = ''.join(c if c.isalnum() or c in '-_' else '_' for c in cfg['dataset']['id'])
    return resolve_path(cfg,cfg['output_root'])/dataset_id/cfg['experiment']['id']/(stamp+'-'+uuid.uuid4().hex[:8])


def train_run(cfg, run_dir=None):
    from .train import fit, evaluate
    run = Path(run_dir).resolve() if run_dir else new_run_path(cfg)
    if run.exists():
        raise ValueError('Run directory exists; each run is immutable')
    run.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    owner = _owner()
    def status(state, **extra):
        value = {'run_id':run.name, 'experiment_id':cfg['experiment']['id'], 'status':state,
                 'started_at':started, 'augmentation_protocol':'post-p0-fix-v1', 'owner':owner, **extra}
        temporary = run/'status.json.tmp'
        temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
        os.replace(temporary, run/'status.json')
    with _run_lock(run):
        status('running')
        try:
            return _train_prepared(cfg, run, fit, evaluate, status)
        except BaseException as error:
            state = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
            status(state, ended_at=datetime.now(timezone.utc).isoformat(), stop_reason=f'{type(error).__name__}: {error}')
            raise


def _train_prepared(cfg, run, fit, evaluate, status):
    (run/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    source_dir = Path(__file__).parent
    code = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_dir.glob('*.py'))}
    # Save actual code too; editing the working copy does not erase run provenance.
    (run/'source').mkdir()
    for p in source_dir.glob('*.py'):
        (run/'source'/p.name).write_bytes(p.read_bytes())
    (run/'provenance.json').write_text(json.dumps({'config_sha256':digest(cfg),'code_sha256':code},indent=2),encoding='utf-8')
    result = fit(resolve_path(cfg,cfg['dataset']['root']),run,training_settings(cfg),run_prepared=True)
    evaluate(resolve_path(cfg,cfg['dataset']['root']),run/'best.pt',run/'validation.json',split='val',device=cfg['training']['device'])
    status('completed', ended_at=datetime.now(timezone.utc).isoformat(), stop_reason=result['stop_reason'], device=result['device'])
    return {**result,'run_dir':str(run),'experiment':cfg['experiment']['id']}


def _validated_run_config(run, expected_config_hash=None):
    cfg = yaml.safe_load((run/'resolved_config.yaml').read_text(encoding='utf-8'))
    provenance = json.loads((run/'provenance.json').read_text(encoding='utf-8'))
    config_hash = digest(cfg)
    if provenance['config_sha256'] != config_hash:
        raise ValueError('Resolved config changed since the run started')
    if expected_config_hash is not None and expected_config_hash != config_hash:
        raise ValueError('Run config does not match the suite manifest')
    expected = provenance['code_sha256']
    current = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
    saved = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((run/'source').glob('*.py'))}
    if current != expected or saved != expected:
        raise ValueError('Code or source snapshot changed since the run started')
    return cfg


def _resume_preflight(run):
    """Check stored provenance and a complete epoch checkpoint before changing status."""
    import subprocess
    import sys
    import torch
    from .data import validate_splits
    from .train import choose_device
    cfg = _validated_run_config(run)
    checkpoint = run/'last.pt'
    if not checkpoint.is_file() or not (run/'best.pt').is_file():
        raise ValueError('Recovery requires intact last.pt and best.pt checkpoints')
    try:
        last = torch.load(checkpoint, map_location='cpu', weights_only=True)
        best = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
    except Exception as error:
        raise ValueError('Recovery checkpoint could not be read') from error
    needed = {'state_dict','config','mapping','split_hashes','optimizer','scheduler','history','epoch',
              'best','best_epoch','stale','resume_protocol','torch_version','runtime','initialization','exposure_membership'}
    if not isinstance(last, dict) or not needed <= set(last) or last['resume_protocol'] != 'epoch-seed-v1':
        raise ValueError('Recovery checkpoint is incomplete or not resumable')
    if not isinstance(best, dict) or not {'state_dict','config','mapping','split_hashes','epoch'} <= set(best):
        raise ValueError('Best checkpoint is incomplete')
    config = training_settings(cfg)
    data = resolve_path(cfg, cfg['dataset']['root'])
    splits, mapping = validate_splits(data)
    mapping = {task:mapping[task] for task in config.get('tasks',['type','style'])}
    hashes = {key:hashlib.sha256((data/f'{key}.csv').read_bytes()).hexdigest() for key in splits}
    if (last['config'] != config or last['mapping'] != mapping or last['split_hashes'] != hashes
            or best['config'] != config or best['mapping'] != mapping or best['split_hashes'] != hashes):
        raise ValueError('Recovery config, mapping or dataset splits differ from checkpoints')
    if not isinstance(last['epoch'], int) or last['epoch'] < 1 or len(last['history']) != last['epoch']:
        raise ValueError('Recovery checkpoint epoch/history is inconsistent')
    if last['epoch'] > config['epochs']:
        raise ValueError('Recovery checkpoint epoch exceeds configured budget')
    if best['epoch'] != last['best_epoch'] or best['epoch'] > last['epoch']:
        raise ValueError('Best and last checkpoint epochs are inconsistent')
    device = choose_device(config['device'])
    runtime = {'device':str(device), 'cuda_runtime':torch.version.cuda,
               'gpu_name':torch.cuda.get_device_name(device) if device.type == 'cuda' else None}
    if last['torch_version'] != str(torch.__version__) or last['runtime'] != runtime:
        raise ValueError('Recovery requires the same PyTorch version, device and CUDA runtime')
    if (run/'environment.txt').read_bytes() != subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']):
        raise ValueError('Recovery requires the same installed package versions')
    completed = last['epoch'] == config['epochs'] or last['stale'] >= config['patience']
    if not completed and any(run.glob('validation*')):
        raise ValueError('Validation outputs already exist for an unfinished run; refusing to overwrite')
    final_state = None
    if completed:
        if 'val' not in best:
            raise ValueError('Best checkpoint lacks validation metrics')
        final_state = {'best_validation_score':last['best'], 'selection_metric':config.get('selection_metric', 'style_macro_f1'),
                       'best_epoch':last['best_epoch'], 'epochs':last['epoch'], 'device':str(device),
                       'initialization':last['initialization'], 'physical_batch_size':config['batch_size'],
                       'effective_batch_size':config['batch_size'] * config.get('accumulation_steps',1),
                       'elapsed_seconds_this_invocation':None, 'peak_gpu_bytes':None,
                       'stop_reason':'early_stopping' if last['stale'] >= config['patience'] else 'max_epochs',
                       'shared_backbone_update_epochs':sum(not h['backbone_frozen'] for h in last['history'])}
    return cfg, _sha256_file(checkpoint), last['epoch'], final_state


def _finalize_interrupted_training(run, cfg, summary, evaluate):
    """Finish validation after a crash at the final checkpoint, without overwriting evidence."""
    best = run/'best.pt'
    expected_hash = _sha256_file(best)
    hash_path = run/'checkpoint.sha256'
    if hash_path.exists() and hash_path.read_text().strip() != expected_hash:
        raise ValueError('Existing best checkpoint hash differs')
    summary_path = run/'summary.json'
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding='utf-8'))
        for key in ('best_validation_score','selection_metric','best_epoch','epochs','device','stop_reason'):
            if existing.get(key) != summary[key]:
                raise ValueError(f'Existing training summary conflicts: {key}')
        summary = existing
    stage = run/('.recovery-validation-'+uuid.uuid4().hex)
    evaluate(resolve_path(cfg,cfg['dataset']['root']),best,stage/'validation.json',split='val',device=cfg['training']['device'])
    generated = list(stage.iterdir())
    for path in generated:
        destination = run/path.name
        if destination.exists() and destination.read_bytes() != path.read_bytes():
            raise ValueError(f'Existing validation output conflicts; staged result retained at {stage}')
    for path in generated:
        destination = run/path.name
        if destination.exists():
            path.unlink()
        else:
            os.replace(path, destination)
    stage.rmdir()
    if not hash_path.exists():
        hash_path.write_text(expected_hash)
    if not summary_path.exists():
        temporary = run/'summary.json.tmp'
        temporary.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        os.replace(temporary, summary_path)
    return summary


def _record_recovery(run, event):
    with (run/'recovery-events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _check_orphan_owner(state):
    previous = state.get('owner')
    if not isinstance(previous, dict) or not previous.get('session_id'):
        raise ValueError('Cannot safely recover a legacy running run without process identity')
    if previous.get('hostname') != socket.gethostname():
        raise ValueError('Cannot verify the original process on a different host')
    if _process_alive(previous.get('pid')):
        raise ValueError('Original run owner process is still active')
    return previous


def abandon_running(run_dir, expected_config_hash):
    """Record an orphaned attempt with no resumable checkpoint; preserve all evidence."""
    run = Path(run_dir).resolve()
    if not run.is_dir():
        raise ValueError('Run directory does not exist')
    if not isinstance(expected_config_hash, str) or len(expected_config_hash) != 64:
        raise ValueError('Expected suite config hash is required')
    with _run_lock(run):
        status_path = run/'status.json'
        state = json.loads(status_path.read_text(encoding='utf-8'))
        if state.get('status') != 'running' or state.get('augmentation_protocol') != 'post-p0-fix-v1':
            raise ValueError('Only a post-fix running orphan can be abandoned')
        previous = _check_orphan_owner(state)
        last = run/'last.pt'
        if last.exists() or last.is_symlink():
            raise ValueError('Checkpoint exists; use guarded resume instead')
        _validated_run_config(run, expected_config_hash)
        owner, event_id = _owner(), uuid.uuid4().hex
        event = {'event':'recovery_abandoned_no_checkpoint', 'event_id':event_id,
                 'time':datetime.now(timezone.utc).isoformat(), 'previous_owner':previous,
                 'new_owner':owner, 'config_sha256':expected_config_hash,
                 'best_checkpoint_present':(run/'best.pt').exists()}
        _record_recovery(run, event)
        value = {**state, 'status':'interrupted', 'ended_at':event['time'],
                 'stop_reason':'Orphaned before last.pt; original artifacts preserved',
                 'abandoned_by':owner, 'recovery_event_id':event_id}
        temporary = run/'status.json.tmp'
        temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
        os.replace(temporary, status_path)
        return {'run_dir':str(run), 'status':'interrupted', 'recovery_event_id':event_id}


def resume_run(run_dir, recover_running=False):
    """Resume a failed run or explicitly recover an orphaned local running run."""
    from .train import fit, evaluate
    run = Path(run_dir).resolve()
    if not run.is_dir():
        raise ValueError('Run directory does not exist')
    with _run_lock(run):
        status_path = run/'status.json'
        state = json.loads(status_path.read_text(encoding='utf-8'))
        if state.get('augmentation_protocol') != 'post-p0-fix-v1':
            raise ValueError('Only a failed or interrupted post-fix run can be resumed')
        recovering = state.get('status') == 'running'
        if recovering:
            if not recover_running:
                raise ValueError('Run is marked running; use explicit recovery after checking the owner')
            previous = _check_orphan_owner(state)
        elif state.get('status') not in ('failed','interrupted'):
            raise ValueError('Only a failed or interrupted post-fix run can be resumed')
        cfg, checkpoint_hash, epoch, final_state = _resume_preflight(run)
        owner = _owner()
        event_id = uuid.uuid4().hex if recovering else None
        def write_status(status, **extra):
            value = {**state, 'status':status, 'owner':owner, 'recovery_event_id':event_id, **extra}
            temporary = run/'status.json.tmp'
            temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
            os.replace(temporary, status_path)
        if recovering:
            _record_recovery(run, {'event':'recovery_started', 'event_id':event_id,
                                  'time':datetime.now(timezone.utc).isoformat(), 'previous_owner':previous,
                                  'new_owner':owner, 'last_checkpoint_sha256':checkpoint_hash, 'epoch':epoch})
        write_status('running', resumed_at=datetime.now(timezone.utc).isoformat(), recovery_event_id=event_id)
        try:
            if final_state is None:
                result = fit(resolve_path(cfg,cfg['dataset']['root']),run,training_settings(cfg),run_prepared=True,resume=True)
                evaluate(resolve_path(cfg,cfg['dataset']['root']),run/'best.pt',run/'validation.json',split='val',device=cfg['training']['device'])
            else:
                result = _finalize_interrupted_training(run, cfg, final_state, evaluate)
            write_status('completed', ended_at=datetime.now(timezone.utc).isoformat(), stop_reason=result['stop_reason'], device=result['device'])
            if recovering:
                _record_recovery(run, {'event':'recovery_completed', 'event_id':event_id, 'time':datetime.now(timezone.utc).isoformat()})
            return {**result,'run_dir':str(run),'experiment':cfg['experiment']['id']}
        except BaseException as error:
            write_status('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                         ended_at=datetime.now(timezone.utc).isoformat(), stop_reason=f'{type(error).__name__}: {error}')
            if recovering:
                _record_recovery(run, {'event':'recovery_failed', 'event_id':event_id,
                                      'time':datetime.now(timezone.utc).isoformat(), 'reason':f'{type(error).__name__}: {error}'})
            raise


def run_experiments(project_file, experiments):
    from .prepare import audit_data
    configs = [load_config(project_file,name) for name in experiments]
    if len(set(experiments)) != len(experiments):
        raise ValueError('Duplicate experiment IDs')
    # These fields must be equal for the supplied E1-E3 controlled comparison.
    first = configs[0]
    baseline = audit_data(first)
    for cfg in configs[1:]:
        if cfg['dataset'] != first['dataset'] or cfg['model'] != first['model']:
            raise ValueError('Suite requires identical dataset/model; run other comparisons separately')
        controls = lambda item: {k:v for k,v in item['training'].items() if k not in ('freeze_epochs','device')}
        if controls(cfg) != controls(first):
            raise ValueError('Suite changes multiple training controls; use a separate explicit comparison')
        if audit_data(cfg)['split_hashes'] != baseline['split_hashes']:
            raise ValueError('Suite manifest hashes differ')
    outputs = [train_run(cfg) for cfg in configs]
    comparison_path = resolve_path(first,first['output_root'])/'comparisons'/('suite-'+uuid.uuid4().hex[:12]+'.csv')
    comparison = compare_runs([r['run_dir'] for r in outputs],comparison_path, vary=('training_mode','augmentation','occlusion'))
    return {'runs':outputs,'comparison':comparison,'comparison_csv':str(comparison_path)}


def compare_runs(run_dirs, output=None, vary=()):
    import torch
    allowed = {'backbone','augmentation','occlusion','training_mode','tasks','seed'}
    if set(vary) - allowed:
        raise ValueError('Unknown comparison axis')
    records, reference, reference_mapping = [], None, {}
    for name in run_dirs:
        root = Path(name)
        status_path = root/'status.json'
        status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
        if status.get('status') != 'completed' or status.get('augmentation_protocol') != 'post-p0-fix-v1':
            raise ValueError('Only completed runs with recorded status can be compared; older runs may have used invalid augmentation')
        payload = torch.load(root/'best.pt',map_location='cpu',weights_only=True)
        cfg = yaml.safe_load((root/'resolved_config.yaml').read_text(encoding='utf-8'))
        model = dict(cfg['model'])
        if 'backbone' in vary:
            model.pop('backbone',None)
            model.pop('weights',None)
        training = {k:v for k,v in cfg['training'].items() if k != 'device'}
        if 'training_mode' in vary:
            training.pop('freeze_epochs',None)
        if 'seed' in vary:
            training.pop('seed',None)
        if 'tasks' in vary:
            training.pop('tasks',None)
        augmentation = dict(cfg['augmentation'])
        if 'occlusion' in vary:
            augmentation.pop('occlusion',None)
        if 'augmentation' in vary:
            augmentation = {'occlusion':augmentation.get('occlusion')} if 'occlusion' not in vary else None
        control = {'splits':payload['split_hashes'],'mapping':payload['mapping'],
                   'model':model, 'augmentation':augmentation, 'training':training}
        if 'tasks' in vary:
            control.pop('mapping')
            for task, values in payload['mapping'].items():
                if task in reference_mapping and reference_mapping[task] != values:
                    raise ValueError('Runs have different label mappings')
                reference_mapping[task] = values
        if reference is None:
            reference = control
        if control != reference:
            raise ValueError('Runs have different data, mapping, model or training controls; comparison is not controlled')
        val = payload['val']
        records.append({'experiment':cfg['experiment']['id'],'run_dir':str(root),
                        'best_epoch':payload['epoch'],
                        'type_macro_f1':val['type']['macro_f1'] if 'type' in val else None,
                        'style_macro_f1':val['style']['macro_f1'] if 'style' in val else None,
                        'mean_macro_f1':sum(val[k]['macro_f1'] for k in payload['mapping'])/len(payload['mapping']),
                        'tasks':','.join(payload['mapping']),
                        'shared_backbone_update_epochs':payload.get('shared_backbone_update_epochs',0)})
    if output:
        output = Path(output)
        if output.exists():
            raise ValueError('Comparison output exists')
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('w',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return {'split':'validation','rows':records,'note':'Test evaluation is separate; mean scores for different task sets are not directly comparable'}


def evaluate_run(run_dir, split='test', device='auto'):
    from .train import evaluate
    root = Path(run_dir)
    cfg = yaml.safe_load((root/'resolved_config.yaml').read_text(encoding='utf-8'))
    destination = root/f'{split}.json'
    if destination.exists():
        raise ValueError(f'{destination} exists; results already recorded')
    return evaluate(resolve_path(cfg,cfg['dataset']['root']),root/'best.pt',destination,split,device)
