"""Build explicit screening, controlled, ablation and seed-repeat run plans."""
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import hashlib
import os
import uuid
import yaml

from .config import digest, load_config, resolve_path, validate
from .model import BACKBONES, WEIGHT_NAMES
from .pipeline import train_run, new_run_path


def plan_suite(project, stage, backbones=None, seeds=None, conditions=None):
    if stage not in ('screening', 'controlled', 'ablation', 'repeated'):
        raise ValueError('Unknown suite stage')
    backbones = list(backbones or (BACKBONES if stage == 'screening' else []))
    if not backbones or len(backbones) != len(set(backbones)) or any(x not in BACKBONES for x in backbones):
        raise ValueError('Provide distinct supported backbones')
    seeds = list(seeds or ([42, 123, 2026] if stage == 'repeated' else [42]))
    if len(set(seeds)) != len(seeds) or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    conditions = list(conditions or (['C3'] if stage in ('screening', 'ablation') else [f'C{i}' for i in range(1,9)]))
    if len(conditions) != len(set(conditions)) or any(x not in [f'C{i}' for i in range(1,9)] for x in conditions):
        raise ValueError('Conditions must be distinct C1–C8 IDs')
    if stage in ('screening','ablation') and conditions != ['C3']:
        raise ValueError('Screening and ablation require C3')
    plans = []
    for backbone in backbones:
        for condition in conditions:
            for seed in seeds:
                for tasks in ([['type'], ['style']] if stage == 'ablation' else [['type','style']]):
                    cfg = deepcopy(load_config(project, condition))
                    cfg['model']['backbone'] = backbone
                    cfg['model']['weights'] = WEIGHT_NAMES[backbone]
                    cfg['training']['seed'] = seed
                    cfg['training']['tasks'] = tasks
                    if stage == 'screening':
                        cfg['training']['epochs'] = 10
                        cfg['training']['freeze_epochs'] = 2
                    if cfg['model']['initialization'] != 'imagenet':
                        raise ValueError('Suite requires fresh ImageNet initialization')
                    cfg['experiment']['id'] = f'{stage}-{backbone}-{condition}-{seed}-{"-".join(tasks)}'
                    validate(cfg)
                    run = new_run_path(cfg)
                    plans.append({'run_dir':str(run), 'config':cfg})
    return plans


def describe_plan(plans):
    return {'runs':len(plans), 'models':sorted({p['config']['model']['backbone'] for p in plans}),
            'seeds':sorted({p['config']['training']['seed'] for p in plans}),
            'dataset':plans[0]['config']['dataset']['id'],
            'items':[{'id':p['config']['experiment']['id'], 'output':p['run_dir'],
                      'tasks':p['config']['training']['tasks'], 'augmentation':p['config']['augmentation']}
                     for p in plans],
            'note':'Configuration-only plan; data, weights and compute budget are not validated.'}


def _now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _suite_lock(path):
    """OS lock is released by the kernel even if the suite process crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if not stream.tell():
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
        except OSError as error:
            raise RuntimeError(f'Suite is already locked: {path}') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _write_manifest(path, manifest):
    manifest['updated_at'] = _now()
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _hash_file(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
        after = os.fstat(stream.fileno())
    current = path.stat()
    fingerprint = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if fingerprint(before) != fingerprint(after) or fingerprint(after) != fingerprint(current):
        raise ValueError(f'Dataset split changed while hashing: {path}')
    return hasher.hexdigest()


def _dataset_contract(cfg):
    return {'root':str(resolve_path(cfg, cfg['dataset']['root'])), 'config':deepcopy(cfg['dataset'])}


def _dataset_snapshot(cfg):
    contract = _dataset_contract(cfg)
    root = Path(contract['root'])
    try:
        hashes = {split:_hash_file(root/f'{split}.csv') for split in ('train','val','test')}
    except OSError as error:
        raise ValueError(f'Cannot hash dataset split manifest: {error}') from error
    return {**contract, 'split_hashes':hashes}


def _check_dataset(manifest):
    current = _dataset_snapshot(manifest['runs'][0]['config'])
    if current != manifest['dataset']:
        raise ValueError('Dataset identity or split hashes changed since suite creation; refusing to train')


def _complete_evidence(run, expected_config_hash):
    """A stale or edited manifest cannot turn an incompatible run into a success."""
    try:
        status = json.loads((run/'status.json').read_text(encoding='utf-8'))
        config = yaml.safe_load((run/'resolved_config.yaml').read_text(encoding='utf-8'))
        provenance = json.loads((run/'provenance.json').read_text(encoding='utf-8'))
        checkpoint = run/'best.pt'
        recorded_hash = (run/'checkpoint.sha256').read_text(encoding='utf-8').strip()
        validation = json.loads((run/'validation.json').read_text(encoding='utf-8'))
        with checkpoint.open('rb') as stream:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(chunk)
        return (status.get('status') == 'completed' and status.get('augmentation_protocol') == 'post-p0-fix-v1'
                and digest(config) == expected_config_hash == provenance.get('config_sha256')
                and isinstance(validation, dict) and checkpoint.is_file()
                and hasher.hexdigest() == recorded_hash)
    except (OSError, ValueError, TypeError, yaml.YAMLError):
        return False


def _validate_manifest(manifest):
    if manifest.get('version') != 2 or not isinstance(manifest.get('runs'), list) or not manifest['runs']:
        raise ValueError('Invalid or legacy suite manifest; version 2 with frozen dataset hashes is required')
    dataset = manifest.get('dataset')
    if not isinstance(dataset, dict) or set(dataset) != {'root','config','split_hashes'} or set(dataset.get('split_hashes', {})) != {'train','val','test'}:
        raise ValueError('Suite manifest lacks frozen dataset split hashes')
    paths = set()
    for item in manifest['runs']:
        cfg = item['config']
        if digest(cfg) != item['config_sha256']:
            raise ValueError('Suite manifest config digest mismatch')
        validate(cfg)
        if _dataset_contract(cfg) != {'root':dataset['root'], 'config':dataset['config']}:
            raise ValueError('Suite runs must use the same dataset identity and configuration')
        output_root = resolve_path(cfg, cfg['output_root'])
        if not item.get('attempts'):
            raise ValueError('Suite run has no attempt')
        if item.get('status') not in ('planned', 'running', 'completed', 'failed', 'interrupted'):
            raise ValueError('Invalid suite run status')
        for attempt in item['attempts']:
            path = Path(attempt['run_dir']).resolve()
            if not path.is_relative_to(output_root) or path == output_root or path in paths:
                raise ValueError('Suite run path is outside output root or duplicated')
            paths.add(path)


def _advance_suite_item(item, path, manifest, retry_failed, recover_running, restart_failed):
    attempt = item['attempts'][-1]
    run = Path(attempt['run_dir'])
    if run.exists() and _complete_evidence(run, item['config_sha256']):
        item['status'] = attempt['status'] = 'completed'
        _write_manifest(path, manifest)
        return
    if item['status'] == 'completed':
        raise ValueError(f'Completed run evidence is missing or differs: {run}')
    run_status = None
    if run.exists():
        status_path = run/'status.json'
        if not status_path.is_file():
            raise ValueError(f'Existing run has no status; refusing to overwrite: {run}')
        run_status = json.loads(status_path.read_text(encoding='utf-8')).get('status')
        if run_status == 'running' and not recover_running:
            item['status'] = attempt['status'] = 'running'
            _write_manifest(path, manifest)
            return
        if run_status in ('failed', 'interrupted') and not (retry_failed or restart_failed):
            item['status'] = attempt['status'] = run_status
            _write_manifest(path, manifest)
            return
        if run_status not in ('running', 'failed', 'interrupted'):
            raise ValueError(f'Run status is not recoverable: {run_status}')
        if run_status == 'running' and not (run/'last.pt').exists() and (retry_failed or restart_failed):
            from .pipeline import abandon_running
            abandon_running(run, item['config_sha256'])
            attempt['status'] = 'interrupted'
            attempt.update(ended_at=_now(), error='Recovered orphan had no checkpoint; fresh attempt created')
            attempt = {'run_dir':str(new_run_path(item['config']).resolve()), 'status':'planned'}
            item['attempts'].append(attempt)
            run = Path(attempt['run_dir'])
            action = 'train'
        elif run_status == 'running' or ((run/'last.pt').is_file() and not restart_failed):
            action = 'resume'
        else:
            attempt = {'run_dir':str(new_run_path(item['config']).resolve()), 'status':'planned'}
            item['attempts'].append(attempt)
            run = Path(attempt['run_dir'])
            action = 'train'
    else:
        if item['status'] in ('failed', 'interrupted'):
            if not (retry_failed or restart_failed):
                return
            attempt = {'run_dir':str(new_run_path(item['config']).resolve()), 'status':'planned'}
            item['attempts'].append(attempt)
            run = Path(attempt['run_dir'])
        action = 'train'
    _validate_manifest(manifest)
    attempt.update(status='running', started_at=_now(), action=action)
    item['status'] = 'running'
    _write_manifest(path, manifest)
    if action == 'train':
        train_run(item['config'], run)
    else:
        from .pipeline import resume_run
        resume_run(run, recover_running=(run_status == 'running'))
    if not _complete_evidence(run, item['config_sha256']):
        raise ValueError('Run returned without valid completion evidence')
    item['status'] = attempt['status'] = 'completed'
    attempt['ended_at'] = _now()
    _write_manifest(path, manifest)


def run_suite(plans=None, manifest_path=None, fail_fast=False, retry_failed=False, recover_running=False, restart_failed=False):
    """Persist an immutable plan; retry failures without overwriting completed or failed runs.

    Pass plans only when creating a manifest; continue later with plans=None and its path.
    A failed attempt with a checkpoint resumes it; without a checkpoint a fresh attempt is
    recorded at a new run directory. restart_failed explicitly starts a fresh attempt even
    when a failed run has a checkpoint. Recovery of running processes is guarded separately.
    """
    if not all(isinstance(value, bool) for value in (fail_fast, retry_failed, recover_running, restart_failed)):
        raise ValueError('Suite execution flags must be boolean')
    if plans is not None and not plans:
        raise ValueError('A suite needs at least one run')
    if manifest_path is None:
        if plans is None:
            raise ValueError('Provide manifest_path to continue a suite')
        cfg = plans[0]['config']
        manifest_path = resolve_path(cfg, cfg['output_root'])/'suites'/f'suite-{uuid.uuid4().hex}.json'
    path = Path(manifest_path).resolve()
    with _suite_lock(path.with_name(path.name+'.lock')):
        if path.exists():
            if plans is not None:
                raise ValueError('Suite manifest exists; continue with plans=None')
            manifest = json.loads(path.read_text(encoding='utf-8'))
        else:
            if plans is None:
                raise FileNotFoundError(f'Suite manifest not found: {path}')
            manifest = {'version':2, 'suite_id':uuid.uuid4().hex, 'created_at':_now(),
                        'dataset':_dataset_snapshot(plans[0]['config']), 'runs':[]}
            for plan in plans:
                cfg = deepcopy(plan['config'])
                manifest['runs'].append({'experiment':cfg['experiment']['id'], 'config':cfg,
                    'config_sha256':digest(cfg), 'status':'planned',
                    'attempts':[{'run_dir':str(Path(plan['run_dir']).resolve()), 'status':'planned'}]})
            _validate_manifest(manifest)
            _write_manifest(path, manifest)
        _validate_manifest(manifest)
        _check_dataset(manifest)
        if any(item['experiment'].startswith('screening-') for item in manifest['runs']):
            from .snapshot import validate_audited_snapshot
            try:
                validate_audited_snapshot(manifest['dataset']['root'])
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ValueError(f'Screening requires an audited training snapshot: {error}') from error
        for item in manifest['runs']:
            _check_dataset(manifest)
            try:
                _advance_suite_item(item, path, manifest, retry_failed, recover_running, restart_failed)
            except BaseException as error:
                attempt = item['attempts'][-1]
                status_path = Path(attempt['run_dir'])/'status.json'
                try:
                    underlying = json.loads(status_path.read_text(encoding='utf-8')).get('status')
                except (OSError, ValueError, AttributeError):
                    underlying = None
                state = ('interrupted' if isinstance(error, KeyboardInterrupt) else
                         'running' if underlying == 'running' else 'failed')
                item['status'] = attempt['status'] = state
                attempt.update(ended_at=_now(), error=f'{type(error).__name__}: {error}')
                _write_manifest(path, manifest)
                if fail_fast or not isinstance(error, Exception):
                    raise
        _check_dataset(manifest)
        counts = {state:sum(item['status'] == state for item in manifest['runs'])
                  for state in ('planned','running','completed','failed','interrupted')}
        return {'manifest':str(path), 'counts':counts, 'runs':[
            {'experiment':item['experiment'], 'status':item['status'], 'run_dir':item['attempts'][-1]['run_dir'],
             'attempts':len(item['attempts'])} for item in manifest['runs']]}


def select_screening(run_dirs, count=3):
    import torch
    import yaml
    if count < 1 or len(run_dirs) < count:
        raise ValueError('Not enough completed screening runs')
    records, controls = [], None
    seen = set()
    for name in run_dirs:
        root = Path(name)
        status = json.loads((root/'status.json').read_text(encoding='utf-8'))
        if status.get('status') != 'completed' or status.get('augmentation_protocol') != 'post-p0-fix-v1':
            raise ValueError('Only completed screening runs may be ranked')
        cfg = yaml.safe_load((root/'resolved_config.yaml').read_text(encoding='utf-8'))
        if not cfg['experiment']['id'].startswith('screening-'):
            raise ValueError('Only screening runs may be ranked')
        payload = torch.load(root/'best.pt', map_location='cpu', weights_only=True)
        backbone = cfg['model']['backbone']
        if backbone in seen:
            raise ValueError('Duplicate screening backbone')
        seen.add(backbone)
        model = {k:v for k,v in cfg['model'].items() if k not in ('backbone','weights')}
        current = {'splits':payload['split_hashes'], 'mapping':payload['mapping'], 'model':model,
                   'training':cfg['training'], 'augmentation':cfg['augmentation']}
        if controls is None:
            controls = current
        if controls != current:
            raise ValueError('Screening runs use incompatible controls')
        score = sum(payload['val'][task]['macro_f1'] for task in ('type','style'))/2
        records.append({'backbone':backbone,'mean_macro_f1':score,'run_dir':str(root)})
    scores = [r['mean_macro_f1'] for r in records]
    tied = {score for score in scores if scores.count(score) > 1}
    benchmark_controls = None
    for record in records:
        if record['mean_macro_f1'] not in tied:
            record['latency_ms'] = None
            continue
        benchmark_path = Path(record['run_dir'])/'benchmark.json'
        if not benchmark_path.exists():
            raise ValueError('Tied screening scores require benchmark.json for each tied run')
        benchmark = json.loads(benchmark_path.read_text(encoding='utf-8'))
        import hashlib
        checkpoint_hash = hashlib.sha256((Path(record['run_dir'])/'best.pt').read_bytes()).hexdigest()
        if benchmark.get('checkpoint_sha256') != checkpoint_hash:
            raise ValueError('Benchmark does not match the selected checkpoint')
        current = tuple(benchmark.get(k) for k in ('device','hardware','torch_version','cuda_runtime','image_size','batch_size','warmup','repetitions'))
        if benchmark_controls is None:
            benchmark_controls = current
        if benchmark_controls != current:
            raise ValueError('Tie-break benchmarks use different hardware or measurement settings')
        record['latency_ms'] = benchmark['latency_ms_mean_per_batch']
    return {'split':'validation', 'selected':sorted(records,key=lambda r:(-r['mean_macro_f1'],r['latency_ms'] or 0,r['backbone']))[:count],
            'tie_break':'mean inference latency on matched benchmark settings, then backbone name',
            'limitation':'Short screening can favor faster convergence.'}
