"""Crash-safe, checkpoint-bound evaluation and report artifacts."""
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import uuid

import torch
from furniture.config import training_settings


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def code_fingerprint(root):
    root = Path(root)
    paths = list((root/'src/furniture').glob('*.py')) + list((root/'scripts').glob('*.py'))
    paths += list((root/'tests').glob('*.py')) + list((root/'configs').rglob('*.yaml'))
    paths += list((root/'notebooks').glob('*.ipynb')) + [root/'pyproject.toml']
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(paths)}


def validate_test_binding(root, binding_path):
    root = Path(root)
    binding = json.loads(Path(binding_path).read_text(encoding='utf-8'))
    if binding.get('code_files') != code_fingerprint(root) or binding.get('result') != 'passed':
        raise ValueError('Combined tests are not bound to the current code and configs')
    log = root/binding['log']
    if not log.resolve().is_relative_to(root.resolve()) or sha(log) != binding.get('log_sha256'):
        raise ValueError('Combined test log is missing or changed')
    return binding


def snapshot_contract(root):
    root = Path(root)
    metadata = json.loads((root/'dataset_metadata.json').read_text(encoding='utf-8'))
    hashes = {name:sha(root/f'{name}.csv') for name in ('train','val','test')}
    if hashes != metadata['split_hashes']:
        raise ValueError('Snapshot split hashes do not match frozen metadata')
    return {'split_hashes':hashes, 'class_mapping':metadata['class_mapping']}


def validate_run(item, snapshot):
    run = Path(item['attempts'][-1]['run_dir'])
    payload = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
    if payload.get('config') != training_settings(item['config']):
        raise ValueError(f'Checkpoint training configuration does not match run: {run}')
    if payload.get('split_hashes') != snapshot['split_hashes']:
        raise ValueError(f'Checkpoint split hashes do not match snapshot: {run}')
    tasks = item['config']['training']['tasks']
    if payload.get('mapping') != {task:snapshot['class_mapping'][task] for task in tasks}:
        raise ValueError(f'Checkpoint class mapping does not match snapshot: {run}')
    metrics = json.loads((run/'validation.json').read_text(encoding='utf-8'))
    for task in tasks:
        for metric in ('accuracy','macro_precision','macro_recall','macro_f1','weighted_f1'):
            value = metrics[task][metric]
            expected = payload['val'][task][metric]
            if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f'Validation metrics do not match checkpoint: {run}, {task}, {metric}')
    return {'run_dir':str(run), 'config_sha256':item['config_sha256'],
            'checkpoint_sha256':sha(run/'best.pt'), 'validation_sha256':sha(run/'validation.json')}


def validate_bundle(path, context, required):
    root = Path(path)
    evidence = json.loads((root/'evidence.json').read_text(encoding='utf-8'))
    if evidence.get('version') != 1 or evidence.get('context') != context:
        raise ValueError(f'Artifact checkpoint, split or inputs differ: {root}')
    files = evidence.get('files')
    if not isinstance(files, dict) or not set(required) <= set(files):
        raise ValueError(f'Artifact is incomplete: {root}')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != root/'evidence.json'}
    if actual != set(files):
        raise ValueError(f'Artifact file set differs: {root}')
    for name, expected in files.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha(path) != expected:
            raise ValueError(f'Artifact file missing or changed: {name}')
    return evidence


def bundle(target, context, required, build):
    """Retain interrupted attempts; only an intact, bound directory is published."""
    target = Path(target)
    def event(action, **details):
        events = target.parent/'recovery-events'
        events.mkdir(parents=True, exist_ok=True)
        value = {'stage':target.name, 'action':action, 'time_utc':datetime.now(timezone.utc).isoformat(),
                 'context_sha256':hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest(), **details}
        temporary = events/f'.{uuid.uuid4().hex}.tmp'
        final = events/f'{uuid.uuid4().hex}.json'
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, final)
    if target.exists():
        try:
            validate_bundle(target, context, required)
        except (OSError, ValueError, KeyError, TypeError) as error:
            event('rejected_existing', error=f'{type(error).__name__}: {error}')
            raise
        event('reused_verified')
        return target
    attempts = target.parent/'_attempts'
    attempts.mkdir(parents=True, exist_ok=True)
    previous = sorted(p.name for p in attempts.glob(f'{target.name}-*') if p.is_dir())
    if previous:
        event('recovery_started', previous_attempts=previous)
    attempt = attempts/f'{target.name}-{uuid.uuid4().hex}'
    attempt.mkdir()
    event('attempt_started', attempt=attempt.name)
    try:
        build(attempt)
        files = {p.relative_to(attempt).as_posix():sha(p) for p in attempt.rglob('*') if p.is_file()}
        if not set(required) <= set(files):
            raise ValueError(f'Incomplete artifact attempt: {attempt}')
        with (attempt/'evidence.json').open('x', encoding='utf-8') as stream:
            json.dump({'version':1, 'context':context, 'files':files}, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        validate_bundle(attempt, context, required)
        os.rename(attempt, target)
    except BaseException as error:
        event('attempt_interrupted', attempt=attempt.name, error=f'{type(error).__name__}: {error}')
        raise
    event('published', attempt=attempt.name)
    return target


def repeated_summary(items, backbone, output):
    """Summarize exactly seed 42, 123, 2026 for the same C1/C2/C3/C5 controls."""
    groups = {}
    for item in items:
        cfg = item['config']
        if cfg['model']['backbone'] != backbone:
            continue
        from scripts.orchestrate_model_release import condition_id
        condition = condition_id(cfg)
        if condition not in ('C1','C2','C3','C5'):
            continue
        seed = cfg['training']['seed']
        control = {'model':cfg['model'], 'training':{k:v for k,v in cfg['training'].items() if k != 'seed'},
                   'augmentation':cfg['augmentation'], 'dataset':cfg['dataset']}
        metrics = json.loads((Path(item['attempts'][-1]['run_dir'])/'validation.json').read_text(encoding='utf-8'))
        for task in ('type','style'):
            group = groups.setdefault((condition, task), {'control':control, 'scores':{}})
            if group['control'] != control or seed in group['scores']:
                raise ValueError(f'Repeated-seed controls differ or seed is duplicated: {condition}, {task}')
            group['scores'][seed] = metrics[task]['macro_f1']
    fields = ['backbone','condition','task','seeds','n_seeds','macro_f1_mean','macro_f1_sample_std']
    with Path(output).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (condition, task), group in sorted(groups.items()):
            if set(group['scores']) != {42, 123, 2026}:
                raise ValueError(f'Incomplete repeated seeds: {condition}, {task}')
            values = [group['scores'][s] for s in (42, 123, 2026)]
            writer.writerow({'backbone':backbone,'condition':condition,'task':task,'seeds':'42,123,2026',
                             'n_seeds':3,'macro_f1_mean':statistics.mean(values),
                             'macro_f1_sample_std':statistics.stdev(values)})
    if len(groups) != 8:
        raise ValueError('Expected four repeated conditions for both tasks')


def controlled_contrasts(controlled, ablation, backbone, output):
    from scripts.orchestrate_model_release import condition_id
    by_condition = {condition_id(i['config']):i for i in controlled
                    if i['config']['model']['backbone'] == backbone and i['config']['training']['seed'] == 42}
    single = {i['config']['training']['tasks'][0]:i for i in ablation
              if i['config']['model']['backbone'] == backbone and len(i['config']['training']['tasks']) == 1}
    if set(by_condition) != {f'C{i}' for i in range(1,9)} or set(single) != {'type','style'}:
        raise ValueError('Controlled or single-task comparisons are incomplete')
    def controls(cfg, axis):
        model, dataset = cfg['model'], cfg['dataset']
        training, augmentation = deepcopy(cfg['training']), deepcopy(cfg['augmentation'])
        if axis == 'training_mode':
            training.pop('freeze_epochs', None)
        elif axis == 'tasks':
            training.pop('tasks', None)
        elif axis == 'augmentation':
            augmentation = {'occlusion':augmentation['occlusion']}
        elif axis == 'occlusion':
            augmentation.pop('occlusion', None)
        return {'model':model,'dataset':dataset,'training':training,'augmentation':augmentation}
    pairs = [('training_mode','C1','C2'), ('augmentation','C2','C3'), ('augmentation','C3','C4'),
             ('occlusion','C3','C5'), ('occlusion','C3','C6'), ('occlusion','C4','C7'), ('occlusion','C4','C8')]
    pairs += [('tasks',f'single-{task}','C3') for task in ('type','style')]
    fields = ['backbone','axis','baseline','comparison','task','baseline_macro_f1','comparison_macro_f1','delta','note']
    with Path(output).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for axis, left_name, right_name in pairs:
            left = single[left_name.removeprefix('single-')] if axis == 'tasks' else by_condition[left_name]
            right = by_condition[right_name]
            if controls(left['config'], axis) != controls(right['config'], axis):
                raise ValueError(f'Comparison controls differ: {left_name}, {right_name}')
            left_metrics = json.loads((Path(left['attempts'][-1]['run_dir'])/'validation.json').read_text(encoding='utf-8'))
            right_metrics = json.loads((Path(right['attempts'][-1]['run_dir'])/'validation.json').read_text(encoding='utf-8'))
            tasks = [left_name.removeprefix('single-')] if axis == 'tasks' else ['type','style']
            for task in tasks:
                a, b = left_metrics[task]['macro_f1'], right_metrics[task]['macro_f1']
                writer.writerow({'backbone':backbone,'axis':axis,'baseline':left_name,'comparison':right_name,
                                 'task':task,'baseline_macro_f1':a,'comparison_macro_f1':b,'delta':b-a,
                                 'note':'Single-task checkpoint selection uses its own task; joint uses mean of tasks' if axis == 'tasks' else ''})


def synthetic_report(destination, evaluation, snapshot, checkpoint_sha256):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination, evaluation = Path(destination), Path(evaluation)
    metrics = json.loads((evaluation/'test.json').read_text(encoding='utf-8'))
    with (evaluation/'test_predictions.csv').open(encoding='utf-8', newline='') as stream:
        predictions = list(csv.DictReader(stream))
    with (destination/'errors.csv').open('x', encoding='utf-8', newline='') as stream:
        fields = ['image_id','image_path','task','true_label','predicted_label','confidence']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in predictions:
            for task in ('type','style'):
                if row[f'true_{task}'] != row[f'pred_{task}']:
                    writer.writerow({'image_id':row['image_id'], 'image_path':row['image_path'], 'task':task,
                                     'true_label':row[f'true_{task}'], 'predicted_label':row[f'pred_{task}'],
                                     'confidence':row[f'confidence_{task}']})
    for task in ('type','style'):
        labels = [k for k, _ in sorted(snapshot['class_mapping'][task].items(), key=lambda pair:pair[1])]
        matrix = metrics[task]['confusion_matrix']
        if len(matrix) != len(labels) or any(len(row) != len(labels) for row in matrix):
            raise ValueError(f'Invalid {task} confusion matrix')
        fig, ax = plt.subplots(figsize=(max(5, len(labels)*.45), max(4, len(labels)*.45)))
        image = ax.imshow(matrix, cmap='Blues')
        fig.colorbar(image, ax=ax)
        ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
               xlabel='Predicted', ylabel='True', title=f'Synthetic test: {task}')
        ax.tick_params(axis='x', labelrotation=90)
        fig.tight_layout()
        fig.savefig(destination/f'{task}-confusion.png', dpi=120)
        plt.close(fig)
        shutil.copy2(evaluation/f'test_{task}_confusion.csv', destination/f'{task}-confusion.csv')
    with (destination/'report.json').open('x', encoding='utf-8') as stream:
        json.dump({'checkpoint_sha256':checkpoint_sha256, 'split_hashes':snapshot['split_hashes'],
                   'split':'synthetic_test', 'samples':len(predictions), 'metrics':metrics}, stream, indent=2)
