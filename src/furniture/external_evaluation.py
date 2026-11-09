"""Locked, closed-set comparison to source-provided external annotations."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from .data import image_hash, read_rows, transform
from .robustness import task_metrics
from .train import choose_device, load_model

TASKS = ('type', 'style')
VERSION = 'external-source-closed-set-v1'


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _read(train_manifest, external_manifest):
    train, external = read_rows(train_manifest), read_rows(external_manifest)
    if not train or not external:
        raise ValueError('Both train and external manifests must be non-empty')
    for name, rows in (('train', train), ('external', external)):
        ids = set()
        for row in rows:
            if not all(row.get(k) for k in ('image_id', 'image_path', 'pixel_hash', 'source_split')):
                raise ValueError(f'{name} rows require image_id, image_path, pixel_hash and source_split')
            if not all(k in row for k in TASKS):
                raise ValueError(f'{name} manifest requires type and style columns')
            for k in TASKS:
                if row[k] is None:
                    row[k] = ''
            if row['image_id'] in ids:
                raise ValueError(f'Duplicate image_id in {name}')
            ids.add(row['image_id'])
            if (row['source_split'].lower() == 'test') != (name == 'external'):
                raise ValueError('External rows must come from source test, and train must not')
            if name == 'train' and any(not row[k] for k in TASKS):
                raise ValueError('Training labels must be non-empty')
    for key in ('image_id', 'pixel_hash', 'group_id'):
        left = {r.get(key) for r in train} - {None, ''}
        right = {r.get(key) for r in external} - {None, ''}
        if left & right:
            raise ValueError(f'Train/external overlap in {key}')
    mapping = {k: {v: i for i, v in enumerate(sorted({r[k] for r in train}))} for k in TASKS}
    return external, mapping


def _state(label, classes):
    return 'missing' if label == '' else 'known' if label in classes else 'unseen'


def inspect_external(rows, mapping):
    """Count annotations only; no predictions or label transformations are performed."""
    states = [{k: _state(r[k], mapping[k]) for k in TASKS} for r in rows]
    counts = {k: {state: sum(s[k] == state for s in states)
                  for state in ('missing', 'known', 'unseen')} for k in TASKS}
    unseen = {k: dict(sorted(Counter(r[k] for r in rows if _state(r[k], mapping[k]) == 'unseen').items()))
              for k in TASKS}
    return {'images': len(rows), 'counts': counts, 'unseen_labels': unseen,
            'known_both': sum(all(s[k] == 'known' for k in TASKS) for s in states)}


def create_external_protocol(train_manifest, external_manifest, output):
    """Freeze manifests, class mapping and scoring rules before producing predictions."""
    rows, mapping = _read(train_manifest, external_manifest)
    result = {'protocol_version': VERSION, 'train_manifest_sha256': _hash(train_manifest),
              'external_manifest_sha256': _hash(external_manifest), 'class_mapping': mapping,
              'annotation_inspection': inspect_external(rows, mapping),
              'label_policy': 'exact label matching; no canonicalization or unknown rejection',
              'metrics_policy': 'per-task known-label subset; joint requires both known',
              'selection_policy': 'never use external source test for model or threshold selection',
              'annotation_provenance': 'Source card reports Qwen2 VL 72B labels and possible inconsistencies; human or expert verification is not stated. Results compare to source annotations, not expert ground truth.'}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


def _check_exposure(payload, rows, visited=None):
    """Fail closed for incomplete warm-start ancestry and all prior holdout membership."""
    visited = set() if visited is None else visited
    exposure = payload.get('exposure_membership')
    membership = payload.get('data_membership')
    if not isinstance(exposure, dict) or not isinstance(membership, dict):
        raise ValueError('Checkpoint lacks complete exposure membership')
    for split in ('train', 'val', 'test'):
        values = exposure.get(split)
        current = membership.get(split)
        if not isinstance(values, dict) or not isinstance(current, dict):
            raise ValueError('Checkpoint lacks complete exposure membership')
        for field, rowkey in (('pixel_hashes', 'pixel_hash'), ('group_ids', 'group_id'), ('image_ids', 'image_id')):
            if field == 'image_ids' and field not in values:
                continue  # Earlier checkpoints record pixel hashes and groups, not source IDs.
            members = values.get(field)
            if not isinstance(members, list) or any(not isinstance(v, str) or not v for v in members):
                raise ValueError('Checkpoint lacks complete exposure membership')
            current_members = current.get(field, [] if field == 'image_ids' else None)
            if (not isinstance(current_members, list) or any(not isinstance(v, str) or not v for v in current_members)
                    or not set(current_members) <= set(members)):
                raise ValueError('Checkpoint exposure does not contain its current membership')
            if set(members) & {r.get(rowkey) for r in rows if r.get(rowkey)}:
                raise ValueError(f'External holdout overlaps checkpoint {split} exposure in {field}')
    initialization = payload.get('initialization')
    if not isinstance(initialization, dict) or initialization.get('mode') not in ('scratch', 'imagenet', 'warm_start'):
        raise ValueError('Checkpoint initialization provenance is incomplete')
    if initialization['mode'] == 'warm_start':
        parent_path = Path(initialization.get('parent_checkpoint', ''))
        expected = initialization.get('parent_sha256')
        if not parent_path.is_file() or not expected or _hash(parent_path) != expected:
            raise ValueError('Warm-start parent checkpoint is unavailable or hash differs')
        if expected in visited:
            raise ValueError('Cyclic checkpoint ancestry')
        visited.add(expected)
        parent = torch.load(parent_path, map_location='cpu', weights_only=True)
        _check_exposure(parent, rows, visited)


def evaluate_external(protocol_path, train_manifest, external_manifest, image_root, checkpoint, output, device='auto'):
    """Explicitly run inference only after a protocol is written; never overwrite results."""
    protocol = json.loads(Path(protocol_path).read_text(encoding='utf-8'))
    rows, mapping = _read(train_manifest, external_manifest)
    if (protocol.get('protocol_version') != VERSION or protocol.get('class_mapping') != mapping
            or protocol.get('train_manifest_sha256') != _hash(train_manifest)
            or protocol.get('external_manifest_sha256') != _hash(external_manifest)
            or protocol.get('annotation_inspection') != inspect_external(rows, mapping)
            or protocol.get('label_policy') != 'exact label matching; no canonicalization or unknown rejection'
            or protocol.get('metrics_policy') != 'per-task known-label subset; joint requires both known'):
        raise ValueError('Protocol, class mapping or manifests differ from the locked inspection')
    destination = Path(output)
    if destination.exists():
        raise ValueError('Output already exists')
    device = choose_device(device)
    model, payload = load_model(checkpoint, device)
    if payload['mapping'] != mapping:
        raise ValueError('Checkpoint mapping differs from train-only class mapping')
    if payload.get('split_hashes', {}).get('train') != _hash(train_manifest):
        raise ValueError('Checkpoint is not bound to this training manifest')
    _check_exposure(payload, rows)
    train_rows = read_rows(train_manifest)
    for field, rowkey in (('pixel_hashes', 'pixel_hash'), ('group_ids', 'group_id')):
        if not {r.get(rowkey) for r in train_rows if r.get(rowkey)} <= set(payload['data_membership']['train'][field]):
            raise ValueError('Checkpoint membership does not contain the training manifest')
    root = Path(image_root).resolve()
    preprocess = transform(payload['config']['image_size'])
    names = {k: [name for name, _ in sorted(mapping[k].items(), key=lambda item: item[1])] for k in TASKS}
    truth, predictions = ({k: [] for k in TASKS} for _ in range(2))
    results, joint_correct, joint_n = [], 0, 0
    for row in rows:
        path = (root / row['image_path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Invalid external image path: {row['image_id']}")
        with Image.open(path) as image:
            if image_hash(image) != row['pixel_hash']:
                raise ValueError(f"Changed image: {row['image_id']}")
            tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(tensor)
        result = {'image_id': row['image_id']}
        for k in TASKS:
            index = int(logits[k].argmax(dim=1).item())
            result.update({f'true_{k}': row[k], f'{k}_state': _state(row[k], mapping[k]),
                           f'predicted_{k}': names[k][index]})
            if result[f'{k}_state'] == 'known':
                truth[k].append(mapping[k][row[k]])
                predictions[k].append(index)
        if all(result[f'{k}_state'] == 'known' for k in TASKS):
            joint_n += 1
            joint_correct += int(all(result[f'true_{k}'] == result[f'predicted_{k}'] for k in TASKS))
        results.append(result)
    count = len(rows)
    metrics = {k: {'eligible': len(truth[k]), 'total': count, 'coverage': len(truth[k]) / count,
                   'metrics': task_metrics(truth[k], predictions[k], names[k]) if truth[k] else None} for k in TASKS}
    report = {'protocol_version': VERSION, 'protocol_sha256': _hash(protocol_path),
              'checkpoint_sha256': _hash(checkpoint), 'annotation_inspection': inspect_external(rows, mapping),
              'exposure_check': 'train/val/test and warm-start ancestors checked for pixel hashes and groups; IDs checked where recorded',
              'tasks': metrics, 'joint': {'eligible': joint_n, 'total': count, 'coverage': joint_n / count,
                                         'correct': joint_correct, 'accuracy': joint_correct / joint_n if joint_n else None},
              'interpretation': 'Comparison against source annotations, not expert ground truth; unseen labels do not imply unknown recognition.'}
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    fields = ['image_id'] + [f'{kind}_{k}' if kind != 'state' else f'{k}_state'
                             for k in TASKS for kind in ('true', 'state', 'predicted')]
    for filename, values in (('predictions.csv', results),
                             ('missing_or_unseen.csv', [r for r in results if any(r[f'{k}_state'] != 'known' for k in TASKS)])):
        with (destination / filename).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    return report
