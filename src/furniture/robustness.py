"""Reproducible fixed-occlusion evaluation, independent of training transforms."""
import csv
from copy import deepcopy
import hashlib
import json
import random
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torchvision.transforms import functional as F

from .data import Square, validate_splits
from .config import digest, training_settings
from .train import choose_device, load_model


LEVELS = {'light': (1, 2, .05, .15), 'strong': (1, 3, .10, .25)}
MEAN, STD = [.485, .456, .406], [.229, .224, .225]
MODEL_KEYS = {'backbone','weights','initialization','checkpoint','checkpoint_scope','image_size','dropout'}
TRAINING_KEYS = {'seed','batch_size','epochs','freeze_epochs','head_lr','finetune_lr','weight_decay',
                 'patience','device','loss_weights','selection_metric','tasks','accumulation_steps','amp'}


def split_digest(root, split):
    return hashlib.sha256((Path(root) / f'{split}.csv').read_bytes()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_manifest(rows, split, split_hash, image_size, seed=42, fill=(127, 127, 127)):
    """Derive each sample's rectangles from its ID, so ordering does not change them."""
    if image_size < 1 or not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('image_size must be positive and seed must be an integer')
    if len(fill) != 3 or any(not isinstance(c, int) or isinstance(c, bool) or not 0 <= c <= 255 for c in fill):
        raise ValueError('fill must be three integers in [0, 255]')
    samples = []
    for row in sorted(rows, key=lambda r: r['image_id']):
        variants = {}
        for level, (minimum, maximum, low, high) in LEVELS.items():
            seed_material = json.dumps([seed, row['image_id'], level], ensure_ascii=False).encode('utf-8')
            rng = random.Random(int.from_bytes(hashlib.sha256(seed_material).digest(), 'big'))
            holes = []
            for _ in range(rng.randint(minimum, maximum)):
                width = max(1, round(image_size * rng.uniform(low, high)))
                height = max(1, round(image_size * rng.uniform(low, high)))
                x, y = rng.randint(0, image_size - width), rng.randint(0, image_size - height)
                holes.append([x, y, x + width, y + height])  # right and bottom are exclusive
            variants[level] = holes
        samples.append({'image_id':row['image_id'], 'pixel_hash':row['pixel_hash'], 'variants':variants})
    return {'version':1, 'split':split, 'split_sha256':split_hash, 'seed':seed,
            'image_size':image_size, 'fill_rgb':list(fill), 'samples':samples,
            'preprocessing':{'convert':'EXIF transpose, RGB; alpha composited over white', 'resize':'aspect-preserving pad', 'interpolation':'PIL bicubic',
                             'pad_rgb':[255,255,255], 'normalization_mean':MEAN, 'normalization_std':STD,
                             'order':['resize_pad','occlude','to_tensor','normalize']},
            'coordinates':'xyxy with exclusive right and bottom; after aspect-preserving resize/pad',
            'note':'Occlusions always applied; transformed versions of the same split, not independent data. '
                   'Without object masks, image coverage does not establish object coverage.'}


def apply_holes(image, holes, fill):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for x1, y1, x2, y2 in holes:
        draw.rectangle((x1, y1, x2-1, y2-1), fill=tuple(fill))
    return result


def task_metrics(truth, predictions, names):
    labels = list(range(len(names)))
    counts = confusion_matrix(truth, predictions, labels=labels)
    totals = counts.sum(axis=1, keepdims=True)
    normalized = counts.astype(float) / totals.clip(min=1)
    return {'accuracy':accuracy_score(truth, predictions),
            'macro_precision':precision_score(truth, predictions, labels=labels, average='macro', zero_division=0),
            'macro_recall':recall_score(truth, predictions, labels=labels, average='macro', zero_division=0),
            'macro_f1':f1_score(truth, predictions, labels=labels, average='macro', zero_division=0),
            'weighted_f1':f1_score(truth, predictions, labels=labels, average='weighted', zero_division=0),
            'labels':names, 'confusion_matrix':counts.tolist(), 'normalized_confusion_matrix':normalized.tolist(),
            'per_class':classification_report(truth, predictions, labels=labels, target_names=names,
                                               output_dict=True, zero_division=0),
            'zero_division':0}


def training_evidence(checkpoint, payload=None):
    """Bind controls to saved provenance and the validation-selected checkpoint."""
    run = Path(checkpoint).parent
    status_path, config_path = run/'status.json', run/'resolved_config.yaml'
    if not status_path.is_file() or not config_path.is_file():
        return {'status':'unknown', 'reason':'Run status or resolved configuration missing'}
    try:
        status = json.loads(status_path.read_text(encoding='utf-8'))
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        if Path(checkpoint).resolve() != (run/'best.pt').resolve():
            raise ValueError('only best.pt is eligible for comparisons')
        provenance = json.loads((run/'provenance.json').read_text(encoding='utf-8'))
        config_sha = digest(config)
        if provenance['config_sha256'] != config_sha:
            raise ValueError('resolved configuration differs from provenance')
        source_hashes = {p.name:file_digest(p) for p in sorted((run/'source').glob('*.py'))}
        if not source_hashes or source_hashes != provenance['code_sha256']:
            raise ValueError('source snapshot differs from provenance')
        checkpoint_sha = file_digest(checkpoint)
        if (run/'checkpoint.sha256').read_text(encoding='utf-8').strip() != checkpoint_sha:
            raise ValueError('best checkpoint checksum differs')
        payload = payload if payload is not None else torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload['config'] != training_settings(config):
            raise ValueError('checkpoint training settings differ from resolved configuration')
        summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
        if payload['epoch'] != summary['best_epoch']:
            raise ValueError('checkpoint epoch is not the selected validation epoch')
        controls = {key:config[key] for key in ('dataset','model','training','augmentation')}
        if any(not isinstance(value, dict) for value in controls.values()):
            raise ValueError('dataset/model/training/augmentation must be mappings')
        if not MODEL_KEYS <= controls['model'].keys() or not TRAINING_KEYS <= controls['training'].keys():
            raise ValueError('model and training controls must be complete')
        if not {'id','source','split'} <= controls['dataset'].keys() or not status.get('device'):
            raise ValueError('dataset identity and selected device must be recorded')
        if not controls['augmentation']:
            raise ValueError('augmentation controls must be recorded')
        controls['selected_device'] = status.get('device')
        return {'status':status.get('status','unknown'),
                'augmentation_protocol':status.get('augmentation_protocol'), 'evidence_version':3, 'controls':controls,
                'resolved_config':config, 'config_sha256':config_sha, 'checkpoint_sha256':checkpoint_sha,
                'checkpoint_selection':'best_validation', 'best_epoch':payload['epoch']}
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, yaml.YAMLError) as error:
        return {'status':'unknown', 'reason':f'Cannot verify training evidence: {error}'}


def evaluate_robustness(data, checkpoint, output, split='val', device='auto', seed=42, manifest=None):
    """Evaluate one checkpoint with fixed clean/light/strong variants; output must be new."""
    if split not in ('val', 'test'):
        raise ValueError('Robustness split must be val or test')
    root, checkpoint, output = Path(data), Path(checkpoint), Path(output)
    if output.exists():
        raise ValueError('Output directory exists; choose a new directory')
    selected_device = choose_device(device)
    model, payload = load_model(checkpoint, selected_device)
    splits, mapping = validate_splits(root)
    checkpoint_mapping = payload['mapping']
    if any(mapping.get(k) != v for k, v in checkpoint_mapping.items()):
        raise ValueError('Label mapping differs from checkpoint')
    for name, expected in payload['split_hashes'].items():
        if split_digest(root, name) != expected:
            raise ValueError('Split manifests differ from training checkpoint')
    image_size = payload['config']['image_size']
    expected = make_manifest(splits[split], split, split_digest(root, split), image_size, seed)
    if manifest is not None:
        supplied = json.loads(Path(manifest).read_text(encoding='utf-8'))
        if supplied != expected:
            raise ValueError('Occlusion manifest differs from requested dataset, split, size, seed or algorithm')
        expected = supplied
    manifest_text = json.dumps(expected, ensure_ascii=False, sort_keys=True, indent=2)
    manifest_hash = hashlib.sha256(manifest_text.encode('utf-8')).hexdigest()
    rows_by_id = {row['image_id']: row for row in splits[split]}
    records, metrics = {}, {}
    model.eval()
    for level in ('clean', 'light', 'strong'):
        truth = {k:[] for k in checkpoint_mapping}
        predicted = {k:[] for k in checkpoint_mapping}
        records[level] = []
        for sample in expected['samples']:
            row = rows_by_id[sample['image_id']]
            with Image.open(root / row['image_path']) as source:
                image = Square(image_size)(source)
            if level != 'clean':
                image = apply_holes(image, sample['variants'][level], expected['fill_rgb'])
            tensor = F.normalize(F.to_tensor(image), MEAN, STD).unsqueeze(0).to(selected_device)
            with torch.inference_mode():
                logits = model(tensor)
            record = {'image_id':row['image_id'], 'image_path':row['image_path']}
            for key, labels in checkpoint_mapping.items():
                probabilities = logits[key].softmax(dim=1)[0]
                best = int(probabilities.argmax().item())
                label_by_index = {i:name for name,i in labels.items()}
                truth[key].append(labels[row[key]])
                predicted[key].append(best)
                record.update({f'true_{key}':row[key], f'pred_{key}':label_by_index[best],
                               f'confidence_{key}':float(probabilities[best].item())})
            records[level].append(record)
        metrics[level] = {key:task_metrics(truth[key], predicted[key],
                          [name for name, _ in sorted(labels.items(), key=lambda item:item[1])])
                          for key, labels in checkpoint_mapping.items()}
    drops = {level:{key:{metric:metrics['clean'][key][metric]-metrics[level][key][metric]
                         for metric in ('accuracy','macro_f1')}
                    for key in checkpoint_mapping} for level in ('light','strong')}
    result = {'split':split, 'seed':seed, 'manifest_sha256':manifest_hash,
              'split_hashes':payload['split_hashes'], 'mapping':checkpoint_mapping,
              'training_evidence':training_evidence(checkpoint, payload),
              'checkpoint_sha256':file_digest(checkpoint),
              'device':str(selected_device), 'metrics':metrics, 'absolute_drop':drops,
              'note':expected['note']}
    output.mkdir(parents=True)
    (output/'occlusion_manifest.json').write_text(manifest_text, encoding='utf-8')
    (output/'robustness.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for level, values in records.items():
        with (output/f'predictions_{level}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    return result


def compare_robustness(result_dirs, output, vary=()):
    """Compare only declared axes while preserving data, mapping and image-variant guards."""
    if not result_dirs:
        raise ValueError('At least one robustness result is required')
    allowed = {'backbone','weights','augmentation','occlusion','training_mode','tasks','seed','device'}
    if isinstance(vary, str) or len(vary) != len(set(vary)) or set(vary) - allowed:
        raise ValueError('Comparison axes must be unique and supported')
    axes = set(vary)
    table, reference, mappings, weights_by_backbone = [], None, {}, {}
    for directory in result_dirs:
        directory = Path(directory)
        result = json.loads((directory/'robustness.json').read_text(encoding='utf-8'))
        manifest = json.loads((directory/'occlusion_manifest.json').read_text(encoding='utf-8'))
        evidence = result.get('training_evidence', {})
        if evidence.get('status') != 'completed' or evidence.get('augmentation_protocol') != 'post-p0-fix-v1':
            raise ValueError(f'Run lacks completed post-P0 training evidence: {directory}')
        if evidence.get('evidence_version') != 3 or not isinstance(evidence.get('controls'), dict):
            raise ValueError(f'Run lacks complete comparison controls; reevaluate checkpoint: {directory}')
        recorded_config = evidence.get('resolved_config')
        if not isinstance(recorded_config, dict) or digest(recorded_config) != evidence.get('config_sha256'):
            raise ValueError(f'Resolved configuration evidence is invalid: {directory}')
        if evidence.get('checkpoint_selection') != 'best_validation' or evidence.get('checkpoint_sha256') != result.get('checkpoint_sha256'):
            raise ValueError(f'Checkpoint selection evidence is invalid: {directory}')
        manifest_hash = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode('utf-8')).hexdigest()
        if manifest_hash != result['manifest_sha256']:
            raise ValueError(f'Manifest hash mismatch: {directory}')
        controls = deepcopy(evidence['controls'])
        if not all(isinstance(controls.get(key), dict) for key in ('dataset','model','training','augmentation')):
            raise ValueError(f'Run lacks complete comparison controls: {directory}')
        if any(controls[key] != recorded_config.get(key) for key in ('dataset','model','training','augmentation')):
            raise ValueError(f'Comparison controls differ from resolved configuration evidence: {directory}')
        model, training, augmentation = controls['model'], controls['training'], controls['augmentation']
        if not MODEL_KEYS <= model.keys() or not TRAINING_KEYS <= training.keys() or not {'id','source','split'} <= controls['dataset'].keys() or not controls.get('selected_device') or not augmentation:
            raise ValueError(f'Run lacks complete comparison controls: {directory}')
        if not isinstance(training['tasks'], list) or set(training['tasks']) != set(result['mapping']):
            raise ValueError('Training task set differs from evaluation mapping')
        # File locations are not experimental axes; identity and split hashes remain enforced.
        controls['dataset'].pop('root', None)
        if isinstance(controls['dataset'].get('source'), dict):
            for key in ('csv_file','image_root'):
                controls['dataset']['source'].pop(key, None)
        if isinstance(controls['dataset'].get('split'), dict):
            controls['dataset']['split'].pop('previous_root', None)
        backbone, weights = model['backbone'], model['weights']
        if 'weights' not in axes:
            if backbone in weights_by_backbone and weights_by_backbone[backbone] != weights:
                raise ValueError('Weights differ within a backbone; explicitly vary weights')
            weights_by_backbone[backbone] = weights
        if 'backbone' in axes:
            model.pop('backbone')
            model.pop('weights')
        elif 'weights' in axes:
            model.pop('weights')
        for axis, key in [('training_mode','freeze_epochs'), ('seed','seed'), ('tasks','tasks'), ('device','device')]:
            if axis in axes:
                training.pop(key)
        if 'device' in axes:
            controls.pop('selected_device',None)
        if 'occlusion' in axes:
            augmentation.pop('occlusion',None)
        if 'augmentation' in axes:
            controls['augmentation'] = {'occlusion':augmentation.get('occlusion')} if 'occlusion' not in axes else None
        if 'tasks' in axes:
            for task, values in result['mapping'].items():
                if task in mappings and mappings[task] != values:
                    raise ValueError('Results differ in label mapping for a shared task')
                mappings[task] = values
            mapping_control = None
        else:
            mapping_control = result['mapping']
        control = (result['split'], manifest_hash,
                   json.dumps(result['split_hashes'], sort_keys=True),
                   json.dumps(mapping_control, sort_keys=True),
                   json.dumps(controls, sort_keys=True))
        if reference is None:
            reference = control
        if control != reference:
            raise ValueError('Results differ in dataset split hashes, mapping, training controls or occlusion manifest')
        for task, clean in result['metrics']['clean'].items():
            table.append({'result_dir':str(directory), 'checkpoint_sha256':result['checkpoint_sha256'],
                          'split':result['split'], 'backbone':backbone, 'weights':weights,
                          'initialization':evidence['controls']['model']['initialization'],
                          'training_seed':evidence['controls']['training']['seed'], 'task':task,
                          'clean_macro_f1':clean['macro_f1'],
                          'light_macro_f1':result['metrics']['light'][task]['macro_f1'],
                          'strong_macro_f1':result['metrics']['strong'][task]['macro_f1'],
                          'light_absolute_drop':result['absolute_drop']['light'][task]['macro_f1'],
                          'strong_absolute_drop':result['absolute_drop']['strong'][task]['macro_f1']})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    return {'split':reference[0], 'manifest_sha256':reference[1], 'vary':list(vary), 'rows':table,
            'note':'Completed post-P0 runs; non-varying controls match. Different task sets have per-task scores only; no ranking is performed.'}
