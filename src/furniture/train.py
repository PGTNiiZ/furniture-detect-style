import csv
import hashlib
import json
import random
import time
import subprocess
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from .data import FurnitureDataset, validate_splits
from .model import AttributeModel


def choose_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is not available')
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def metrics_for(truth, prediction, mapping):
    result = {}
    for key in mapping:
        labels = list(range(len(mapping[key])))
        names = list(mapping[key])
        matrix = confusion_matrix(truth[key], prediction[key], labels=labels)
        normalized = matrix.astype(float) / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        result[key] = {
            "accuracy": accuracy_score(truth[key], prediction[key]),
            "macro_precision": precision_score(truth[key], prediction[key], labels=labels, average='macro', zero_division=0),
            "macro_recall": recall_score(truth[key], prediction[key], labels=labels, average='macro', zero_division=0),
            "macro_f1": f1_score(truth[key], prediction[key], labels=labels, average="macro", zero_division=0),
            "weighted_f1": f1_score(truth[key], prediction[key], labels=labels, average="weighted", zero_division=0),
            "confusion_matrix": matrix.tolist(), "normalized_confusion_matrix": normalized.tolist(),
            "per_class": classification_report(truth[key], prediction[key], labels=labels,
                                                target_names=names, zero_division=0, output_dict=True)}
    return result


def run_epoch(model, loader, mapping, device, optimizer=None, frozen=False, loss_weights=None, include_predictions=False,
              accumulation_steps=1, scaler=None, amp=False):
    model.train(optimizer is not None)
    if optimizer is not None and frozen:
        model.backbone.eval()  # freeze BatchNorm running statistics as well as parameters
    truth, prediction = ({key: [] for key in mapping} for _ in range(2))
    confidence = {key: [] for key in mapping}
    total, seen = 0.0, 0
    task_totals = {key:0.0 for key in mapping}
    loss_weights = loss_weights or {key:1.0 for key in mapping}
    pending_samples = 0
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = {key: value.to(device) for key, value in labels.items()}
        with torch.set_grad_enabled(optimizer is not None), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(images)
            task_losses = {key:torch.nn.functional.cross_entropy(output[key], labels[key]) for key in mapping}
            loss = sum(loss_weights[key] * task_losses[key] for key in mapping)
            if optimizer is not None:
                scaled_loss = loss * images.size(0)
                if scaler is None:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()
                pending_samples += images.size(0)
                if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    for group in optimizer.param_groups:
                        for parameter in group['params']:
                            if parameter.grad is not None:
                                parameter.grad.div_(pending_samples)
                    if scaler is None:
                        optimizer.step()
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    pending_samples = 0
        seen += images.size(0)
        total += loss.item() * images.size(0)
        for key in mapping:
            task_totals[key] += task_losses[key].item() * images.size(0)
            truth[key].extend(labels[key].detach().cpu().tolist())
            prediction[key].extend(output[key].argmax(1).detach().cpu().tolist())
            if include_predictions:
                confidence[key].extend(output[key].softmax(1).max(1).values.detach().cpu().tolist())
    result = {"loss":total/seen,"task_losses":{key:v/seen for key,v in task_totals.items()},
              **metrics_for(truth, prediction, mapping)}
    if include_predictions:
        result['_predictions'] = {'truth':truth,'predicted':prediction,'confidence':confidence}
    return result


def fit(data, output, config, run_prepared=False, resume=False):
    started = time.perf_counter()
    seed_everything(config["seed"])
    root, output = Path(data).resolve(), Path(output)
    if output.exists() and not run_prepared:
        raise ValueError("Run directory exists; select a new output to avoid overwriting")
    if config["epochs"] < 1 or config["batch_size"] < 2:
        raise ValueError("epochs must be >=1 and batch_size >=2")
    splits, mapping = validate_splits(root)
    tasks = config.get('tasks', ['type', 'style'])
    if tasks not in (['type', 'style'], ['type'], ['style']):
        raise ValueError('Unsupported task set')
    mapping = {task: mapping[task] for task in tasks}
    if len(splits["train"]) < 2:
        raise ValueError("At least two training images required")
    device = choose_device(config["device"])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    amp = config.get('amp', False)
    accumulation_steps = config.get('accumulation_steps', 1)
    if amp and device.type != 'cuda':
        raise ValueError('AMP is supported only when CUDA is selected')
    if not isinstance(accumulation_steps, int) or isinstance(accumulation_steps, bool) or accumulation_steps < 1:
        raise ValueError('accumulation_steps must be a positive integer')
    scaler = torch.cuda.amp.GradScaler() if amp else None
    model = AttributeModel(mapping, config["pretrained"] and not resume, config.get('dropout',0.0), config.get('backbone','resnet18'), config.get('weights_name')).to(device)
    initialization = None if resume else initialize_from_checkpoint(model, mapping, splits, config, device)
    loaders = {s: DataLoader(FurnitureDataset(rows, root, mapping, config["image_size"],
                                            config["augment"] if s == "train" else False),
                             batch_size=config["batch_size"], shuffle=s == "train", num_workers=0,
                             drop_last=s == "train" and len(rows) % config["batch_size"] == 1)
               for s, rows in splits.items() if s != "test"}
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config["finetune_lr"]},
        {"params": model.heads.parameters(), "lr": config["head_lr"]}], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=2)
    output.mkdir(parents=True, exist_ok=run_prepared)
    split_hashes = {s: hashlib.sha256((root / f"{s}.csv").read_bytes()).hexdigest() for s in splits}
    best, stale, history, best_epoch = -1.0, 0, [], 0
    first_epoch = 0
    runtime = {'device':str(device), 'cuda_runtime':torch.version.cuda,
               'gpu_name':torch.cuda.get_device_name(device) if device.type == 'cuda' else None}
    if resume:
        last = torch.load(output/'last.pt', map_location=device, weights_only=True)
        if last.get('resume_protocol') != 'epoch-seed-v1':
            raise ValueError('Checkpoint does not support exact epoch-boundary resume')
        if last['config'] != config or last['mapping'] != mapping or last['split_hashes'] != split_hashes:
            raise ValueError('Resume config, mapping or dataset splits differ')
        if last.get('torch_version') != str(torch.__version__):
            raise ValueError('Resume requires the same PyTorch version')
        if last.get('runtime') != runtime:
            raise ValueError('Resume requires the same device and CUDA runtime')
        if (output/'environment.txt').read_bytes() != subprocess.check_output([__import__('sys').executable, '-m', 'pip', 'freeze']):
            raise ValueError('Resume requires the same installed package versions')
        model.load_state_dict(last['state_dict'], strict=True)
        model.weights_identifier = last.get('model_metadata', {}).get('weights')
        optimizer.load_state_dict(last['optimizer'])
        scheduler.load_state_dict(last['scheduler'])
        if scaler is not None:
            scaler.load_state_dict(last['scaler'])
        history, best, stale, best_epoch = last['history'], last['best'], last['stale'], last['best_epoch']
        initialization = last['initialization']
        model.previous_exposure = last['exposure_membership']
        first_epoch = last['epoch']
        if first_epoch >= config['epochs'] or stale >= config['patience']:
            raise ValueError('Checkpoint has already reached the configured stopping condition')
    else:
        (output / "config.json").write_text(json.dumps(config, indent=2))
        (output / "environment.txt").write_bytes(subprocess.check_output([__import__('sys').executable, '-m', 'pip', 'freeze']))
        dataset_metadata = {'split_hashes':split_hashes, 'mapping':mapping,
                            'audit':json.loads((root/'audit.json').read_text(encoding='utf-8')) if (root/'audit.json').exists() else None}
        (output/'dataset_metadata.json').write_text(json.dumps(dataset_metadata, indent=2),encoding='utf-8')
        environment = {'device':str(device), 'torch_version':str(torch.__version__), 'cuda_runtime':torch.version.cuda,
                       'gpu_name':torch.cuda.get_device_name(device) if device.type == 'cuda' else None}
        (output/'environment.json').write_text(json.dumps(environment, indent=2),encoding='utf-8')
    loss_weights = config.get('loss_weights', {'type':1.0,'style':1.0})
    metric_name = config.get('selection_metric', 'style_macro_f1')
    membership = {s:{'pixel_hashes':sorted({r['pixel_hash'] for r in rs}),
                     'group_ids':sorted({r.get('group_id') for r in rs if r.get('group_id')})}
                  for s,rs in splits.items()}
    inherited = getattr(model, 'previous_exposure', {})
    exposure = {s:{field:sorted(set(membership[s][field]) | set(inherited.get(s,{}).get(field,[])))
                   for field in ['pixel_hashes','group_ids']} for s in membership}
    for epoch in range(first_epoch, config["epochs"]):
        seed_everything(config['seed'] + epoch)
        for transform in loaders['train'].dataset.transform.transforms:
            if hasattr(transform, 'pipeline') and hasattr(transform.pipeline, 'set_random_seed'):
                transform.pipeline.set_random_seed(config['seed'] + epoch)
        frozen = epoch < config["freeze_epochs"]
        model.freeze_backbone(frozen)
        learning_rates = {'backbone':optimizer.param_groups[0]['lr'], 'heads':optimizer.param_groups[1]['lr']}
        train = run_epoch(model, loaders["train"], mapping, device, optimizer, frozen, loss_weights,
                          accumulation_steps=accumulation_steps, scaler=scaler, amp=amp)
        val = run_epoch(model, loaders["val"], mapping, device, loss_weights=loss_weights)
        score = sum(val[key]['macro_f1'] for key in mapping)/len(mapping) if metric_name == 'mean_macro_f1' else val['style']['macro_f1']
        scheduler.step(score)
        history.append({"epoch": epoch+1, "backbone_frozen":frozen, "train": train, "val": val,
                        'learning_rates': learning_rates,
                        'selection_metric':metric_name, 'selection_score':score})
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(f"epoch={epoch+1} loss={train['loss']:.4f} val_{metric_name}={score:.4f}", flush=True)
        if score > best:
            best, stale, best_epoch = score, 0, epoch+1
        else:
            stale += 1
        payload = {"state_dict": model.state_dict(), "mapping": mapping, "config": config,
                   'model_metadata': {'backbone':model.backbone_name, 'feature_dim':model.feature_dim,
                                      'weights':model.weights_identifier},
                   'preprocessing': {'image_size':config['image_size'], 'color':'EXIF-corrected RGB',
                                     'resize':'aspect-preserving PIL bicubic pad', 'pad_rgb':[255,255,255],
                                     'mean':[.485,.456,.406], 'std':[.229,.224,.225],
                                     'train_order':'resize/pad, augmentation, occlusion, tensor, normalize',
                                     'evaluation_order':'resize/pad, tensor, normalize',
                                     'protocol':'controlled ImageNet normalization, not necessarily official weights recipe'},
                   "epoch": epoch+1, "val": val, "split_hashes": split_hashes,
                   "data_membership": membership, "exposure_membership":exposure, "initialization":initialization,
                   "shared_backbone_update_epochs":sum(not h['backbone_frozen'] for h in history),
                   "audit": json.loads((root / "audit.json").read_text()) if (root / "audit.json").exists() else {}}
        if best_epoch == epoch+1:
            atomic_torch_save(payload, output / "best.pt")
        atomic_torch_save({**payload, 'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict(),
                           'history':history, 'best':best, 'best_epoch':best_epoch, 'stale':stale,
                           'scaler':scaler.state_dict() if scaler is not None else None,
                           'resume_protocol':'epoch-seed-v1', 'torch_version':str(torch.__version__), 'runtime':runtime}, output / 'last.pt')
        if stale >= config["patience"]:
            break
    (output / "checkpoint.sha256").write_text(hashlib.sha256((output / "best.pt").read_bytes()).hexdigest())
    summary = {"best_validation_score":best,"selection_metric":metric_name,"best_epoch":best_epoch,
               "epochs":len(history),"device":str(device),'initialization':initialization,
               'physical_batch_size':config['batch_size'], 'effective_batch_size':config['batch_size'] * accumulation_steps,
               'elapsed_seconds_this_invocation':time.perf_counter()-started,
               'peak_gpu_bytes':torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
               'stop_reason':'early_stopping' if stale >= config['patience'] else 'max_epochs',
               "shared_backbone_update_epochs":sum(not h['backbone_frozen'] for h in history)}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def atomic_torch_save(payload, path):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.ck-', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_model(checkpoint, device):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model = AttributeModel(payload["mapping"], pretrained=False, dropout=payload["config"].get("dropout",0.0), backbone=payload["config"].get("backbone",'resnet18')).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def evaluate(data, checkpoint, output, split="test", device="auto"):
    device = choose_device(device)
    model, payload = load_model(checkpoint, device)
    root = Path(data)
    rows, mapping = validate_splits(root)
    mapping = {task: mapping[task] for task in payload['mapping']}
    if mapping != payload["mapping"]:
        raise ValueError("Label mapping differs from checkpoint")
    for name, expected in payload["split_hashes"].items():
        if hashlib.sha256((root / f"{name}.csv").read_bytes()).hexdigest() != expected:
            raise ValueError("Split manifests differ from training checkpoint")
    loader = DataLoader(FurnitureDataset(rows[split], root, mapping, payload["config"]["image_size"]),
                        batch_size=payload["config"]["batch_size"])
    result = run_epoch(model, loader, mapping, device, loss_weights=payload['config'].get('loss_weights'), include_predictions=True)
    details = result.pop('_predictions')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2))
    names = {key:list(values) for key,values in mapping.items()}
    with Path(output).with_name(Path(output).stem+'_predictions.csv').open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream, ['image_id','image_path'] + [f'{prefix}_{key}' for key in mapping for prefix in ('true','pred','confidence')])
        writer.writeheader()
        for i,row in enumerate(rows[split]):
            writer.writerow({'image_id':row['image_id'],'image_path':row['image_path'],
                             **{f'true_{key}':names[key][details['truth'][key][i]] for key in mapping},
                             **{f'pred_{key}':names[key][details['predicted'][key][i]] for key in mapping},
                             **{f'confidence_{key}':details['confidence'][key][i] for key in mapping}})
    for key in mapping:
        with Path(output).with_name(Path(output).stem+f'_{key}_confusion.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(['true / predicted',*names[key]])
            for name, row in zip(names[key], result[key]['confusion_matrix']):
                writer.writerow([name,*row])
    return result


def initialize_from_checkpoint(model, mapping, splits, config, device):
    if config.get('initialization') != 'checkpoint':
        return {'mode':'imagenet' if config['pretrained'] else 'scratch'}
    path = Path(config['init_checkpoint'])
    parent = torch.load(path, map_location=device, weights_only=True)
    if parent.get('config', {}).get('backbone', 'resnet18') != model.backbone_name:
        raise ValueError('Checkpoint backbone differs from selected backbone')
    if 'data_membership' not in parent:
        raise ValueError('Checkpoint lacks split membership; use new-format checkpoints for audited transfer')
    previous = parent.get('exposure_membership',parent['data_membership'])
    for field, rowkey in [('pixel_hashes','pixel_hash'),('group_ids','group_id')]:
        for old_split in ['train','val','test']:
            old_values = set(previous[old_split][field])
            for new_split in ['train','val','test']:
                if old_split == new_split:
                    continue
                new_values = {r.get(rowkey) for r in splits[new_split] if r.get(rowkey)}
                if old_values & new_values:
                    raise ValueError('Warm-start leakage: previous training/holdout membership moved across splits')
    model.previous_exposure = previous
    scope = config.get('checkpoint_scope','full')
    if scope == 'full':
        if parent['mapping'] != mapping:
            raise ValueError('Full checkpoint requires identical label mapping; use checkpoint_scope: backbone for new classes')
        model.load_state_dict(parent['state_dict'],strict=True)
    elif scope == 'backbone':
        state = {key.removeprefix('backbone.'):value for key,value in parent['state_dict'].items() if key.startswith('backbone.')}
        model.backbone.load_state_dict(state,strict=True)
    else:
        raise ValueError('Unknown checkpoint scope')
    return {'mode':'warm_start','scope':scope,'parent_checkpoint':str(path),
            'parent_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'optimizer':'fresh; not an exact resume'}
