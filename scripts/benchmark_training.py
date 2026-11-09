"""Time short train-only passes; never read held-out examples or save model weights."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from furniture.config import load_config, resolve_path, training_settings
from furniture.data import FurnitureDataset, read_rows
from furniture.model import AttributeModel
from furniture.train import choose_device, run_epoch, seed_everything


def benchmark(project, backbone, output, batch_size=32, batches=3, device='auto', allow_unvalidated_train=False):
    path = Path(output)
    if path.exists():
        raise ValueError('Benchmark output already exists')
    if batch_size < 2 or batches < 1:
        raise ValueError('batch size must be >=2 and batches >=1')
    cfg = load_config(project, 'C3')
    cfg['model']['backbone'] = backbone
    from furniture.model import WEIGHT_NAMES
    if backbone not in WEIGHT_NAMES:
        raise ValueError('Unknown backbone')
    cfg['model']['weights'] = WEIGHT_NAMES[backbone]
    settings = training_settings(cfg)
    settings['batch_size'] = batch_size
    root = resolve_path(cfg, cfg['dataset']['root'])
    manifest = root/'train.csv'
    train_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows = read_rows(manifest)
    meta_path = root/'dataset_metadata.json'
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if train_hash != meta['split_hashes']['train']:
            raise ValueError('Train manifest does not match dataset metadata')
        mapping = meta['class_mapping']
        audit = meta.get('training_audit', {})
        audit_status = audit.get('status', 'snapshot metadata present; verify audit separately')
    elif allow_unvalidated_train:
        mapping = {key:{label:i for i,label in enumerate(sorted({r[key] for r in rows if r.get(key)}))}
                   for key in ('type','style')}
        if not all(mapping.values()) or any(not r.get('type') or not r.get('style') for r in rows):
            raise ValueError('Train labels are incomplete')
        audit_status = 'unvalidated partial snapshot; timing only, not a training protocol'
    else:
        raise ValueError('Dataset metadata is missing; benchmark cannot assume the snapshot passed audit')
    if len(rows) < batch_size * (batches + 1):
        raise ValueError('Not enough training images for requested warm-up and measured batches')
    seed_everything(settings['seed'])
    selected = choose_device(device)
    if settings['amp'] and selected.type != 'cuda':
        raise ValueError('Configured AMP benchmark requires CUDA')
    scaler = torch.cuda.amp.GradScaler() if settings['amp'] else None
    model = AttributeModel(mapping, pretrained=True, dropout=settings['dropout'], backbone=backbone,
                           weights_name=settings['weights_name']).to(selected)
    # Model initialization consumes a different amount of randomness for each backbone.
    # Restart the data RNG so every timing run sees the same training examples.
    seed_everything(settings['seed'])
    dataset = FurnitureDataset(rows, root, mapping, settings['image_size'], settings['augment'])
    generator = torch.Generator().manual_seed(settings['seed'])
    indices = torch.randperm(len(rows), generator=generator).tolist()[:batch_size * (batches + 1)]
    warm = DataLoader(Subset(dataset, indices[:batch_size]), batch_size=batch_size, num_workers=0)
    measured = DataLoader(Subset(dataset, indices[batch_size:]), batch_size=batch_size, num_workers=0)
    optimizer = torch.optim.AdamW([{'params':model.backbone.parameters(), 'lr':settings['finetune_lr']},
                                   {'params':model.heads.parameters(), 'lr':settings['head_lr']}],
                                  weight_decay=settings['weight_decay'])
    phases = {}
    for frozen in (True, False):
        model.freeze_backbone(frozen)
        if selected.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(selected)
        run_epoch(model, warm, mapping, selected, optimizer, frozen, settings['loss_weights'],
                  accumulation_steps=settings['accumulation_steps'], scaler=scaler, amp=settings['amp'])
        if selected.type == 'cuda': torch.cuda.synchronize(selected)
        start = time.perf_counter()
        run_epoch(model, measured, mapping, selected, optimizer, frozen, settings['loss_weights'],
                  accumulation_steps=settings['accumulation_steps'], scaler=scaler, amp=settings['amp'])
        if selected.type == 'cuda': torch.cuda.synchronize(selected)
        elapsed = time.perf_counter()-start
        samples = batch_size*batches
        phases['frozen' if frozen else 'unfrozen'] = {
            'seconds':elapsed, 'measured_samples':samples, 'samples_per_second':samples/elapsed,
            'linear_epoch_estimate_seconds':elapsed*len(rows)/samples,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated(selected) if selected.type == 'cuda' else None}
    result = {'dataset_id':cfg['dataset']['id'], 'train_manifest_sha256':train_hash,
              'train_rows':len(rows), 'backbone':backbone, 'weights':model.weights_identifier,
              'device':str(selected), 'gpu':torch.cuda.get_device_name(selected) if selected.type=='cuda' else None,
              'torch':str(torch.__version__), 'batch_size':batch_size, 'warmup_batches_per_phase':1,
              'accumulation_steps':settings['accumulation_steps'],
              'amp':settings['amp'],
              'dataset_audit_status':audit_status,
              'sampling_seed':settings['seed'],
              'sample_ids':[rows[i]['image_id'] for i in indices],
              'measured_batches_per_phase':batches, 'phases':phases,
              'note':'Train-only timing with fresh pretrained model; no evaluation or saved checkpoint. '
                     'Linear estimate ignores validation, checkpoint writes and startup; not a full-suite forecast.'}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/project_controlled.yaml')
    parser.add_argument('--backbone',default='resnet18')
    parser.add_argument('--output',required=True)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--batches',type=int,default=3)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--allow-unvalidated-train',action='store_true',help='Time train only when audit is incomplete; output cannot justify screening')
    args = parser.parse_args()
    print(json.dumps(benchmark(args.config,args.backbone,args.output,args.batch_size,args.batches,args.device,args.allow_unvalidated_train),indent=2))
