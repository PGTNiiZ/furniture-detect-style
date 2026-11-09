"""Verify pretrained backward and AdamW update on audited training data only."""
import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from furniture.config import load_config, training_settings
from furniture.data import FurnitureDataset
from furniture.model import AttributeModel, BACKBONES, WEIGHT_NAMES
from furniture.snapshot import validate_audited_snapshot
from furniture.train import choose_device, seed_everything


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_snapshot(root):
    splits, mapping, metadata = validate_audited_snapshot(root)
    return splits['train'], mapping, metadata


def run(dataset, project, output, device='auto', batch_size=2):
    root, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError('Output already exists')
    if not isinstance(batch_size,int) or isinstance(batch_size,bool) or batch_size < 2:
        raise ValueError('batch_size must be >= 2')
    rows, mapping, metadata = validate_training_snapshot(root)
    if len(rows) < batch_size:
        raise ValueError('Not enough training samples')
    cfg = load_config(project, 'C3')
    settings = training_settings(cfg)
    if settings['image_size'] != 224:
        raise ValueError('Controlled comparison requires 224 input')
    chosen = choose_device(device)
    seed_everything(settings['seed'])
    dataset_obj = FurnitureDataset(rows[:batch_size], root, mapping, settings['image_size'], settings['augment'])
    images, labels = next(iter(DataLoader(dataset_obj,batch_size=batch_size,num_workers=0)))
    images = images.to(chosen)
    labels = {key:value.to(chosen) for key,value in labels.items()}
    report = {'scope':'one pretrained backward and AdamW update on audited train only; no metrics or checkpoint',
              'snapshot_protocol':metadata['dataset_protocol_version'], 'train_manifest_sha256':file_sha(root/'train.csv'),
              'training_audit':metadata['training_audit'], 'sample_ids':[r['image_id'] for r in rows[:batch_size]],
              'device':str(chosen), 'torch':str(torch.__version__), 'batch_size':batch_size, 'models':{}}
    output.parent.mkdir(parents=True, exist_ok=True)
    for name in BACKBONES:
        model = optimizer = None
        logits = loss = parameter = before = None
        try:
            seed_everything(settings['seed'])
            if chosen.type == 'cuda': torch.cuda.reset_peak_memory_stats(chosen)
            model = AttributeModel(mapping, pretrained=True, backbone=name, weights_name=WEIGHT_NAMES[name]).to(chosen)
            model.train()
            optimizer = torch.optim.AdamW([{'params':model.backbone.parameters(), 'lr':settings['finetune_lr']},
                                           {'params':model.heads.parameters(), 'lr':settings['head_lr']}],
                                          weight_decay=settings['weight_decay'])
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = sum(settings['loss_weights'][key] * torch.nn.functional.cross_entropy(logits[key],labels[key]) for key in mapping)
            if not torch.isfinite(loss).item(): raise ValueError('Loss is non-finite')
            loss.backward()
            if any(model.heads[f'{key}_head'].weight.grad is None or not torch.isfinite(model.heads[f'{key}_head'].weight.grad).all().item()
                   or model.heads[f'{key}_head'].weight.grad.abs().sum().item() == 0 for key in mapping):
                raise ValueError('A task head did not receive a finite nonzero gradient')
            parameter = next((p for p in model.backbone.parameters() if p.grad is not None and torch.isfinite(p.grad).all().item()
                              and p.grad.abs().sum().item() > 0), None)
            if parameter is None: raise ValueError('No finite nonzero backbone gradient')
            before = parameter.detach().clone()
            optimizer.step()
            if torch.equal(before,parameter.detach()): raise ValueError('Backbone parameter did not change')
            if chosen.type == 'cuda': torch.cuda.synchronize(chosen)
            report['models'][name] = {'status':'passed','weights':model.weights_identifier,
                                      'feature_dim':model.feature_dim,
                                      'peak_gpu_bytes':torch.cuda.max_memory_allocated(chosen) if chosen.type == 'cuda' else None}
        except Exception as error:
            report['models'][name] = {'status':'failed','error':f'{type(error).__name__}: {error}'}
        finally:
            del optimizer, model, logits, loss, parameter, before
            gc.collect()
            if chosen.type == 'cuda': torch.cuda.empty_cache()
        temporary=output.with_suffix(output.suffix+'.tmp')
        temporary.write_text(json.dumps(report,indent=2),encoding='utf-8')
        for attempt in range(5):
            try:
                os.replace(temporary,output)
                break
            except PermissionError:
                if attempt == 4: raise
                time.sleep(.2*(attempt+1))
        print(f"{name}: {report['models'][name]['status']}",flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--config',default='configs/project_controlled.yaml')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--batch-size',type=int,default=2)
    args=parser.parse_args()
    result=run(args.dataset,args.config,args.output,args.device,args.batch_size)
    raise SystemExit(0 if all(r['status']=='passed' for r in result['models'].values()) else 1)
