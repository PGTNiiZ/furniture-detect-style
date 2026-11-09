"""Separate, resumable validation-only ZIP pilot using frozen v2 training primitives."""
import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset
from furniture.data import FurnitureDataset, image_hash
from furniture.model import AttributeModel
from furniture.pipeline import _process_alive
from furniture.suite import _suite_lock
from furniture.train import atomic_torch_save, run_epoch, seed_everything
from new_data.zip_dataset import ZipFurnitureDataset


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf8',newline='') as f: return list(csv.DictReader(f))


class TimedLoader:
    def __init__(self, loader): self.loader,self.wait = loader,0.0
    def __len__(self): return len(self.loader)
    def __iter__(self):
        iterator=iter(self.loader)
        while True:
            start=time.perf_counter()
            try: batch=next(iterator)
            except StopIteration: return
            self.wait+=time.perf_counter()-start
            yield batch


@contextmanager
def record_failure(output, status):
    try:
        yield
    except Exception as error:
        event={'event':'failed','error_type':type(error).__name__,'error':str(error),
               'pid':os.getpid(),'time_ns':time.time_ns()}
        with (output/f"failure-{event['time_ns']}-{event['pid']}.json").open('x',encoding='utf8') as f:
            json.dump(event,f,indent=2)
        status('failed',error_type=event['error_type'],error=event['error'],event_file=f"failure-{event['time_ns']}-{event['pid']}.json")
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['smoke','pilot'],required=True)
    parser.add_argument('--wait-for-v2',action='store_true'); parser.add_argument('--workflow-pid',type=int)
    parser.add_argument('--post-pid',type=int); parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); config_path=Path(args.config)
    cfg=json.loads(config_path.read_text(encoding='utf8')); snap=Path(cfg['snapshot']); root=Path(cfg['v2_root'])
    archive=Path(os.path.expandvars(cfg['archive_path'])).resolve(); cfgsha=sha(config_path)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=args.resume)
    def status(stage, **items):
        tmp=output/'status.json.tmp'
        tmp.write_text(json.dumps({'stage':stage,**items},indent=2),encoding='utf8'); os.replace(tmp,output/'status.json')
    with _suite_lock(output/'pilot.lock'), record_failure(output,status):
        audit=json.loads((snap/'audit.json').read_text(encoding='utf8'))
        verification_path=snap/cfg['verification_file']
        verify=json.loads(verification_path.read_text(encoding='utf8'))
        if verify['status']!='pass_conservative_group_checks' or sha(archive)!=audit['archive_sha256']:
            raise ValueError('Snapshot or archive is not verified')
        for name,expected in audit['manifest_sha256'].items():
            if sha(snap/name)!=expected: raise ValueError(f'Snapshot file changed: {name}')
        for name,expected in audit['holdout_manifest_sha256'].items():
            if sha(root/(name+'.csv'))!=expected: raise ValueError(f'Holdout manifest changed: {name}')
        selected=json.loads(Path(cfg['baseline_selection']).read_text(encoding='utf8'))['selected']
        if (selected['backbone'],selected['condition'])!=(cfg['backbone'],cfg['condition']):
            raise ValueError('Baseline selection differs from pilot config')
        baseline_config=Path(selected['run_dir'])/'config.json'
        baseline_checkpoint=Path(selected['run_dir'])/'best.pt'
        if sha(baseline_checkpoint)!=selected['checkpoint_sha256']:
            raise ValueError('Baseline checkpoint checksum differs from selection')
        base=json.loads(baseline_config.read_text(encoding='utf8'))
        aug=yaml.safe_load(Path(cfg['augmentation']).read_text(encoding='utf8'))
        for key in ('seed','batch_size','freeze_epochs','head_lr','finetune_lr','weight_decay','loss_weights','image_size','backbone'):
            if cfg[key]!=base[key]: raise ValueError(f'Pilot differs from baseline setting: {key}')
        if aug!=base['augment'] or base['amp'] or base['accumulation_steps']!=1:
            raise ValueError('Augmentation or optimizer control differs from baseline')
        mapping=json.loads((snap/'class-mapping.json').read_text(encoding='utf8'))
        if mapping!=json.loads((root/'audit.json').read_text(encoding='utf8'))['class_mapping']:
            raise ValueError('Class mappings differ; design eligible subset first')
        weights=Path(cfg['torch_home'])/'hub/checkpoints/densenet121-a639ec97.pth'
        if not weights.is_file() or sha(weights)!=cfg['weights_sha256']:
            raise ValueError('Pinned ImageNet weights are unavailable or changed')
        source_files=[*sorted(Path('src/furniture').glob('*.py')),Path('new_data/zip_dataset.py'),Path(__file__)]
        def capture_provenance():
            return {'config_sha256':sha(config_path), 'source_sha256':{str(p):sha(p) for p in source_files},
                    'zip_sha256':sha(archive), 'snapshot_sha256':{name:sha(snap/name) for name in audit['manifest_sha256']},
                    'holdout_sha256':{name:sha(root/(name+'.csv')) for name in audit['holdout_manifest_sha256']},
                    'weights_sha256':sha(weights), 'augmentation_sha256':sha(cfg['augmentation']),
                    'selection_sha256':sha(cfg['baseline_selection']),
                    'baseline_config_sha256':sha(baseline_config), 'baseline_checkpoint_sha256':sha(baseline_checkpoint),
                    'snapshot_audit_sha256':sha(snap/'audit.json'),
                    'verification_sha256':sha(verification_path), 'torch':str(torch.__version__)}
        provenance=capture_provenance()
        provenance_path=output/'provenance.json'
        if args.resume:
            if json.loads(provenance_path.read_text(encoding='utf8'))!=provenance:
                raise ValueError('Cannot resume: provenance changed')
        else:
            provenance_path.write_text(json.dumps(provenance,indent=2),encoding='utf8')
        os.environ['TORCH_HOME']=str(Path(cfg['torch_home']).resolve())
        torch.set_num_threads(1); seed_everything(cfg['seed'])
        if args.mode=='pilot' and args.wait_for_v2:
            if not args.workflow_pid or not args.post_pid: raise ValueError('Expected v2 owner PIDs')
            while True:
                workflow=json.loads(Path('artifacts/model-only-release/workflow-status.json').read_text(encoding='utf8'))['stage']
                post=json.loads(Path('artifacts/model-only-release/post-training-status.json').read_text(encoding='utf8'))['stage']
                if workflow=='complete' and post=='complete': break
                if (workflow!='complete' and not _process_alive(args.workflow_pid)) or (post!='complete' and not _process_alive(args.post_pid)):
                    raise RuntimeError(f'Existing v2 workflow stopped: workflow={workflow}, post={post}')
                status('waiting_for_v2',workflow=workflow,post_training=post)
                time.sleep(30)
        if capture_provenance()!=provenance:
            raise ValueError('Code, data, weights, or protocol changed while waiting')
        device=torch.device('cpu' if args.mode=='smoke' else 'cuda')
        if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        train_rows=rows(snap/'train.csv'); val_rows=rows(root/'validation.csv')
        for row in val_rows:
            with Image.open(root/row['image_path']) as image:
                if image_hash(image)!=row['pixel_hash']:
                    raise ValueError('Changed validation image: '+row['image_id'])
        train_data=ZipFurnitureDataset(train_rows,archive,mapping,'furniture_dataset/data/processed/',cfg['image_size'],aug)
        train_loader=DataLoader(Subset(train_data,range(4)) if args.mode=='smoke' else train_data,
                                batch_size=4 if args.mode=='smoke' else cfg['batch_size'],
                                shuffle=True,num_workers=cfg['num_workers'],drop_last=False)
        val_loader=DataLoader(FurnitureDataset(val_rows,root,mapping,cfg['image_size'],False),
                              batch_size=cfg['batch_size'],shuffle=False,num_workers=0)
        model=AttributeModel(mapping,pretrained=True,backbone=cfg['backbone'],weights_name='IMAGENET1K_V1').to(device)
        optimizer=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':cfg['finetune_lr']},
                                    {'params':model.heads.parameters(),'lr':cfg['head_lr']}],weight_decay=cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2)
        source_sha=sha(Path(__file__)); history=[]; start_epoch=0; best=-1.0
        if args.resume and (output/'last.pt').exists():
            state=torch.load(output/'last.pt',map_location=device,weights_only=True)
            if state['config_sha256']!=cfgsha or state['source_sha256']!=source_sha or state['train_sha256']!=audit['manifest_sha256']['train.csv']:
                raise ValueError('Cannot resume: code, config or train manifest changed')
            model.load_state_dict(state['state_dict']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
            history=state['history']; start_epoch=state['epoch']; best=state['best']
        for epoch in range(start_epoch,1 if args.mode=='smoke' else cfg['epochs']):
            if capture_provenance()!=provenance:
                raise ValueError('Code, data, weights, or protocol changed before epoch')
            status('running',epoch=epoch+1,mode=args.mode,device=str(device))
            seed_everything(cfg['seed']+epoch)
            for transform in train_data.transform.transforms:
                if hasattr(transform,'pipeline') and hasattr(transform.pipeline,'set_random_seed'):
                    transform.pipeline.set_random_seed(cfg['seed']+epoch)
            frozen=epoch<cfg['freeze_epochs']; model.freeze_backbone(frozen)
            timed=TimedLoader(train_loader); started=time.perf_counter()
            train=run_epoch(model,timed,mapping,device,optimizer,frozen,cfg['loss_weights'])
            elapsed=time.perf_counter()-started
            if args.mode=='smoke':
                status('smoke_complete',images=4,backbone=cfg['backbone'],device=str(device),training_seconds=elapsed,
                       note='One optimizer step to validate mechanics; not a research result')
                return
            val=run_epoch(model,val_loader,mapping,device,loss_weights=cfg['loss_weights'])
            if capture_provenance()!=provenance:
                raise ValueError('Code, data, weights, or protocol changed during epoch')
            score=(val['type']['macro_f1']+val['style']['macro_f1'])/2; scheduler.step(score)
            history.append({'epoch':epoch+1,'frozen':frozen,'train_images':len(train_rows),'optimizer_steps':len(train_loader),
                            'train_wall_seconds':elapsed,'data_wait_seconds':timed.wait,
                            'train_images_per_second':len(train_rows)/elapsed,'train':train,'validation':val})
            payload={'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                     'mapping':mapping,'config':cfg,'config_sha256':cfgsha,'source_sha256':source_sha,
                     'train_sha256':audit['manifest_sha256']['train.csv'],'archive_sha256':audit['archive_sha256'],
                     'validation_sha256':audit['holdout_manifest_sha256']['validation'],'epoch':epoch+1,'history':history,
                     'best':max(best,score),'preprocessing':'same functions as baseline; ZIP inputs were pre-resized/recompressed'}
            if score>best:
                best=score; atomic_torch_save(payload,output/'best.pt')
            atomic_torch_save(payload,output/'last.pt')
            (output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8')
        status('complete',epochs=len(history),train_images=len(train_rows),validation_images=len(val_rows),
               validation_coverage={'type':1.0,'style':1.0},best_mean_macro_f1=best,
               best_checkpoint_sha256=sha(output/'best.pt'),note='Pilot; no test predictions; 3 epochs vs baseline up to 30')


if __name__=='__main__': main()
