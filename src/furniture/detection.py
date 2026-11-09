"""Delegate box-aware augmentation to Ultralytics, keep a resolved config for the run."""
from datetime import datetime,timezone
from pathlib import Path
import uuid
import yaml
from .config import read_yaml,exact_keys,number


def train_detector(config_path):
    from ultralytics import YOLO
    path = Path(config_path).resolve()
    cfg = read_yaml(path)
    exact_keys(cfg,['project_root','weights','data','output_root','train','augmentation'],'detector')
    exact_keys(cfg['train'],['epochs','batch','imgsz','device','seed','workers','optimizer','lr0'],'detector.train')
    exact_keys(cfg['augmentation'],['fliplr','flipud','degrees','translate','scale','hsv_h','hsv_s','hsv_v','mosaic','mixup'],'detector.augmentation')
    for key in ['epochs','batch','imgsz']:
        number(cfg['train'][key],key,1,integer=True)
    for key in ['seed','workers']:
        number(cfg['train'][key],key,0,integer=True)
    for key,value in cfg['augmentation'].items():
        number(value,key,0,180 if key=='degrees' else 1)
    root = (path.parent/cfg['project_root']).resolve()
    source = read_yaml(root/cfg['data'])
    for key in ['path','train','val','names']:
        if key not in source:
            raise ValueError(f'Detection data YAML needs {key}')
    source['path'] = str((root/source['path']).resolve())
    for split in ['train','val']:
        if not (Path(source['path'])/source[split]).exists():
            raise ValueError(f'Missing room data: {split}; annotate images and boxes first')
    output = root/cfg['output_root']
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    run = output/name
    run.mkdir(parents=True,exist_ok=False)
    data_yaml = run/'resolved_data.yaml'
    data_yaml.write_text(yaml.safe_dump(source,allow_unicode=True),encoding='utf-8')
    (run/'detector_config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True),encoding='utf-8')
    weights = cfg['weights'] if len(Path(cfg['weights']).parts)==1 else str(root/cfg['weights'])
    model = YOLO(weights)
    from .train import choose_device
    train_settings = dict(cfg['train'])
    train_settings['device'] = str(choose_device(train_settings['device']))
    model.train(data=str(data_yaml),project=str(output),name=name,exist_ok=True,**train_settings,**cfg['augmentation'])
    return {'run_dir':str(run),'task':'detection','weights':str(run/'weights/best.pt')}
