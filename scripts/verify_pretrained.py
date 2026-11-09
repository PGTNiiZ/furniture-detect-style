"""Download and forward-check the exact configured torchvision weights, without training."""
import argparse
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import torch
from torchvision import models

from furniture.model import AttributeModel, BACKBONES, WEIGHT_NAMES
from furniture.train import choose_device


def verify_one(name, device):
    weights = models.get_model_weights(name)[WEIGHT_NAMES[name]]
    model = AttributeModel({'type': {'a': 0, 'b': 1}, 'style': {'x': 0, 'y': 1}},
                           pretrained=True, backbone=name, weights_name=weights.name).to(device).eval()
    with torch.inference_mode():
        result = model(torch.zeros(1, 3, 224, 224, device=device))
    if any(tuple(result[k].shape) != (1, 2) or not torch.isfinite(result[k]).all().item() for k in ('type', 'style')):
        raise ValueError('Model returned invalid output')
    path = Path(torch.hub.get_dir()) / 'checkpoints' / Path(urlparse(weights.url).path).name
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    checksum = digest.hexdigest()
    evidence = {'weights':str(weights), 'weights_url':weights.url, 'sha256':checksum,
                'feature_dim':model.feature_dim, 'input_shape':[1,3,224,224], 'device':str(device)}
    del result, model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return evidence


def verify(output, device='auto'):
    output = Path(output)
    if output.exists():
        raise ValueError('Output exists; choose a new report path')
    selected = choose_device(device)
    report = {'started_at':datetime.now(timezone.utc).isoformat(), 'torch':str(torch.__version__),
              'torchvision':__import__('torchvision').__version__, 'device':str(selected), 'models':{},
              'note':'Pretrained initialization and forward check only; no training or accuracy evaluation.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    for name in BACKBONES:
        try:
            report['models'][name] = {'status':'passed', **verify_one(name, selected)}
        except Exception as error:
            report['models'][name] = {'status':'failed', 'error':f'{type(error).__name__}: {error}'}
            gc.collect()
            if selected.type == 'cuda':
                torch.cuda.empty_cache()
        temporary = output.with_suffix(output.suffix + '.tmp')
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        for attempt in range(5):
            try:
                os.replace(temporary, output)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(.2 * (attempt + 1))
        print(f"{name}: {report['models'][name]['status']}", flush=True)
    report['ended_at'] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    report = verify(args.output, args.device)
    raise SystemExit(0 if all(r['status'] == 'passed' for r in report['models'].values()) else 1)
