import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import pytest

from furniture.config import load_config
from furniture.data import validate_splits
from furniture.model import AttributeModel
from furniture.smoke import make_smoke


def test_training_benchmark_uses_only_train_and_no_checkpoint(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_training.py'
    spec = importlib.util.spec_from_file_location('benchmark_training', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    root = tmp_path/'data'
    make_smoke(root)
    splits, mapping = validate_splits(root)
    train_hash = hashlib.sha256((root/'train.csv').read_bytes()).hexdigest()
    (root/'dataset_metadata.json').write_text(json.dumps({'split_hashes':{'train':train_hash},'class_mapping':mapping}))
    cfg = load_config('configs/project_controlled.yaml', 'C3')
    cfg['project_root'] = str(tmp_path)
    cfg['dataset']['root'] = 'data'
    cfg['model']['image_size'] = 64
    monkeypatch.setattr(module, 'load_config', lambda *args: cfg)
    monkeypatch.setattr(module, 'AttributeModel', lambda *args, **kwargs: AttributeModel(args[0], pretrained=False, backbone=kwargs['backbone']))
    result = module.benchmark('ignored','resnet18',tmp_path/'benchmark.json',batch_size=2,batches=1,device='cpu')
    assert result['train_rows'] == len(splits['train'])
    assert result['train_manifest_sha256'] == train_hash
    assert set(result['phases']) == {'frozen','unfrozen'}
    assert not list(tmp_path.rglob('*.pt'))
    def model_with_different_rng_use(*args, **kwargs):
        torch.rand(1000)
        return AttributeModel(args[0], pretrained=False, backbone=kwargs['backbone'])
    monkeypatch.setattr(module, 'AttributeModel', model_with_different_rng_use)
    repeated = module.benchmark('ignored','resnet18',tmp_path/'repeat.json',batch_size=2,batches=1,device='cpu')
    assert repeated['sample_ids'] == result['sample_ids']
    assert repeated['sampling_seed'] == result['sampling_seed']
    (root/'dataset_metadata.json').unlink()
    with pytest.raises(ValueError, match='metadata is missing'):
        module.benchmark('ignored','resnet18',tmp_path/'blocked.json',batch_size=2,batches=1,device='cpu')
    result = module.benchmark('ignored','resnet18',tmp_path/'unvalidated.json',batch_size=2,batches=1,
                              device='cpu',allow_unvalidated_train=True)
    assert result['dataset_audit_status'].startswith('unvalidated')
