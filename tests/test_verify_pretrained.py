import importlib.util
import json
from pathlib import Path

import pytest
import torch


def test_pretrained_verification_records_failures_without_claiming_success(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'verify_pretrained.py'
    spec = importlib.util.spec_from_file_location('verify_pretrained', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'BACKBONES', ['ok', 'bad'])
    def fake(name, device):
        if name == 'bad':
            raise RuntimeError('unavailable weights')
        return {'weights':'test', 'device':str(device)}
    monkeypatch.setattr(module, 'verify_one', fake)
    output = tmp_path / 'verify.json'
    result = module.verify(output, 'cpu')
    assert result['models']['ok']['status'] == 'passed'
    assert result['models']['bad']['status'] == 'failed'
    assert json.loads(output.read_text())['models']['bad']['error'].startswith('RuntimeError')
    with pytest.raises(ValueError, match='exists'):
        module.verify(output, 'cpu')
