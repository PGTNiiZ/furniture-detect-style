import copy
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from furniture import augmentation, data, train
from furniture.config import load_config, training_settings
from furniture.smoke import make_smoke

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


@pytest.mark.parametrize('name', ['none_off', 'easy_off', 'hard_off', 'easy_light', 'easy_strong'])
def test_fit_passes_configuration_to_real_training_transform(tmp_path, monkeypatch, name):
    root = tmp_path/'data'
    make_smoke(root)
    cfg = training_settings(load_config(CONFIGS/'project_smoke.yaml'))
    expected = yaml.safe_load((CONFIGS/'augmentation'/f'{name}.yaml').read_text())
    cfg['augment'] = expected
    cfg['device'] = 'cpu'
    datasets = []
    actual_dataset = data.FurnitureDataset

    def capture_dataset(*args, **kwargs):
        dataset = actual_dataset(*args, **kwargs)
        datasets.append((args[4], dataset))
        return dataset

    class StopAfterBatch(Exception):
        pass

    batches = []
    def stop_after_batch(model, loader, *args, **kwargs):
        batches.append(next(iter(loader))[0])
        raise StopAfterBatch

    monkeypatch.setattr(train, 'FurnitureDataset', capture_dataset)
    monkeypatch.setattr(train, 'run_epoch', stop_after_batch)
    monkeypatch.setattr(train.subprocess, 'check_output', lambda *args, **kwargs: b'')
    with pytest.raises(StopAfterBatch):
        train.fit(root, tmp_path/'run', cfg)
    assert datasets[0][0] == expected
    assert datasets[1][0] is False
    assert batches[0].shape[0] == cfg['batch_size']
    pipeline = datasets[0][1].transform.transforms[1].pipeline
    p, o = expected['params'], expected['occlusion']
    assert [t.p for t in pipeline.transforms] == [p['horizontal_flip_p'], p['affine_p'],
                                                   p['brightness_contrast_p'], p['blur_p'], o['probability']]
    assert pipeline.transforms[1].rotate == (-p['rotate_degrees'], p['rotate_degrees'])
    assert pipeline.transforms[1].scale['x'] == tuple(p['scale'])
    assert pipeline.transforms[2].brightness_limit == (-p['brightness_limit'], p['brightness_limit'])
    assert pipeline.transforms[-1].fill == o['fill']
    assert pipeline.transforms[-1].num_holes_range == tuple(o['holes'])
    assert pipeline.transforms[-1].hole_height_range == tuple(o['height_fraction'])
    assert torch.equal(datasets[1][1][0][0], datasets[1][1][0][0])
    splits, mapping = data.validate_splits(root)
    test_dataset = actual_dataset(splits['test'], root, mapping, cfg['image_size'])
    assert torch.equal(test_dataset[0][0], test_dataset[0][0])
    assert len(test_dataset.transform.transforms) == 3


def test_preview_and_train_use_same_transform_builder(tmp_path, monkeypatch):
    cfg = yaml.safe_load((CONFIGS/'augmentation/easy_light.yaml').read_text())
    calls = []
    original = augmentation.pil_transforms
    def spy(value):
        calls.append(copy.deepcopy(value))
        return original(value)
    monkeypatch.setattr(augmentation, 'pil_transforms', spy)
    image = tmp_path/'input.png'
    Image.new('RGB', (32,32), 'black').save(image)
    data.transform(32, cfg)(Image.open(image))
    augmentation.preview(image, tmp_path/'preview.png', 32, cfg, count=1)
    assert calls == [cfg, cfg]


def test_preset_parameter_validation():
    cfg = yaml.safe_load((CONFIGS/'augmentation/easy_light.yaml').read_text())
    cfg['params']['affine_p'] = 1.1
    with pytest.raises(ValueError, match='out of range'):
        augmentation.settings(cfg)
    cfg['params']['affine_p'] = .3
    cfg['params']['unknown'] = True
    with pytest.raises(ValueError, match='unknown keys'):
        augmentation.settings(cfg)


def test_none_off_is_exactly_deterministic_preprocessing():
    config = yaml.safe_load((CONFIGS/'augmentation/none_off.yaml').read_text())
    image = Image.new('RGB', (32, 32), (17, 39, 82))
    assert torch.equal(data.transform(32, config)(image), data.transform(32, False)(image))


def test_preview_dataset_selects_only_training_samples_and_is_reproducible(tmp_path):
    import json
    root = tmp_path/'data'
    make_smoke(root)
    first = tmp_path/'preview1'
    second = tmp_path/'preview2'
    for target in (first, second):
        result = augmentation.preview_dataset(root, target, 32, 42, 3, CONFIGS/'augmentation')
        assert result['samples'] == 3
    a = json.loads((first/'preview.json').read_text())
    b = json.loads((second/'preview.json').read_text())
    assert a == b
    assert len({(r['type'], r['style']) for r in a['samples']}) == 2
    for sample in a['samples']:
        assert sample['sample_id'].startswith('train-')
        assert (first/sample['file']).read_bytes() == (second/sample['file']).read_bytes()
