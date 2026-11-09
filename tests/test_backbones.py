import gc
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from torchvision.models import get_model_weights

from furniture.data import transform
from furniture.model import AttributeModel, BACKBONES, WEIGHT_NAMES
from furniture.train import load_model


@pytest.mark.parametrize('name', list(BACKBONES))
def test_backbone_output_shapes_and_checkpoint_roundtrip(name, tmp_path):
    torch.set_num_threads(1)
    mapping = {'type': {'chair': 0, 'table': 1}, 'style': {'modern': 0}}
    model = AttributeModel(mapping, pretrained=False, backbone=name)
    model.eval()
    image = torch.zeros(1, 3, 224, 224)
    with torch.inference_mode():
        outputs = model(image)
    assert outputs['type'].shape == (1, 2)
    assert outputs['style'].shape == (1, 1)
    assert model.feature_dim > 0
    checkpoint = tmp_path/'checkpoint.pt'
    torch.save({'state_dict': model.state_dict(), 'mapping':mapping, 'config': {'backbone':name}}, checkpoint)
    del model
    gc.collect()
    loaded, payload = load_model(checkpoint, torch.device('cpu'))
    checkpoint.unlink()
    assert payload['mapping'] == mapping
    with torch.inference_mode():
        restored = loaded(image)
    for key in outputs:
        torch.testing.assert_close(restored[key], outputs[key], rtol=0, atol=0)
    del loaded, payload, outputs, restored
    gc.collect()


@pytest.mark.parametrize('name', list(BACKBONES))
def test_explicit_weights_enum_and_controlled_preprocessing(name):
    config = yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/models'/f'{name}.yaml').read_text())
    assert config['backbone'] == name and config['weights'] == WEIGHT_NAMES[name]
    weights = get_model_weights(name)[config['weights']]
    official = weights.transforms()
    assert official.mean == [.485, .456, .406]
    assert official.std == [.229, .224, .225]
    assert config['image_size'] == 224


def test_evaluation_preprocessing_rgb_padding_and_normalization():
    image = Image.new('RGB', (8, 4), (255, 0, 0))
    result = transform(8, False)(image)
    torch.testing.assert_close(result[:, 0, 0], torch.tensor([(1-.485)/.229, (1-.456)/.224, (1-.406)/.225]))
    torch.testing.assert_close(result[:, 4, 4], torch.tensor([(1-.485)/.229, -.456/.224, -.406/.225]))
