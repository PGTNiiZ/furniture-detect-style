import csv
import json

import pytest
import torch
from PIL import Image

from scripts import model_inference as inference
from scripts.detect_then_classify import detect_and_classify


class TinyModel:
    def __init__(self):
        self.inputs = []

    def __call__(self, tensor):
        self.inputs.append(tensor.clone())
        return {'type': torch.tensor([[0., 3.]]), 'style': torch.tensor([[3., 0., 1.]])}


@pytest.fixture
def predictor(monkeypatch):
    model = TinyModel()
    payload = {'mapping': {'type': {'sofa': 0, 'table': 1}, 'style': {'modern': 0, 'vintage': 1, 'zen': 2}},
               'config': {'image_size': 8}, 'preprocessing': {'image_size': 8, 'color': 'EXIF-corrected RGB',
               'resize': 'aspect-preserving PIL bicubic pad', 'pad_rgb': [255, 255, 255],
               'evaluation_order': 'resize/pad, tensor, normalize', 'mean': [0, 0, 0], 'std': [1, 1, 1]}}
    loads = []
    monkeypatch.setattr(inference, 'load_model', lambda checkpoint, device: (loads.append(checkpoint) or model, payload))
    result = inference.FurniturePredictor('model.pt', 'cpu')
    return result, model, payload, loads


def test_predict_many_transparent_and_corrupt_with_checkpoint_labels(tmp_path, predictor):
    pred, model, payload, loads = predictor
    alpha = tmp_path/'transparent.png'
    Image.new('RGBA', (8, 8), (255, 0, 0, 0)).save(alpha)
    broken = tmp_path/'bad.jpg'
    broken.write_bytes(b'not an image')
    rows = pred.predict([('a', alpha), ('bad', broken), ('b', alpha)], top_k=10)
    assert loads == ['model.pt'] and len(model.inputs) == 2
    assert torch.all(model.inputs[0] == 1)  # transparent pixels composite over white
    assert [row['status'] for row in rows] == ['ok', 'error', 'ok']
    assert rows[0]['type']['label'] == 'table' and rows[0]['style']['label'] == 'modern'
    assert len(rows[0]['type']['top_k']) == 2 and len(rows[0]['style']['top_k']) == 3
    with pytest.raises(ValueError, match='unique'):
        pred.predict([('a', alpha), ('a', alpha)])


def test_csv_input_and_json_csv_outputs(tmp_path, predictor):
    image = tmp_path/'image.png'
    Image.new('RGB', (8, 8), 'white').save(image)
    source = tmp_path/'images.csv'
    source.write_text('image_id,image_path\nitem-7,image.png\n', encoding='utf-8')
    records = predictor[0].predict(inference.input_items(csv_file=source))
    output = tmp_path/'predictions.json'
    inference.write_results(output, records)
    assert json.loads(output.read_text(encoding='utf-8'))[0]['sample_id'] == 'item-7'
    with pytest.raises(FileExistsError):
        inference.write_results(output, records)
    csv_output = tmp_path/'predictions.csv'
    inference.write_results(csv_output, records)
    with csv_output.open(encoding='utf-8', newline='') as stream:
        row = next(csv.DictReader(stream))
    assert row['sample_id'] == 'item-7' and row['style'] == 'modern'
    assert len(json.loads(row['style_top_k'])) == 3


def test_inference_rejects_missing_preprocessing(monkeypatch, predictor):
    payload = predictor[2]
    payload.pop('preprocessing')
    with pytest.raises(ValueError, match='preprocessing'):
        inference.FurniturePredictor('model.pt', 'cpu')


def test_optional_detector_passes_supported_crop_to_classifier(tmp_path, predictor):
    image = tmp_path/'room.png'
    Image.new('RGB', (16, 16), 'white').save(image)
    class Box:
        cls = torch.tensor(0)
        conf = torch.tensor(.8)
        xyxy = torch.tensor([[2, 3, 12, 14]])
    class Detection:
        names = {0:'dining table'}
        boxes = [Box()]
    class Detector:
        def predict(self, **kwargs):
            return [Detection()]
    result = detect_and_classify(image, Detector(), predictor[0])
    assert len(result['objects']) == 1
    assert result['objects'][0]['bbox_xyxy'] == [2, 3, 12, 14]
    assert result['objects'][0]['type']['label'] == 'table'
