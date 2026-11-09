import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.detect_then_classify import detect_and_classify, export_room


def box(label, coords, confidence=.8):
    return SimpleNamespace(cls=torch.tensor(label), conf=torch.tensor(confidence), xyxy=torch.tensor([coords]))


class Detector:
    def __init__(self, boxes):
        self.boxes = boxes
        self.seen = None
    def predict(self, **kwargs):
        self.seen = kwargs['source']
        return [SimpleNamespace(boxes=self.boxes, names={0:'chair', 1:'couch', 2:'person'})]


class Predictor:
    device = 'cpu'
    mapping = {'type':{'chair':0,'sofa':1}, 'style':{'modern':0,'vintage':1}}
    def __init__(self): self.calls = 0
    def predict_image(self, image, top_k):
        self.calls += 1
        if self.calls == 2:
            raise ValueError('bad crop')
        return {'type':{'label':'sofa','confidence':.8,'top_k':[{'label':'sofa','confidence':.8}]},
                'style':{'label':'modern','confidence':.6,'top_k':[{'label':'modern','confidence':.6}]}}


def test_room_multiple_objects_clip_padding_disagreement_and_errors(tmp_path):
    image = tmp_path/'room.png'
    Image.new('RGBA', (20, 10), (255,0,0,0)).save(image)
    detector = Detector([box(0,[-4,1,10,9]), box(1,[11,1,19,9]), box(0,[3,2,3,4]), box(2,[0,0,4,4])])
    result = detect_and_classify(image, detector, Predictor(), image_id='room1', class_mapping={'chair':'chair','couch':'sofa'}, padding_fraction=.25)
    assert result['image_size'] == {'width':20,'height':10} and len(result['objects']) == 4
    first, second, invalid, unmapped = result['objects']
    assert first['bbox_xyxy'] == [0,1,10,9] and first['crop_xyxy'] == [0,0,13,10]
    assert first['category'] == 'sofa' and first['detector_label'] == 'chair' and first['category_disagreement']
    assert first['image_id'] == 'room1' and first['object_id'] == '0'
    assert second['status'] == 'crop_error' and invalid['status'] == 'crop_error'
    assert unmapped['status'] == 'unmapped_detector_label'
    assert np.all(detector.seen == 255)  # transparent input is composited on white
    out = tmp_path/'out'
    out.mkdir()
    export_room(result, image, out)
    assert (out/'annotated.png').is_file() and json.loads((out/'result.json').read_text(encoding='utf8'))['image_id'] == 'room1'
    with (out/'objects.csv').open(encoding='utf8',newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 4


def test_room_no_detections_and_broken_image(tmp_path):
    image = tmp_path/'empty.png'
    Image.new('RGB',(8,8),'white').save(image)
    result = detect_and_classify(image, Detector([]), Predictor())
    assert result['status'] == 'no_detections' and result['objects'] == []
    bad = tmp_path/'bad.jpg'
    bad.write_bytes(b'broken')
    result = detect_and_classify(bad, Detector([]), Predictor())
    assert result['status'] == 'image_error' and result['objects'] == []


def test_room_exif_orientation_before_detection(tmp_path):
    image = tmp_path/'oriented.jpg'
    exif = Image.Exif()
    exif[274] = 6
    Image.new('RGB',(20,10),'white').save(image, exif=exif)
    detector = Detector([box(0,[0,0,10,20])])
    result = detect_and_classify(image, detector, Predictor(), class_mapping={'chair':'chair'}, padding_fraction=0)
    assert result['image_size'] == {'width':10,'height':20}
    assert detector.seen.shape[:2] == (20,10)
    assert result['objects'][0]['bbox_xyxy'] == [0,0,10,20]


def test_room_invalid_padding_and_nonfinite_box(tmp_path):
    image = tmp_path/'room.png'
    Image.new('RGB',(20,20)).save(image)
    with pytest.raises(ValueError,match='padding'):
        detect_and_classify(image, Detector([]), Predictor(), padding_fraction=-1)
    result = detect_and_classify(image, Detector([box(0,[0,0,float('nan'),2])]), Predictor())
    assert result['objects'][0]['status'] == 'crop_error'
    json.dumps(result, allow_nan=False)
