import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from furniture.room_evaluation import evaluate_room_domain, export_room_detections


def write_csv(path, fields, rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_room_domain_metrics_include_misses_and_false_positives(tmp_path, monkeypatch):
    import furniture.room_evaluation as module
    Image.new('RGB',(100,100),'white').save(tmp_path/'room.png')
    annotation_fields=['room_id','image_id','image_path','object_id','split','type','style','x1','y1','x2','y2']
    annotations=[dict(room_id='room1',image_id='image1',image_path='room.png',object_id='o1',split='test',
                      type='chair',style='modern',x1=0,y1=0,x2=30,y2=30),
                 dict(room_id='room1',image_id='image1',image_path='room.png',object_id='o2',split='test',
                      type='sofa',style='rustic',x1=40,y1=40,x2=80,y2=80)]
    detection_fields=['image_id','detector_type','confidence','x1','y1','x2','y2']
    detections=[dict(image_id='image1',detector_type='chair',confidence=.9,x1=0,y1=0,x2=30,y2=30),
                dict(image_id='image1',detector_type='chair',confidence=.8,x1=65,y1=0,x2=90,y2=25),
                dict(image_id='image1',detector_type='lamp',confidence=.9,x1=0,y1=40,x2=20,y2=60)]
    annotation_path,detector_path=tmp_path/'annotation.csv',tmp_path/'detector.csv'
    write_csv(annotation_path,annotation_fields,annotations)
    write_csv(detector_path,detection_fields,detections)
    class_path=tmp_path/'classes.json'
    class_path.write_text(json.dumps({'chair':'chair','couch':'sofa'}))
    checkpoint=tmp_path/'model.pt'
    checkpoint.write_bytes(b'test checkpoint')
    payload={'mapping':{'type':{'chair':0,'sofa':1},'style':{'modern':0,'rustic':1}}}
    monkeypatch.setattr(module,'load_model',lambda *a:(object(),payload))
    monkeypatch.setattr(module,'classify_crop',lambda *a:{'type':{'chair':.9,'sofa':.1},'style':{'modern':.8,'rustic':.2}})
    out=tmp_path/'report'
    report=evaluate_room_domain(annotation_path,detector_path,tmp_path,checkpoint,out,class_path,device='cpu')
    assert report['objects']==2
    assert report['detection']['matched']==1
    assert report['detection']['misses']==1
    assert report['detection']['false_positives']==1
    assert report['detection']['excluded']['unmapped_class']==1
    assert report['reference_crops']['type']['accuracy']==pytest.approx(.5)
    assert report['matched_detector_crops']['type']['accuracy']==pytest.approx(1)
    assert (out/'room_predictions.csv').is_file()
    assert json.loads((out/'room_metrics.json').read_text(encoding='utf-8'))['misses'][0]['object_id']=='o2'
    with pytest.raises(ValueError,match='already exists'):
        evaluate_room_domain(annotation_path,detector_path,tmp_path,checkpoint,out,class_path,device='cpu')


def test_iou_and_box_validation():
    from furniture.room_evaluation import _box, _iou
    assert _iou((0,0,10,10),(5,5,15,15)) == pytest.approx(25/175)
    with pytest.raises(ValueError,match='Invalid bounding box'):
        _box({'x1':0,'y1':0,'x2':11,'y2':5},10,10)


def test_detector_export_uses_annotated_images_and_records_provenance(tmp_path,monkeypatch):
    Image.new('RGB',(30,30),'white').save(tmp_path/'room.png')
    ann=tmp_path/'annotation.csv'
    write_csv(ann,['room_id','image_id','image_path','object_id','split'],
              [dict(room_id='r1',image_id='i1',image_path='room.png',object_id='o1',split='test'),
               dict(room_id='r1',image_id='i1',image_path='room.png',object_id='o2',split='test')])
    box=SimpleNamespace(cls=torch.tensor(0),conf=torch.tensor(.9),xyxy=torch.tensor([[1.,2.,20.,22.]]))
    calls=[]
    class FakeYOLO:
        def __init__(self,weights): pass
        def predict(self,**kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=[box],names={0:'chair'})]
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=FakeYOLO))
    output=tmp_path/'pred.csv'
    result=export_room_detections(ann,tmp_path,'mock.pt',output,device='cpu')
    assert result['images']==1 and result['detections']==1 and len(calls)==1
    assert json.loads((tmp_path/'pred.csv.metadata.json').read_text())['detections_sha256']
    with output.open(newline='') as stream:
        assert list(csv.DictReader(stream))[0]['detector_type']=='chair'
    with pytest.raises(ValueError,match='confidence is incompatible'):
        evaluate_room_domain(ann,output,tmp_path,tmp_path/'not-loaded.pt',tmp_path/'report',
                             tmp_path/'not-loaded.json',device='cpu',confidence=.01)
