import hashlib
from pathlib import Path

from PIL import Image
import pytest
import torch
from types import SimpleNamespace

from scripts.room_evaluation import evaluate
from scripts.export_room_detections import export


class Predictor:
    mapping = {'type':{'chair':0,'sofa':1},'style':{'modern':0}}
    exposure_membership = {split:{'pixel_hashes':[]} for split in ('train','val','test')}
    def predict_image(self,image):
        return {'type':{'label':'chair'}, 'style':{'label':'modern'}}


def sample(tmp_path):
    paths = [tmp_path/'room.png',tmp_path/'empty.png']
    for p in paths: Image.new('RGB',(100,100),'white').save(p)
    images = [{'image_id':f'i{i}','image_path':p.name,'room_id':f'r{i}','split':'final',
               'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for i,p in enumerate(paths)]
    objects = [dict(image_id='i0',object_id='one',type='chair',style='',x1='0',y1='0',x2='30',y2='30'),
               dict(image_id='i0',object_id='two',type='sofa',style='modern',x1='40',y1='40',x2='80',y2='80')]
    detections = [dict(image_id='i0',detector_type='couch',confidence='.9',x1='0',y1='0',x2='30',y2='30'),
                  dict(image_id='i1',detector_type='chair',confidence='.8',x1='0',y1='0',x2='25',y2='25')]
    protocol = {'version':'room-eval-v1','status':'locked','locked_at_utc':'2026-01-01T00:00:00Z',
                'development_reference':'independent-dev-v1','confidence':.25,'iou_threshold':.5,
                'crop_padding_fraction':0,'min_crop_size':2,'detector_to_category':{'chair':'chair','couch':'sofa'},
                'independent_from_classifier_training':True,
                'matching':'class-agnostic greedy one-to-one by descending IoU'}
    return images,objects,detections,protocol


def test_room_evaluation_includes_misses_false_positives_and_missing_style(tmp_path):
    images,objects,detections,protocol = sample(tmp_path)
    result=evaluate(images,objects,detections,tmp_path,Predictor(),protocol)
    assert result['objects']==2 and result['images']==2
    assert result['detection']['matches']==1 and result['detection']['misses']==1
    assert result['detection']['false_positives']==1 and result['detection']['matched_category_disagreements']==1
    assert result['reference_crops']['style']['missing_labels']==1
    assert result['reference_crops']['style']['eligible']==1
    assert result['matched_detector_crops']['style']['eligible']==0
    assert result['matched_detector_crops']['type']['coverage_of_objects']==.5


def test_room_evaluation_rejects_unlocked_protocol_and_room_leakage(tmp_path):
    images,objects,detections,protocol = sample(tmp_path)
    protocol['status']='draft'
    with pytest.raises(ValueError,match='locked'):
        evaluate(images,objects,detections,tmp_path,Predictor(),protocol)
    protocol['status']='locked'
    images[1]['room_id']='r0'; images[1]['split']='development'
    with pytest.raises(ValueError,match='crosses'):
        evaluate(images,objects,detections,tmp_path,Predictor(),protocol)


def test_room_evaluation_rejects_classifier_exposure_overlap(tmp_path):
    from furniture.data import image_hash
    images,objects,detections,protocol = sample(tmp_path)
    predictor=Predictor()
    with Image.open(tmp_path/'room.png') as image:
        digest=image_hash(image)
    predictor.exposure_membership={'train':{'pixel_hashes':[digest]},'val':{'pixel_hashes':[]},'test':{'pixel_hashes':[]}}
    with pytest.raises(ValueError,match='overlaps classifier'):
        evaluate(images,objects,detections,tmp_path,predictor,protocol)


def test_export_includes_images_without_objects(tmp_path):
    images,_,_,_ = sample(tmp_path)
    class Detector:
        calls = 0
        def predict(self,**kwargs):
            self.calls += 1
            boxes = [SimpleNamespace(cls=torch.tensor(0),conf=torch.tensor(.9),
                                     xyxy=torch.tensor([[0.,0.,25.,25.]]))] if self.calls == 1 else []
            return [SimpleNamespace(boxes=boxes,names={0:'chair'})]
    detector=Detector()
    result=export(images,tmp_path,detector,tmp_path/'detections.csv',split='final',confidence=.05,imgsz=640,device='cpu')
    assert result == {'images':2,'detections':1} and detector.calls == 2
