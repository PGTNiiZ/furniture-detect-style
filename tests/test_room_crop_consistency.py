import hashlib
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.detect_then_classify import detect_and_classify
from scripts.room_evaluation import evaluate


class Predictor:
    device='cpu'
    mapping={'type':{'chair':0},'style':{'modern':0}}
    exposure_membership={s:{'pixel_hashes':[]} for s in ('train','val','test')}
    def __init__(self): self.crops=[]
    def predict_image(self,image,top_k=3):
        self.crops.append((image.size,image.tobytes()))
        return {'type':{'label':'chair','confidence':1.,'top_k':[{'label':'chair','confidence':1.}]},
                'style':{'label':'modern','confidence':1.,'top_k':[{'label':'modern','confidence':1.}]}}


class Detector:
    def __init__(self,coords): self.coords=coords
    def predict(self,**kwargs):
        box=SimpleNamespace(cls=torch.tensor(0),conf=torch.tensor(.9),xyxy=torch.tensor([self.coords]))
        return [SimpleNamespace(boxes=[box],names={0:'chair'})]


@pytest.mark.parametrize('coords', [[2.25,3.5,16.25,17.75],[-2.5,1.25,19.5,19.75]])
def test_detector_crop_pixels_identical_in_inference_and_evaluation(tmp_path,coords):
    path=tmp_path/'room.png'
    pixels=np.arange(20*20*3,dtype=np.uint8).reshape(20,20,3)
    Image.fromarray(pixels).save(path)
    infer_predictor=Predictor()
    inference=detect_and_classify(path,Detector(coords),infer_predictor,class_mapping={'chair':'chair'},
                                  padding_fraction=.2,min_size=2)
    image={'image_id':'i','image_path':path.name,'room_id':'r','split':'final','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    obj={'image_id':'i','object_id':'o','type':'chair','style':'modern','x1':'0','y1':'0','x2':'20','y2':'20'}
    detection={'image_id':'i','detector_type':'chair','confidence':'.9',**dict(zip(('x1','y1','x2','y2'),map(str,coords)))}
    protocol={'version':'room-eval-v1','status':'locked','locked_at_utc':'2026-01-01T00:00:00Z',
              'development_reference':'independent-dev','independent_from_classifier_training':True,
              'matching':'class-agnostic greedy one-to-one by descending IoU','confidence':.25,'iou_threshold':.3,
              'crop_padding_fraction':.2,'min_crop_size':2,'detector_to_category':{'chair':'chair'}}
    eval_predictor=Predictor()
    result=evaluate([image],[obj],[detection],tmp_path,eval_predictor,protocol)
    assert result['detection']['matches']==1
    assert infer_predictor.crops[0] == eval_predictor.crops[1]
    assert inference['objects'][0]['bbox_xyxy'] == result['predictions'][0]['detector_bbox_xyxy']
    assert inference['objects'][0]['crop_xyxy'] == result['predictions'][0]['detector_crop_xyxy']


def test_too_small_detector_crop_rejected_in_both_paths(tmp_path):
    path=tmp_path/'room.png'; Image.new('RGB',(20,20),'white').save(path)
    coords=[5.25,5.25,5.75,5.75]
    inference=detect_and_classify(path,Detector(coords),Predictor(),class_mapping={'chair':'chair'},min_size=2)
    assert inference['objects'][0]['status']=='crop_error'
    image={'image_id':'i','image_path':path.name,'room_id':'r','split':'final','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    obj={'image_id':'i','object_id':'o','type':'chair','style':'modern','x1':'0','y1':'0','x2':'20','y2':'20'}
    det={'image_id':'i','detector_type':'chair','confidence':'.9',**dict(zip(('x1','y1','x2','y2'),map(str,coords)))}
    protocol={'version':'room-eval-v1','status':'locked','locked_at_utc':'2026-01-01T00:00:00Z',
              'development_reference':'independent-dev','independent_from_classifier_training':True,
              'matching':'class-agnostic greedy one-to-one by descending IoU','confidence':.25,'iou_threshold':.3,
              'crop_padding_fraction':.05,'min_crop_size':2,'detector_to_category':{'chair':'chair'}}
    result=evaluate([image],[obj],[det],tmp_path,Predictor(),protocol)
    assert result['detection']['invalid_or_small_crops']==1 and result['detection']['matches']==0
    assert result['rejected_detections'][0]['error'] == inference['objects'][0]['error'].removeprefix('ValueError: ')
