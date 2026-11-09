import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import torch

from furniture.data import read_rows, validate_splits, write_rows
from furniture.model import AttributeModel
from furniture.smoke import make_smoke


def test_pretrained_step_requires_audited_data_and_updates_backbone(tmp_path, monkeypatch):
    script=Path(__file__).resolve().parents[1]/'scripts'/'verify_pretrained_steps.py'
    spec=importlib.util.spec_from_file_location('verify_pretrained_steps',script)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    torch.set_num_threads(1)
    root=tmp_path/'data'; make_smoke(root)
    for split in ('train','val','test'):
        rows=read_rows(root/f'{split}.csv')
        for row in rows: row['source_split']='train'
        write_rows(root/f'{split}.csv',rows)
    splits,mapping=validate_splits(root)
    shutil.copyfile(root/'val.csv',root/'validation.csv')
    shutil.copyfile(root/'test.csv',root/'synthetic_test.csv')
    (root/'real_source_test.csv').write_text('image_id\n')
    for name in ('candidate_evidence.csv','candidate_audit.json','quarantine.csv','near_duplicate_review.csv','group_audit.csv'):
        (root/name).write_text('fixture')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    metadata={'dataset_protocol_version':'furniture-dataset-2.0','class_mapping':mapping,
              'training_audit':{'status':'incomplete','closed_set':True,'group_overlap_count':0,
                                'exact_overlap_count':0,'remaining_cross_split_candidate_count':0},
              'manifest_sha256':{s:sha(root/f'{s}.csv') for s in ('train','validation','synthetic_test','real_source_test')},
              'split_hashes':{s:sha(root/f'{s}.csv') for s in ('train','val','test')},
              **{key:sha(root/name) for name,key in [('candidate_evidence.csv','candidate_evidence_sha256'),
                      ('candidate_audit.json','candidate_audit_sha256'),('quarantine.csv','quarantine_sha256'),
                      ('near_duplicate_review.csv','near_duplicate_review_sha256'),('group_audit.csv','group_audit_sha256')]}}
    (root/'dataset_metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='audit has not passed'):
        module.validate_training_snapshot(root)
    metadata['training_audit']['status']='pass_conservative_quarantine'
    (root/'dataset_metadata.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(module,'BACKBONES',{'resnet18':None})
    monkeypatch.setattr(module,'AttributeModel',lambda m,**kw:AttributeModel(m,pretrained=False,backbone=kw['backbone']))
    result=module.run(root,'configs/project_controlled.yaml',tmp_path/'result.json',device='cpu',batch_size=2)
    assert result['models']['resnet18']['status']=='passed'
    assert len(result['sample_ids'])==2 and not list(tmp_path.rglob('*.pt'))
    (root/'val.csv').write_text('tampered')
    with pytest.raises(ValueError,match='alias differs'):
        module.validate_training_snapshot(root)
