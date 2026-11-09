import csv
import json

import pytest
from furniture.robustness import task_metrics
from scripts.prediction_audit import audit_closed_set, audit_confusion_csv, audit_external, audit_external_exclusions


def write(path, rows):
    with path.open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def test_closed_set_recomputes_metrics_and_rejects_missing_ids_labels_and_matrix(tmp_path):
    mapping={'type':{'chair':0,'sofa':1},'style':{'modern':0,'rustic':1}}
    manifest=[{'image_id':'a','image_path':'a.png','type':'chair','style':'modern'},
              {'image_id':'b','image_path':'b.png','type':'sofa','style':'rustic'}]
    predictions=[{'image_id':'a','image_path':'a.png','true_type':'chair','pred_type':'chair','true_style':'modern','pred_style':'rustic'},
                 {'image_id':'b','image_path':'b.png','true_type':'sofa','pred_type':'chair','true_style':'rustic','pred_style':'rustic'}]
    m,p=tmp_path/'manifest.csv',tmp_path/'pred.csv'
    write(m,manifest); write(p,predictions)
    metrics={'type':task_metrics([0,1],[0,0],['chair','sofa']), 'style':task_metrics([0,1],[1,1],['modern','rustic'])}
    assert audit_closed_set(m,p,metrics,mapping)['samples']==2
    matrix = tmp_path/'matrix.csv'
    matrix.write_text('true / predicted,chair,sofa\nchair,1,0\nsofa,1,0\n')
    audit_confusion_csv(matrix,mapping['type'],metrics['type'])
    matrix.write_text('true / predicted,chair,sofa\nchair,0,0\nsofa,1,0\n')
    with pytest.raises(ValueError,match='CSV counts'): audit_confusion_csv(matrix,mapping['type'],metrics['type'])
    predictions[0]['true_type']='sofa'; write(p,predictions)
    with pytest.raises(ValueError,match='Reference label'): audit_closed_set(m,p,metrics,mapping)
    predictions[0]['true_type']='chair'; write(p,predictions[:1])
    with pytest.raises(ValueError,match='IDs'): audit_closed_set(m,p,metrics,mapping)
    write(p,predictions); metrics['type']['confusion_matrix'][0][0]=0
    with pytest.raises(ValueError,match='Confusion'): audit_closed_set(m,p,metrics,mapping)


def test_external_recomputes_coverage_and_joint(tmp_path):
    mapping={'type':{'chair':0},'style':{'modern':0}}
    manifest=[{'image_id':'a','type':'chair','style':'modern'}, {'image_id':'b','type':'chair','style':''},
              {'image_id':'c','type':'new','style':'modern'}]
    predictions=[{'image_id':'a','true_type':'chair','type_state':'known','predicted_type':'chair','true_style':'modern','style_state':'known','predicted_style':'modern'},
                 {'image_id':'b','true_type':'chair','type_state':'known','predicted_type':'chair','true_style':'','style_state':'missing','predicted_style':'modern'},
                 {'image_id':'c','true_type':'new','type_state':'unseen','predicted_type':'chair','true_style':'modern','style_state':'known','predicted_style':'modern'}]
    m,p=tmp_path/'external.csv',tmp_path/'pred.csv'; write(m,manifest); write(p,predictions)
    metric=task_metrics([0,0],[0,0],['chair'])
    style=task_metrics([0,0],[0,0],['modern'])
    report={'tasks':{'type':{'eligible':2,'total':3,'coverage':2/3,'metrics':metric},
                     'style':{'eligible':2,'total':3,'coverage':2/3,'metrics':style}},
            'annotation_inspection':{'counts':{'type':{'missing':0,'known':2,'unseen':1},'style':{'missing':1,'known':2,'unseen':0}}},
            'joint':{'eligible':1,'total':3,'coverage':1/3,'correct':1,'accuracy':1.0}}
    assert audit_external(m,p,report,mapping)['known_both']==1
    exclusions = tmp_path/'exclusions.csv'; write(exclusions,predictions[1:])
    audit_external_exclusions(p,exclusions)
    write(exclusions,predictions[1:2])
    with pytest.raises(ValueError,match='missing/unseen'): audit_external_exclusions(p,exclusions)
    report['joint']['eligible']=2
    with pytest.raises(ValueError,match='joint'): audit_external(m,p,report,mapping)
