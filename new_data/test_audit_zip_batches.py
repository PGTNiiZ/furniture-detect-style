import csv
import hashlib
import io
import json
import sys
import zipfile

import pytest
from PIL import Image
from furniture.data import dhash
from new_data.audit_zip_batches import main


def setup_case(tmp_path, corrupt=False, wrong_label=False):
    prefix = 'furniture_dataset/data/processed/'
    archive = tmp_path/'images.zip'
    image = Image.new('RGB', (12,12), (100,30,20))
    data = io.BytesIO(); image.save(data,format='PNG')
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr(prefix+'labels.json', json.dumps({'type2id':{'bed':0},'style2id':{'modern':0}}))
        z.writestr(prefix+'train.csv', 'path,type,style,type_id,style_id,src_index\n'
                   + f"images/train/{'chair' if wrong_label else 'bed'}/modern/a.png,bed,modern,0,0,7\n")
        z.writestr(prefix+f"images/train/{'chair' if wrong_label else 'bed'}/modern/a.png", b'broken' if corrupt else data.getvalue())
    v2 = tmp_path/'v2'; v2.mkdir()
    for split in ('validation','synthetic_test','real_source_test'):
        with (v2/(split+'.csv')).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['image_id','dhash']); writer.writeheader()
            writer.writerow({'image_id':split+'-id','dhash':dhash(image)})
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'archive_path':str(archive),'archive_prefix':prefix,
      'expected_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(), 'v2_root':str(v2),
      'batch_size':2,'num_workers':0,'image_size':16,'seed':42}),encoding='utf-8')
    return config


def test_reports_candidates_and_checks_batch_labels(tmp_path, monkeypatch):
    config=setup_case(tmp_path)
    out=tmp_path/'output'
    monkeypatch.setattr(sys,'argv',['audit','--config',str(config),'--output',str(out)])
    main()
    result=json.loads((out/'audit.json').read_text(encoding='utf-8'))
    assert result['errors']==0 and result['batch_trial']['images']==1
    assert result['candidate_pairs_by_split']=={'validation':1,'synthetic_test':1,'real_source_test':1}
    assert not result['training_approved']


@pytest.mark.parametrize('corrupt,wrong_label', [(True,False),(False,True)])
def test_errors_reported_and_no_batch_or_training(tmp_path, monkeypatch, corrupt, wrong_label):
    config=setup_case(tmp_path,corrupt,wrong_label)
    out=tmp_path/'output'
    monkeypatch.setattr(sys,'argv',['audit','--config',str(config),'--output',str(out)])
    with pytest.raises(SystemExit,match='Audit errors'):
        main()
    result=json.loads((out/'audit.json').read_text(encoding='utf-8'))
    assert result['errors']==1 and 'batch_trial' not in result
    assert len(list(csv.DictReader((out/'errors.csv').open(encoding='utf-8'))))==1
