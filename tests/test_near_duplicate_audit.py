import csv
import importlib.util
from pathlib import Path


def test_exhaustive_audit_uses_only_source_train_and_reports_cross_split(tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts'/'audit_near_duplicates.py'
    spec=importlib.util.spec_from_file_location('audit_near_duplicates',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fields=['image_id','source_split','dhash','pixel_hash']
    rows={'train':[('a','train','0000000000000000','a'),('external','test','0000000000000000','x')],
          'val':[('b','train','0000000000000001','b')],
          'test':[('c','train','ffffffffffffffff','c')]}
    for split,values in rows.items():
        with (tmp_path/f'{split}.csv').open('w',newline='') as stream:
            writer=csv.writer(stream); writer.writerow(fields); writer.writerows(values)
    report=module.audit(tmp_path,tmp_path/'out',threshold=1)
    assert report['rows']==3 and report['examined_pairs']==3
    assert report['candidate_pairs']==1 and report['cross_split_pairs']==1 and not report['truncated']
    assert 'external' not in (tmp_path/'out.csv').read_text()
