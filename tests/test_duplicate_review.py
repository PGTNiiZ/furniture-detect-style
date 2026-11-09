import csv
import importlib.util
from pathlib import Path


def test_gallery_includes_only_train_validation_candidates(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts'/'create_duplicate_review.py'
    spec=importlib.util.spec_from_file_location('create_duplicate_review',script)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    root=tmp_path/'data'; (root/'images').mkdir(parents=True)
    for split, image_id in [('train','a'),('val','b')]:
        (root/'images'/f'{image_id}.png').write_bytes(b'fixture')
        with (root/f'{split}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,['image_id','image_path']); writer.writeheader()
            writer.writerow({'image_id':image_id,'image_path':f'images/{image_id}.png'})
    candidate=tmp_path/'pairs.csv'
    with candidate.open('w',newline='') as f:
        writer=csv.DictWriter(f,['image_id_a','split_a','image_id_b','split_b','distance']); writer.writeheader()
        writer.writerows([{'image_id_a':'a','split_a':'train','image_id_b':'b','split_b':'val','distance':'1'},
                          {'image_id_a':'a','split_a':'train','image_id_b':'hidden','split_b':'test','distance':'1'}])
    result=module.create(root,candidate,tmp_path/'gallery')
    page=(tmp_path/'gallery'/'index.html').read_text(encoding='utf-8')
    assert result['pairs']==1 and 'hidden' not in page and 'Download decisions CSV' in page
