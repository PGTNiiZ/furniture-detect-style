import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from new_data.build_conservative_snapshot import Groups, jpeg_sha, main


def test_transitive_groups():
    groups=Groups(); groups.union('a','b'); groups.union('b','c')
    assert groups.find('a') == groups.find('c')
    assert groups.find('z') != groups.find('a')


def test_candidate_link_to_holdout_excludes_whole_group(tmp_path, monkeypatch):
    root=tmp_path/'v2'; (root/'images').mkdir(parents=True)
    train=[]
    for index in range(3):
        generator=np.random.default_rng(index)
        image=Image.fromarray(generator.integers(0,256,(32,32,3),dtype=np.uint8))
        ident=f'train-{index:07d}'; image.save(root/'images'/f'{ident}.png')
        train.append({'image_id':ident,'image_path':f'images/{ident}.png','type':'bed','style':'modern'})
    for split, index in [('validation',10),('synthetic_test',11),('real_source_test',12)]:
        image=Image.fromarray(np.random.default_rng(index).integers(0,256,(32,32,3),dtype=np.uint8))
        image.save(root/'images'/f'{split}.png')
        data=[{'image_id':split+'-id','image_path':f'images/{split}.png','type':'bed','style':'modern'}]
        write(root/(split+'.csv'),data)
    write(root/'train.csv',train)
    write(root/'group_audit.csv',[{'image_id':r['image_id'],'candidate_component_id':('shared' if i<2 else 'other')} for i,r in enumerate(train)])
    (root/'quarantine.csv').write_text('image_id\n',encoding='utf8')
    (root/'audit.json').write_text(json.dumps({'class_mapping':{'type':{'bed':0},'style':{'modern':0}}}),encoding='utf8')
    candidates=tmp_path/'candidates.csv'
    write(candidates,[{'train_id':train[0]['image_id'],'holdout_id':'validation-id','holdout_split':'validation','distance':'3'}])
    archive=tmp_path/'source.zip'; prefix='furniture_dataset/data/processed/'
    with zipfile.ZipFile(archive,'w') as z:
        buf=io.StringIO(newline=''); w=csv.DictWriter(buf,fieldnames=['src_index','path','type','style','style_raw']); w.writeheader()
        for i,row in enumerate(train):
            path=f'images/train/bed/modern/{i}.jpg'
            w.writerow({'src_index':i,'path':path,'type':'bed','style':'modern','style_raw':'modern'})
            with Image.open(root/row['image_path']) as im:
                encoded=io.BytesIO(); im.convert('RGB').resize((256,256),Image.Resampling.LANCZOS).save(encoded,'JPEG',quality=95)
            z.writestr(prefix+path,encoded.getvalue())
            assert hashlib.sha256(encoded.getvalue()).hexdigest()==jpeg_sha(root/row['image_path'])
        z.writestr(prefix+'train.csv',buf.getvalue())
        z.writestr(prefix+'labels.json',json.dumps({'type2id':{'bed':0},'style2id':{'modern':0}}))
    config=tmp_path/'cfg.json'
    config.write_text(json.dumps({'archive_path':str(archive),'archive_prefix':prefix,
        'expected_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'v2_root':str(root),
        'holdout_candidates':str(candidates),'dhash_threshold':0}),encoding='utf8')
    out=tmp_path/'snapshot'; monkeypatch.setattr(sys,'argv',['builder','--config',str(config),'--output',str(out)])
    main()
    with (out/'train.csv').open(encoding='utf8',newline='') as f: kept=list(csv.DictReader(f))
    with (out/'excluded.csv').open(encoding='utf8',newline='') as f: excluded=list(csv.DictReader(f))
    assert [r['image_id'] for r in kept] == [train[2]['image_id']]
    assert {r['image_id'] for r in excluded} == {train[0]['image_id'],train[1]['image_id']}
    assert all('candidate_component_touches_holdout' in r['reasons'] for r in excluded)


def write(path, rows):
    with Path(path).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
