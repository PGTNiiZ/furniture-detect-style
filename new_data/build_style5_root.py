"""Build data/furniture-v3-style5 from v2-conservative + newlabel manifests.
style := style_group (original 37-class label kept as style_orig). Images copied unchanged.
Unmappable real-source row(s) are excluded and listed in excluded_unmappable.csv.
Usage: python -I new_data/build_style5_root.py
"""
import csv, hashlib, json, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OLD, NEW = ROOT/'data/furniture-v2-conservative', ROOT/'data/furniture-v3-style5'
LAB = ROOT/'new_data/style-groups-v3/newlabel'
def rd(p): return list(csv.DictReader(open(p, encoding='utf-8', newline='')))
def wr(p, rows, fields):
    with open(p, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); w.writeheader(); w.writerows(rows)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if NEW.exists(): sys.exit('exists: '+str(NEW))
NEW.mkdir(); (NEW/'images').mkdir()
fields = ['image_id','image_path','type','style','style_orig','source_split','pixel_hash','dhash','group_id']
excluded = []; counts = {}
for split in ('train','validation','synthetic_test','real_source_test'):
    out = []
    for r in rd(LAB/f'{split}.csv'):
        if not r['style_group'] or not r['type']:
            excluded.append(r); continue
        src = OLD/r['image_path']
        assert src.exists(), src
        shutil.copy2(src, NEW/r['image_path'])
        out.append(dict(image_id=r['image_id'], image_path=r['image_path'], type=r['type'], style=r['style_group'],
                        style_orig=r['style'], source_split=r['source_split'], pixel_hash=r['pixel_hash'],
                        dhash=r['dhash'], group_id=r['group_id']))
    wr(NEW/f'{split}.csv', out, fields); counts[split] = len(out)
shutil.copyfile(NEW/'validation.csv', NEW/'val.csv'); shutil.copyfile(NEW/'synthetic_test.csv', NEW/'test.csv')
if excluded: wr(NEW/'excluded_unmappable.csv', excluded, list(excluded[0].keys()))
shutil.copyfile(LAB/'style_groups.json', NEW/'style_groups.json')
for f in ('quarantine.csv','DATASET_NOTICE.md'): shutil.copyfile(OLD/f, NEW/f)
meta = {'derived_from':'data/furniture-v2-conservative','label_source':'newlabel.zip sha256 f6c141973d00c83ceb4bae4ae8e82b2fc09284238de490b75cd2aaa800fe0380',
        'counts':counts,'excluded_unmappable':[r['image_id'] for r in excluded],
        'manifest_sha256':{s:sha(NEW/f'{s}.csv') for s in counts}}
(NEW/'dataset_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print(counts, [r['image_id'] for r in excluded])
