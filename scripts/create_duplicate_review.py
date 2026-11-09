"""Make an offline train/validation candidate gallery for human decisions."""
import argparse
import csv
import html
import os
from pathlib import Path
from urllib.parse import quote

from furniture.data import read_rows


def create(data, candidates, output):
    data, output = Path(data).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Output directory exists')
    rows = {r['image_id']:r for split in ('train','val') for r in read_rows(data/f'{split}.csv')}
    with Path(candidates).open(encoding='utf-8',newline='') as stream:
        pairs = [r for r in csv.DictReader(stream) if {r['split_a'],r['split_b']} == {'train','val'}]
    for pair in pairs:
        if pair['image_id_a'] not in rows or pair['image_id_b'] not in rows:
            raise ValueError('Candidate IDs do not match train/validation manifests')
    output.mkdir(parents=True)
    panels=[]
    for pair in pairs:
        cards=[]
        for side in ('a','b'):
            item=rows[pair[f'image_id_{side}']]
            image=(data/item['image_path']).resolve()
            if not image.is_relative_to(data) or not image.is_file():
                raise ValueError('Image path is missing or outside the dataset')
            relative=quote(os.path.relpath(image,output).replace('\\','/'),safe='/')
            label=html.escape(f"{item['image_id']} ({pair[f'split_{side}']})")
            cards.append(f'<figure><img loading="lazy" src="{relative}" alt="{label}"><figcaption>{label}</figcaption></figure>')
        a=html.escape(pair['image_id_a'],quote=True); b=html.escape(pair['image_id_b'],quote=True)
        distance=html.escape(pair['distance'])
        panels.append(f'<article data-a="{a}" data-b="{b}">{"".join(cards)}<div>Hamming distance: {distance}<br>'
                      '<label>Decision <select><option value="uncertain">uncertain</option>'
                      '<option value="same">same source/object</option><option value="different">different</option>'
                      '</select></label></div></article>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Train/validation duplicate review</title>
<style>body{font:16px sans-serif;max-width:1000px;margin:auto;padding:24px}article{display:flex;align-items:center;gap:20px;border-top:1px solid #ccc;padding:16px}figure{margin:0;width:35%}img{width:100%;height:220px;object-fit:contain;background:#eee}button,select{padding:8px}</style>
<h1>Train/validation candidate review</h1><p>Hamming distance does not prove shared identity. Review each pair; this page contains no source test images. Selections stay in this browser until exported.</p><button id="save">Download decisions CSV</button><main>'''+''.join(panels)+'''</main>
<script>
const cards=[...document.querySelectorAll('article')];
cards.forEach(c=>{const key='furniture-review:'+c.dataset.a+'|'+c.dataset.b,s=c.querySelector('select');s.value=localStorage.getItem(key)||'uncertain';s.onchange=()=>localStorage.setItem(key,s.value)});
document.getElementById('save').onclick=()=>{const q=x=>'"'+String(x).replaceAll('"','""')+'"';const lines=['image_id_a,image_id_b,decision',...cards.map(c=>[c.dataset.a,c.dataset.b,c.querySelector('select').value].map(q).join(','))];const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\r\\n')],{type:'text/csv'}));a.download='duplicate-decisions.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
</script></html>'''
    (output/'index.html').write_text(page,encoding='utf-8')
    return {'gallery':str(output/'index.html'),'pairs':len(pairs),'split':'train/val'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--candidates',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(create(args.data,args.candidates,args.output))
