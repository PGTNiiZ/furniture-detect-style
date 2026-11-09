"""Build a portable visual review of prepared classification datasets."""
import html
import json
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

from .augmentation import pil_transforms, settings
from .data import Square, read_rows


def create_gallery(root, output, size, augmentation, seed=42, variants=3):
    if variants < 0:
        raise ValueError('variants must be non-negative')
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('Gallery output exists; choose a new directory')
    splits = {s: read_rows(root / f'{s}.csv') for s in ('train', 'val', 'test')}
    output.mkdir(parents=True)
    assets = output / 'images'
    assets.mkdir()
    random.seed(seed)
    torch.manual_seed(seed)
    transform = T.Compose(pil_transforms(augmentation))
    cards = []
    for split, rows in splits.items():
        for row in rows:
            idx = len(cards)
            with Image.open(root / row['image_path']) as image:
                base = Square(size)(image)
            files = []
            for n in range(1 + (variants if split == 'train' else 0)):
                name = f'{idx:08d}-{n}.jpg'
                (base if n == 0 else transform(base)).save(assets / name, quality=90)
                files.append('images/' + name)
            cards.append({'id': row['image_id'], 'type': row['type'], 'style': row['style'],
                          'split': split, 'path': row['image_path'], 'images': files})
    # Escape HTML-significant characters so dataset labels cannot terminate the script.
    payload = json.dumps(cards, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    config = html.escape(json.dumps(settings(augmentation), ensure_ascii=False))
    page = '''<!doctype html><html lang="th"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset review</title><style>body{font:15px system-ui;margin:2rem;background:#f4f5f7;color:#202632}select,input{padding:.5rem;margin:.3rem}#cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:1rem}.card{background:white;border-radius:8px;padding:1rem;overflow:hidden}.images{display:flex;gap:.5rem;overflow:auto}.images figure{margin:0;min-width:130px;width:130px}.images img{width:130px;height:130px;object-fit:contain;background:#eee}small{overflow-wrap:anywhere}pre{white-space:pre-wrap}</style>
<h1>ตรวจสอบ Dataset</h1><p>ภาพแรกผ่าน resize/pad เหมือนก่อนเข้าโมเดล ภาพถัดไปเป็นตัวอย่าง augmentation แบบสุ่มของ train เท่านั้น; val/test ไม่มี augmentation</p>
<details><summary>การตั้งค่า augmentation และ seed</summary><pre>CONFIG\nseed: SEED</pre></details>
<label>Split <select id="split"></select></label><label>Type <select id="type"></select></label><label>Style <select id="style"></select></label><label>ค้นหา ID / path <input id="search"></label><p id="count"></p><main id="cards"></main>
<script>const data=DATA; const fields=['split','type','style'];
for(const f of fields){const el=document.getElementById(f);el.add(new Option('ทั้งหมด',''));for(const v of [...new Set(data.map(x=>x[f]))].sort())el.add(new Option(v,v));el.onchange=render} document.getElementById('search').oninput=render;
function render(){const q=document.getElementById('search').value.toLowerCase();const rows=data.filter(x=>fields.every(f=>!document.getElementById(f).value||x[f]===document.getElementById(f).value)&&(x.id+' '+x.path).toLowerCase().includes(q));document.getElementById('count').textContent=`แสดง ${rows.length} / ${data.length} ภาพ`;const container=document.getElementById('cards');container.replaceChildren();for(const row of rows){const card=document.createElement('article');card.className='card';const title=document.createElement('strong');title.textContent=`${row.type} / ${row.style} · ${row.split}`;const info=document.createElement('p');const small=document.createElement('small');small.textContent=`${row.id} — ${row.path}`;info.append(small);const images=document.createElement('div');images.className='images';row.images.forEach((src,i)=>{const figure=document.createElement('figure'),img=document.createElement('img'),caption=document.createElement('figcaption');img.src=src;img.loading='lazy';img.alt=i?`Augmented ${i}`:'Original';caption.textContent=i?`Augmented ${i}`:'Original (resize/pad)';figure.append(img,caption);images.append(figure)});card.append(title,info,images);container.append(card)}}render();</script></html>'''
    page = page.replace('CONFIG', config).replace('SEED', str(seed)).replace('DATA', payload)
    (output / 'index.html').write_text(page, encoding='utf-8')
    return {'gallery': str((output / 'index.html').resolve()), 'counts': {s: len(rows) for s, rows in splits.items()},
            'type_counts': dict(Counter(r['type'] for rows in splits.values() for r in rows)),
            'variants_per_train_image': variants, 'seed': seed}
