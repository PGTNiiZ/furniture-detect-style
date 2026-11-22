"""Blind review pack for style5 criteria. Reviewer sees image id + type only; answer key is separate.
Usage: .venv/Scripts/python.exe -I new_data/make_style5_review.py
"""
import csv, json, random, collections
from pathlib import Path
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT/'data/furniture-v3-style5'; OUT = ROOT/'new_data/style5-criteria/review-v1'; OUT.mkdir(exist_ok=True)
g = json.load(open(ROOT/'new_data/style5-criteria/style_groups_criteria_v1.json'))
m = {s: k for k, v in g['groups'].items() for s in v}
und = list(g['undecided_need_human'])
rows = list(csv.DictReader(open(DATA/'train.csv', encoding='utf-8')))
rng = random.Random(42)
def pick(styles, n):
    by = collections.defaultdict(list)
    for r in rows:
        if r['style_orig'] in styles: by[r['style_orig']].append(r)
    for v in by.values(): rng.shuffle(v)
    out, keys = [], sorted(by)
    while len(out) < n:  # round robin across original styles
        for k in keys:
            if by[k] and len(out) < n: out.append(by[k].pop())
    return out
sel = []
for grp, styles in g['groups'].items():
    sel += [(r, grp) for r in pick(styles, 14)]
for s in und: sel += [(r, 'UNDECIDED') for r in pick([s], 5)]
rng.shuffle(sel)
key, sheet_rows = [], []
CELL, COLS, PER = 300, 5, 20
for si in range(0, len(sel), PER):
    chunk = sel[si:si+PER]
    sheet = Image.new('RGB', (COLS*CELL, 4*(CELL+24)), 'white'); d = ImageDraw.Draw(sheet)
    for i, (r, grp) in enumerate(chunk):
        n = si+i+1; x, y = (i % COLS)*CELL, (i//COLS)*(CELL+24)
        with Image.open(DATA/r['image_path']) as im:
            im = im.convert('RGB'); im.thumbnail((CELL-8, CELL-8)); sheet.paste(im, (x+4, y+24))
        d.text((x+6, y+5), f"#{n:03d}  {r['type']}", fill='black')
        key.append({'no': n, 'image_id': r['image_id'], 'original_style': r['style_orig'], 'proposed_group': grp})
        sheet_rows.append({'no': n, 'image_id': r['image_id'], 'type': r['type'], 'your_group': '', 'fits_criteria_yes_no_unsure': '', 'note': ''})
    sheet.save(OUT/f'sheet-{si//PER+1}.png')
for name, data in (('review.csv', sheet_rows), ('ANSWER-KEY-do-not-open-before-review.csv', key)):
    with open(OUT/name, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator='\n'); w.writeheader(); w.writerows(data)
print(len(sel), 'images,', len(sel)//PER, 'sheets')
