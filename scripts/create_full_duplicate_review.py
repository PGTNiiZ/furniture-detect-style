"""Build a portable review of source-train cross-split dHash candidates."""
import argparse
import csv
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

from furniture.data import read_rows


NAMES = {'train': 'train', 'val': 'validation', 'test': 'synthetic_test'}
FIELDS = ['image_id_a', 'split_a', 'type_a', 'style_a', 'image_id_b', 'split_b',
          'type_b', 'style_b', 'distance', 'exact_pixels', 'decision']
README = '''# Cross-split duplicate review

Open `index.html` locally and inspect each pair. The images are resized JPEG previews;
consult the original dataset image identified by `image_id` when detail is insufficient.
The source image ID, original split, label metadata and Hamming distance are shown.
No real-source test image is included.

Select `same` when evidence supports the same source image or physical object;
`related` when evidence supports the same product/source identity, such as different
views; `different` for independently distinct items; or `uncertain` when the evidence
does not suffice. `unreviewed` is the initial state and is not a decision. Export the
CSV, retain it as evidence, and provide it to the snapshot workflow. Both `uncertain`
and `unreviewed` require resolution or an explicitly adopted quarantine policy.
Selections are stored in the browser until exported; export regularly. dHash is only
a candidate detector, and the previews alone do not establish exact-pixel equality.

`quarantine-proposal.json`, `quarantine-proposal-impact.csv` and
`quarantine-proposal-image-ids.csv` describe a conservative alternative only. They
quarantine every component touching a cross-split candidate, where components are
formed using all source-train dHash candidates and existing group/exact-pixel links.
This uses graph connectivity solely to limit risk, not to assert identity. It can
quarantine unrelated images. Check the per-class impact and approve a policy before
using it; these files do not change a snapshot. After any adopted quarantine or
human-confirmed same/related decisions, rebuild transitive groups, split whole groups,
and audit again; do not split groups by moving individual images.
'''


def create(data, candidates, output):
    data, candidates, output = Path(data).resolve(), Path(candidates).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Output directory exists; refusing to overwrite review evidence')
    audit_path = candidates.with_suffix('.json')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit.get('source_split') != 'train' or audit.get('truncated') is not False:
        raise ValueError('Expected an exhaustive source-train audit')
    rows = {}
    for split in NAMES:
        path = data / f'{split}.csv'
        if hashlib.sha256(path.read_bytes()).hexdigest() != audit['manifest_sha256'][split]:
            raise ValueError(f'{split} manifest differs from candidate audit')
        for row in read_rows(path):
            if row.get('source_split') != 'train':
                continue
            image_id = row['image_id']
            if image_id in rows:
                raise ValueError(f'Duplicate image ID: {image_id}')
            rows[image_id] = {**row, '_split': NAMES[split]}
    # This graph defines quarantine scope only; dHash edges are not identity decisions.
    parent = {image_id: image_id for image_id in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)
    groups = {}
    for image_id, row in rows.items():
        for field in ('group_id', 'pixel_hash'):
            key = (field, row.get(field))
            if not key[1]:
                continue
            if key in groups:
                union(image_id, groups[key])
            else:
                groups[key] = image_id
    pairs, seen = [], set()
    with candidates.open(encoding='utf-8', newline='') as stream:
        for pair in csv.DictReader(stream):
            sides = [rows.get(pair[f'image_id_{side}']) for side in ('a', 'b')]
            if any(row is None or row['_split'] != NAMES.get(pair[f'split_{side}'])
                   for row, side in zip(sides, ('a', 'b'))):
                raise ValueError('Candidate ID or split does not match audited source-train manifests')
            key = tuple(sorted((pair['image_id_a'], pair['image_id_b'])))
            if key in seen or key[0] == key[1]:
                raise ValueError(f'Duplicate or self pair: {key}')
            seen.add(key)
            union(*key)
            if pair['split_a'] == pair['split_b']:
                continue
            item = {'distance': pair['distance'], 'exact_pixels': pair['exact_pixels'], 'decision': 'unreviewed'}
            for side, row in zip(('a', 'b'), sides):
                item.update({f'image_id_{side}': row['image_id'], f'split_{side}': row['_split'],
                             f'type_{side}': row['type'], f'style_{side}': row['style']})
            pairs.append(item)
    if len(pairs) != audit['cross_split_pairs'] or len(seen) != audit['candidate_pairs']:
        raise ValueError('Candidate count differs from the audit')

    endpoints = {pair[f'image_id_{side}'] for pair in pairs for side in ('a', 'b')}
    quarantined_groups = {find(image_id) for image_id in endpoints}
    quarantine = {image_id for image_id in rows if find(image_id) in quarantined_groups}
    counts = defaultdict(Counter)
    for image_id, row in rows.items():
        for dimension, value in [('all', '*'), ('type', row['type']), ('style', row['style'])]:
            key = (row['_split'], dimension, value)
            counts[key]['before'] += 1
            counts[key]['quarantine'] += image_id in quarantine
    impact = [{'split': split, 'dimension': dimension, 'class': label, 'before': c['before'],
               'quarantine': c['quarantine'], 'remaining': c['before'] - c['quarantine']}
              for (split, dimension, label), c in sorted(counts.items())]

    output.mkdir(parents=True)
    (output / 'images').mkdir()
    for image_id in sorted(endpoints):
        image = (data / rows[image_id]['image_path']).resolve()
        if not image.is_relative_to(data) or not image.is_file():
            raise ValueError(f'Image path missing or outside dataset: {image_id}')
        # Numbered names keep paths safe even when source IDs contain punctuation.
    thumbnails = {image_id: f'{i:04d}.jpg' for i, image_id in enumerate(sorted(endpoints))}
    for image_id, name in thumbnails.items():
        with Image.open(data / rows[image_id]['image_path']) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            image.thumbnail((480, 480))
            image.save(output / 'images' / name, quality=85)
    with (output / 'candidates-unreviewed.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, FIELDS); writer.writeheader(); writer.writerows(pairs)
    with (output / 'quarantine-proposal-impact.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, ['split', 'dimension', 'class', 'before', 'quarantine', 'remaining'])
        writer.writeheader(); writer.writerows(impact)
    with (output / 'quarantine-proposal-image-ids.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, ['image_id', 'split', 'type', 'style', 'reason'])
        writer.writeheader()
        for image_id in sorted(quarantine):
            row = rows[image_id]
            writer.writerow({'image_id': image_id, 'split': row['_split'], 'type': row['type'],
                             'style': row['style'], 'reason': 'cross_split_candidate_endpoint' if image_id in endpoints else 'candidate_graph_or_existing_group_closure'})
    pair_counts = Counter('/'.join(sorted((p['split_a'], p['split_b']))) for p in pairs)
    report = {'status': 'proposal_only_not_applied', 'candidate_count': len(pairs),
              'candidate_counts_by_split': dict(sorted(pair_counts.items())), 'candidate_endpoints': len(endpoints),
              'quarantine_images': len(quarantine), 'source_train_rows': len(rows),
              'rule': 'Quarantine all connected components touching a cross-split candidate in the graph of all dHash candidates, pre-existing group IDs and exact-pixel matches. Graph connectivity is for quarantine only, not a duplicate identity finding.',
              'limitations': 'This is a deliberately conservative upper-bound proposal, not a finding of duplicate identity. Recheck class coverage, all candidate edges and group overlap before adopting; no snapshot was modified.',
              'candidate_csv_sha256': hashlib.sha256(candidates.read_bytes()).hexdigest(),
              'source_manifest_sha256': audit['manifest_sha256'], 'counts': [r for r in impact if r['dimension'] == 'all']}
    (output / 'quarantine-proposal.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    cards = []
    escape = lambda x: html.escape(str(x), quote=True)
    for pair in pairs:
        encoded = escape(json.dumps(pair, ensure_ascii=False))
        figures = []
        for side in ('a', 'b'):
            image_id = pair[f'image_id_{side}']
            label = f"{image_id} | {pair[f'split_{side}']} | type: {pair[f'type_{side}']} | style: {pair[f'style_{side}']}"
            figures.append(f'<figure><img loading="lazy" src="images/{thumbnails[image_id]}" alt="{escape(label)}"><figcaption>{escape(label)}</figcaption></figure>')
        cards.append(f'<article data-pair="{encoded}">{"".join(figures)}<div>Hamming distance: {escape(pair["distance"])}<br>Exact pixels: {escape(pair["exact_pixels"])}<br><label>Decision <select><option value="unreviewed">unreviewed</option><option value="same">same</option><option value="related">related</option><option value="different">different</option><option value="uncertain">uncertain</option></select></label></div></article>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Cross-split duplicate review</title>
<style>body{font:16px sans-serif;max-width:1100px;margin:auto;padding:24px}article{display:flex;align-items:center;gap:20px;border-top:1px solid #ccc;padding:16px}figure{margin:0;width:35%}img{width:100%;height:220px;object-fit:contain;background:#eee}button,select{padding:8px}</style>
<h1>Source-train cross-split candidate review</h1><p>dHash distance flags candidates; it does not establish identity. same = same image/object; related = same product/source or another shared identity; different = independently distinct; uncertain = insufficient evidence. This gallery has no real-source test images. Inspect every pair; unreviewed and uncertain remain unresolved. Decisions stay in this browser until exported.</p><button id="save">Download decisions CSV</button> <span id="progress"></span><main>'''+''.join(cards)+'''</main>
<script>
const cards=[...document.querySelectorAll('article')], fields='''+json.dumps(FIELDS)+''';
function update(){document.querySelector('#progress').textContent=cards.filter(c=>!['unreviewed','uncertain'].includes(c.querySelector('select').value)).length+' / '+cards.length+' resolved'}
cards.forEach(c=>{const p=JSON.parse(c.dataset.pair),key='furniture-full-review-v1:'+p.image_id_a+'|'+p.image_id_b,s=c.querySelector('select');s.value=localStorage.getItem(key)||'unreviewed';s.onchange=()=>{localStorage.setItem(key,s.value);update()}});update();
document.querySelector('#save').onclick=()=>{const q=x=>'"'+String(x).replaceAll('"','""')+'"';const lines=[fields.join(','),...cards.map(c=>{const p=JSON.parse(c.dataset.pair);p.decision=c.querySelector('select').value;return fields.map(f=>q(p[f])).join(',')})];const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\r\\n')],{type:'text/csv'}));a.download='duplicate-decisions.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
</script></html>'''
    (output / 'index.html').write_text(page, encoding='utf-8')
    (output / 'README.md').write_text(README, encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(create(args.data, args.candidates, args.output), indent=2))
