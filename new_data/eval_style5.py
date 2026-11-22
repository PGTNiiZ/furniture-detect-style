"""Evaluate v3-style5 seeds. Synthetic/validation come from saved run metrics; real-source is predicted here.
Reports are kept separate (never averaged across test sets). Usage: python -I new_data/eval_style5.py
"""
import csv, json, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scripts.model_inference import FurniturePredictor
RUNS = ROOT/'artifacts/runs/furniture-v3-style5'; DATA = ROOT/'data/furniture-v3-style5'
OUT = RUNS/'report'; OUT.mkdir(exist_ok=True)
rd = lambda p: list(csv.DictReader(open(p, encoding='utf-8', newline='')))
status = {r['image_id']: r['relabel_status'] for r in rd(ROOT/'new_data/style-groups-v3/newlabel/real_source_test.csv')}
real = rd(DATA/'real_source_test.csv'); train = rd(DATA/'train.csv')
def met(t, p): return {'n': len(t), 'acc': accuracy_score(t, p), 'macro_f1': f1_score(t, p, average='macro', labels=sorted(set(t)), zero_division=0)}
def majority(split_rows, key):
    c = collections.Counter(r[key] for r in train).most_common(1)[0][0]
    t = [r[key] for r in split_rows]; return {'class': c, **met(t, [c]*len(t))}
res = {'seeds': {}, 'baseline_majority': {}}
for s in (42, 123, 2026):
    run = RUNS/f'densenet121-C6-{s}'
    pr = FurniturePredictor(run/'best.pt', device='auto')
    recs = pr.predict([(r['image_id'], DATA/r['image_path']) for r in real], top_k=1)
    assert all(x['status'] == 'ok' for x in recs)
    out = {}
    for task in ('type', 'style'):
        t = [r[task] for r in real]; p = [x[task]['label'] for x in recs]
        out[f'real_{task}'] = met(t, p)
        out[f'real_{task}_perclass'] = classification_report(t, p, output_dict=True, zero_division=0)
        for sub in ('kept', 'relabeled'):
            idx = [i for i, r in enumerate(real) if status[r['image_id']] == sub]
            out[f'real_{task}_{sub}'] = met([t[i] for i in idx], [p[i] for i in idx])
        labs = sorted(set(t) | set(p))
        cm = np.zeros((len(labs), len(labs)), int)
        for a, b in zip(t, p): cm[labs.index(a), labs.index(b)] += 1
        out[f'real_{task}_confusion'] = {'labels': labs, 'matrix': cm.tolist()}
    for split, fn in (('validation', 'validation.json'), ('synthetic_test', 'test.json')):
        m = json.load(open(run/fn))
        out[split] = {k: {'acc': m[k]['accuracy'], 'macro_f1': m[k]['macro_f1']} for k in ('type', 'style')}
    out['best_epoch'] = json.load(open(run/'summary.json')).get('best_epoch')
    res['seeds'][s] = out
for name, fn in (('validation', 'validation'), ('synthetic_test', 'synthetic_test'), ('real_source_test', None)):
    rows = rd(DATA/f'{name}.csv')
    res['baseline_majority'][name] = {k: majority(rows, k) for k in ('type', 'style')}
(OUT/'metrics.json').write_text(json.dumps(res, indent=2, default=float), encoding='utf-8')
def ms(vals): v = np.array(vals); return f'{v.mean():.3f} ± {v.std(ddof=1):.3f}'
L = ['| split | task | acc (mean±sd) | macro-F1 (mean±sd) | majority-baseline acc / F1 |', '|---|---|---|---|---|']
for split in ('validation', 'synthetic_test', 'real_source_test'):
    for task in ('type', 'style'):
        g = (lambda o, k: o[split][task][k]) if split != 'real_source_test' else (lambda o, k: o[f'real_{task}'][k])
        b = res['baseline_majority'][split][task]
        L.append(f"| {split} | {task} | {ms([g(o,'acc') for o in res['seeds'].values()])} | {ms([g(o,'macro_f1') for o in res['seeds'].values()])} | {b['acc']:.3f} / {b['macro_f1']:.3f} ({b['class']}) |")
L += ['', 'real_source_test style by relabel status (mean over seeds):']
for sub in ('kept', 'relabeled'):
    a = [o[f'real_style_{sub}'] for o in res['seeds'].values()]
    L.append(f"- {sub} (n={a[0]['n']}): acc {np.mean([x['acc'] for x in a]):.3f}, macro-F1 {np.mean([x['macro_f1'] for x in a]):.3f}")
(OUT/'summary.md').write_text('\n'.join(L), encoding='utf-8'); print('\n'.join(L))
print('best epochs', {s: o['best_epoch'] for s, o in res['seeds'].items()})
