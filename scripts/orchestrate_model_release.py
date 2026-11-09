"""Continue the frozen model-only protocol after the existing screening finishes."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

import torch

from furniture.benchmark import benchmark
from furniture.external_evaluation import evaluate_external
from furniture.pipeline import _process_alive
from furniture.suite import _complete_evidence, _suite_lock, plan_suite, run_suite, select_screening
from scripts.release_evidence import bundle, controlled_contrasts, repeated_summary, snapshot_contract, synthetic_report, validate_run


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT/'configs/project_v2_conservative.yaml'
SUITES = ROOT/'artifacts/runs/suites'
OUTPUT = ROOT/'artifacts/model-only-release'
SCREENING = SUITES/'furniture-v2-screening.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def condition_id(cfg):
    match = re.search(r'-(C[1-8])-\d+-(?:type|style)(?:-style)?$', cfg['experiment']['id'])
    if not match:
        raise ValueError('Experiment ID has no valid controlled condition')
    return match.group(1)


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(temporary, path)


def status(stage, **extra):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT/'workflow-status.json'
    temporary = OUTPUT/f'.workflow-status.{os.getpid()}.tmp'
    temporary.write_text(json.dumps({'stage': stage, 'updated_at': datetime.now(timezone.utc).isoformat(), **extra},
                                    ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def completed_runs(manifest, expected=None):
    data = read(manifest)
    runs = data['runs']
    if expected is not None and len(runs) != expected:
        raise ValueError(f'Unexpected run count in {manifest}')
    if any(item['status'] != 'completed' for item in runs):
        raise ValueError(f'Suite is incomplete: {manifest}')
    for item in runs:
        run = Path(item['attempts'][-1]['run_dir'])
        if not _complete_evidence(run, item['config_sha256']):
            raise ValueError(f'Invalid completion evidence: {run}')
    return runs


def source_contract(items, expected=None):
    provenance = [read(Path(item['attempts'][-1]['run_dir'])/'provenance.json')['code_sha256'] for item in items]
    current = {p.name:sha(p) for p in sorted((ROOT/'src/furniture').glob('*.py'))}
    baseline = expected if expected is not None else current
    if current != baseline or any(value != baseline for value in provenance):
        raise ValueError('Source differs across runs or from the frozen screening source')
    return baseline


def validate_suite_runs(items):
    snapshot = snapshot_contract(ROOT/'data/furniture-v2-conservative')
    return [validate_run(item, snapshot) for item in items]


def wait_for_screening(owner_pid):
    while True:
        data = read(SCREENING)
        counts = dict(Counter(item['status'] for item in data['runs']))
        status('waiting_for_screening', counts=counts, screening_manifest=str(SCREENING))
        if counts.get('failed') or counts.get('interrupted'):
            raise ValueError('Screening has failed or interrupted runs; inspect and recover the suite')
        if counts.get('completed') == 10 and len(data['runs']) == 10:
            return completed_runs(SCREENING, 10)
        if not _process_alive(owner_pid):
            raise ValueError('Screening process exited before completing all 10 runs')
        time.sleep(30)


def screen_selection(items):
    runs = [Path(item['attempts'][-1]['run_dir']) for item in items]
    scores = []
    for run in runs:
        payload = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
        scores.append(sum(payload['val'][task]['macro_f1'] for task in ('type', 'style'))/2)
    tied = {score for score in scores if scores.count(score) > 1}
    for run, score in zip(runs, scores):
        if score in tied and not (run/'benchmark.json').exists():
            temporary = run/f'.benchmark-{uuid.uuid4().hex}.json'
            benchmark(run/'best.pt', temporary, device='auto', batch_size=1, warmup=5, repetitions=30)
            os.rename(temporary, run/'benchmark.json')
    result = select_screening([str(run) for run in runs], count=3)
    result['checkpoint_sha256'] = {str(run):sha(run/'best.pt') for run in runs}
    return result


def run_stage(stage, backbones, expected, source, conditions=None, seeds=None, recover=False):
    manifest = SUITES/f'furniture-v2-model-only-{stage}.json'
    status(stage, manifest=str(manifest))
    if manifest.exists():
        result = run_suite(None, manifest, retry_failed=recover, recover_running=recover)
    else:
        plans = plan_suite(PROJECT, stage, backbones=backbones, conditions=conditions, seeds=seeds)
        if len(plans) != expected:
            raise ValueError(f'Unexpected planned count for {stage}')
        result = run_suite(plans, manifest)
    if result['counts']['completed'] != expected:
        raise ValueError(f'{stage} did not complete; inspect {manifest}: {result["counts"]}')
    items = completed_runs(manifest, expected)
    source_contract(items, source)
    validate_suite_runs(items)
    return items


def select_controlled(items):
    candidates, excluded = [], []
    for item in items:
        cfg = item['config']
        condition = condition_id(cfg)
        if condition == 'C1':
            continue
        if condition not in {f'C{i}' for i in range(2, 9)}:
            raise ValueError(f'Unexpected controlled condition: {condition}')
        run = Path(item['attempts'][-1]['run_dir'])
        payload = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
        if payload.get('shared_backbone_update_epochs', 0) < 1:
            excluded.append({'run_dir':str(run), 'reason':'Selected checkpoint has no shared backbone update'})
            continue
        score = sum(payload['val'][task]['macro_f1'] for task in ('type', 'style'))/2
        if not math.isfinite(score):
            raise ValueError(f'Nonfinite validation score: {run}')
        candidates.append({'backbone':cfg['model']['backbone'], 'condition':condition,
                           'validation_mean_macro_f1':score, 'run_dir':str(run),
                           'checkpoint_sha256':sha(run/'best.pt'), 'split_hashes':payload['split_hashes']})
    if len(candidates) + len(excluded) != 21 or not candidates:
        raise ValueError('Expected 21 C2-C8 runs and at least one checkpoint with a backbone update')
    selected = sorted(candidates, key=lambda x:(-x['validation_mean_macro_f1'], x['backbone'], x['condition']))[0]
    return {'rule':'highest validation mean macro-F1 across type/style in C2-C8 with >=1 shared-backbone update; tie: backbone then condition',
            'selected':selected, 'candidates':candidates, 'excluded':excluded,
            'frozen_at':datetime.now(timezone.utc).isoformat(), 'test_used_for_selection':False}


def write_validation_table(path, stages):
    fields = ['stage','backbone','condition','seed','task','accuracy','macro_precision','macro_recall','macro_f1',
              'weighted_f1','epochs','best_epoch','shared_backbone_update_epochs','run_dir','checkpoint_sha256']
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for stage, items in stages.items():
            for item in items:
                cfg = item['config']
                run = Path(item['attempts'][-1]['run_dir'])
                metrics, summary = read(run/'validation.json'), read(run/'summary.json')
                condition = condition_id(cfg)
                for task in cfg['training']['tasks']:
                    values = metrics[task]
                    writer.writerow({'stage':stage, 'backbone':cfg['model']['backbone'], 'condition':condition,
                        'seed':cfg['training']['seed'], 'task':task, **{k:values[k] for k in
                        ('accuracy','macro_precision','macro_recall','macro_f1','weighted_f1')},
                        'epochs':summary['epochs'], 'best_epoch':summary['best_epoch'],
                        'shared_backbone_update_epochs':summary['shared_backbone_update_epochs'],
                        'run_dir':str(run), 'checkpoint_sha256':sha(run/'best.pt')})


def artifact_contexts(final, controlled, ablation, repeated):
    root = ROOT/'data/furniture-v2-conservative'
    checkpoint = Path(final['run_dir'])/'best.pt'
    snapshot = snapshot_contract(root)
    if final['checkpoint_sha256'] != sha(checkpoint) or final['split_hashes'] != snapshot['split_hashes']:
        raise ValueError('Selected checkpoint or snapshot split hashes differ from the frozen selection')
    all_runs = controlled + ablation + repeated
    run_evidence = [validate_run(item, snapshot) for item in all_runs]
    context = {'checkpoint_sha256':final['checkpoint_sha256'], 'snapshot':snapshot}
    comparison_context = {**context, 'runs':run_evidence}
    protocol = ROOT/'artifacts/protocols/furniture-v2-external-protocol.json'
    external_context = {**context, 'protocol_sha256':sha(protocol), 'external_manifest_sha256':sha(root/'real_source_test.csv')}
    return root, checkpoint, snapshot, context, comparison_context, protocol, external_context


def complete_artifacts(final, controlled, ablation, repeated):
    from furniture.train import evaluate
    root, checkpoint, snapshot, context, comparison_context, protocol, external_context = artifact_contexts(final, controlled, ablation, repeated)
    def build_comparisons(path):
        write_validation_table(path/'validation-comparisons.csv',
                               {'controlled':controlled, 'single-task':ablation, 'repeated':repeated})
        repeated_summary(controlled + repeated, final['backbone'], path/'repeated-seeds.csv')
        controlled_contrasts(controlled, ablation, final['backbone'], path/'controlled-contrasts.csv')
    status('comparison_report', run_dir=final['run_dir'])
    bundle(OUTPUT/'validation-report', comparison_context,
           ['validation-comparisons.csv','repeated-seeds.csv','controlled-contrasts.csv'], build_comparisons)
    status('synthetic_test', run_dir=final['run_dir'])
    test = bundle(OUTPUT/'synthetic-test', context,
                  ['test.json','test_predictions.csv','test_type_confusion.csv','test_style_confusion.csv'],
                  lambda path:evaluate(root, checkpoint, path/'test.json', split='test'))
    report_context = {**context, 'evaluation_evidence_sha256':sha(test/'evidence.json')}
    status('synthetic_report', run_dir=final['run_dir'])
    bundle(OUTPUT/'synthetic-test-report', report_context,
           ['report.json','errors.csv','type-confusion.csv','style-confusion.csv','type-confusion.png','style-confusion.png'],
           lambda path:synthetic_report(path, test, snapshot, final['checkpoint_sha256']))
    status('external_reference', run_dir=final['run_dir'])
    bundle(OUTPUT/'external-reference', external_context,
           ['result/metrics.json','result/predictions.csv','result/missing_or_unseen.csv'],
           lambda path:evaluate_external(protocol, root/'train.csv', root/'real_source_test.csv', root, checkpoint, path/'result'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--screening-pid', type=int, required=True)
    parser.add_argument('--recover', action='store_true', help='Use the existing suite recovery checks for incomplete stages')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with _suite_lock(OUTPUT/'workflow.lock'):
        try:
            screening = wait_for_screening(args.screening_pid)
            source = source_contract(screening)
            validate_suite_runs(screening)
            selection_path = OUTPUT/'screening-selection.json'
            if not selection_path.exists():
                write_once(selection_path, screen_selection(screening))
            screening_selection = read(selection_path)
            backbones = [r['backbone'] for r in screening_selection['selected']]
            if len(backbones) != 3 or len(set(backbones)) != 3:
                raise ValueError('Screening selection must contain three distinct backbones')
            allowed = {str(Path(item['attempts'][-1]['run_dir'])):item['config']['model']['backbone'] for item in screening}
            if any(allowed.get(r['run_dir']) != r['backbone'] or
                   sha(Path(r['run_dir'])/'best.pt') != screening_selection['checkpoint_sha256'].get(r['run_dir'])
                   for r in screening_selection['selected']):
                raise ValueError('Screening selection is not bound to completed checkpoints')
            controlled = run_stage('controlled', backbones, 24, source, recover=args.recover)
            final_path = OUTPUT/'selection-frozen.json'
            if not final_path.exists():
                write_once(final_path, select_controlled(controlled))
            final = read(final_path)['selected']
            if final != select_controlled(controlled)['selected']:
                raise ValueError('Frozen selection differs from the registered validation rule')
            checkpoint = Path(final['run_dir'])/'best.pt'
            valid_runs = {str(Path(item['attempts'][-1]['run_dir'])) for item in controlled}
            if final['run_dir'] not in valid_runs or sha(checkpoint) != final['checkpoint_sha256']:
                raise ValueError('Frozen selection checkpoint changed')
            winner = final['backbone']
            ablation = run_stage('ablation', [winner], 2, source, recover=args.recover)
            repeated = run_stage('repeated', [winner], 8, source, conditions=['C1','C2','C3','C5'], seeds=[123,2026], recover=args.recover)
            complete_artifacts(final, controlled, ablation, repeated)
            status('complete', selected=final, note='Synthetic test and external source annotations reported separately.')
        except BaseException as error:
            status('stopped', error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
