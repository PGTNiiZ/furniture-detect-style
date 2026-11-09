"""Recompute held-out and robustness metrics against frozen manifests."""
import csv
import json
import math
from pathlib import Path

from furniture.robustness import task_metrics


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def matched(manifest, predictions):
    def keyed(values, key):
        result = {}
        for row in values:
            value = row.get(key)
            if not value or value in result:
                raise ValueError(f'Missing or duplicate {key}')
            result[value] = row
        return result
    reference = keyed(rows(manifest), 'image_id')
    predicted = keyed(rows(predictions), 'image_id')
    if set(reference) != set(predicted):
        raise ValueError('Prediction IDs do not exactly match the manifest')
    if not reference:
        raise ValueError('Manifest must not be empty')
    return [(reference[key], predicted[key]) for key in sorted(reference)]


def same_metrics(actual, recorded):
    for key in ('accuracy','macro_precision','macro_recall','macro_f1','weighted_f1'):
        if not math.isclose(actual[key], recorded[key], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f'Metric mismatch: {key}')
    if actual['confusion_matrix'] != recorded['confusion_matrix']:
        raise ValueError('Confusion matrix mismatch')


def audit_confusion_csv(path, mapping, recorded):
    labels = [name for name,_ in sorted(mapping.items(),key=lambda item:item[1])]
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        table = list(csv.reader(stream))
    if not table or table[0][1:] != labels or [r[0] for r in table[1:]] != labels:
        raise ValueError('Confusion CSV labels do not match checkpoint mapping')
    try:
        matrix = [[int(value) for value in row[1:]] for row in table[1:]]
    except ValueError:
        raise ValueError('Confusion CSV contains invalid counts') from None
    if matrix != recorded['confusion_matrix']:
        raise ValueError('Confusion CSV counts differ from metrics')


def audit_external_exclusions(predictions, exclusions):
    all_rows = rows(predictions)
    expected = {r['image_id']:r for r in all_rows if r['type_state'] != 'known' or r['style_state'] != 'known'}
    cases = rows(exclusions)
    observed = {r['image_id']:r for r in cases}
    if len(observed) != len(cases) or observed != expected:
        raise ValueError('External missing/unseen cases differ from prediction eligibility')


def audit_closed_set(manifest, predictions, metrics, mapping):
    pairs = matched(manifest, predictions)
    scores = {}
    for task, classes in mapping.items():
        names = [name for name,_ in sorted(classes.items(),key=lambda item:item[1])]
        truth, predicted = [], []
        for reference, row in pairs:
            if row.get('image_path') != reference['image_path'] or row.get(f'true_{task}') != reference[task]:
                raise ValueError(f'Reference label or image path differs for {reference["image_id"]}')
            label = row.get(f'pred_{task}')
            if reference[task] not in classes or label not in classes:
                raise ValueError('Prediction or reference is outside checkpoint mapping')
            truth.append(classes[reference[task]])
            predicted.append(classes[label])
        scores[task] = task_metrics(truth,predicted,names)
        same_metrics(scores[task], metrics[task])
    return {'samples':len(pairs),'tasks':list(mapping),'result':'passed'}


def audit_external(manifest, predictions, report, mapping):
    pairs = matched(manifest, predictions)
    both_known, joint_correct = 0, 0
    for task, classes in mapping.items():
        names = [name for name,_ in sorted(classes.items(),key=lambda item:item[1])]
        truth, predicted = [], []
        counts = {'missing':0,'known':0,'unseen':0}
        for reference, row in pairs:
            value = reference[task]
            state = 'missing' if not value else 'known' if value in classes else 'unseen'
            if row.get(f'true_{task}') != value or row.get(f'{task}_state') != state:
                raise ValueError(f'External reference or eligibility differs for {reference["image_id"]}')
            label = row.get(f'predicted_{task}')
            if label not in classes:
                raise ValueError('External prediction is outside checkpoint mapping')
            counts[state] += 1
            if state == 'known':
                truth.append(classes[value]); predicted.append(classes[label])
        part = report['tasks'][task]
        if (part['eligible'] != len(truth) or part['total'] != len(pairs)
                or not math.isclose(part['coverage'], len(truth)/len(pairs), abs_tol=1e-12)
                or report['annotation_inspection']['counts'][task] != counts):
            raise ValueError(f'External eligible counts or coverage mismatch: {task}')
        if truth:
            same_metrics(task_metrics(truth,predicted,names),part['metrics'])
        elif part['metrics'] is not None:
            raise ValueError('Metrics supplied for empty eligible subset')
    for reference,row in pairs:
        if all(reference[t] in mapping[t] for t in mapping):
            both_known += 1
            joint_correct += int(all(row[f'predicted_{t}'] == reference[t] for t in mapping))
    joint = report['joint']
    if (joint['eligible'] != both_known or joint['total'] != len(pairs) or joint['correct'] != joint_correct
            or not math.isclose(joint['coverage'], both_known/len(pairs), abs_tol=1e-12)
            or (both_known and not math.isclose(joint['accuracy'], joint_correct/both_known, abs_tol=1e-12))):
        raise ValueError('External joint denominator or correctness mismatch')
    return {'samples':len(pairs),'known_both':both_known,'result':'passed'}
