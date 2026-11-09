"""Reports from completed runs; no numbers are synthesized for missing experiments."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

import torch
import yaml

from .pipeline import compare_runs


def _json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _condition(cfg, payload):
    training = {key:value for key,value in cfg['training'].items() if key != 'seed'}
    return {'dataset':cfg['dataset'], 'model':cfg['model'], 'training':training,
            'augmentation':cfg['augmentation'], 'mapping':payload['mapping'], 'splits':payload['split_hashes']}


def _write_csv(path, rows, fields):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_report(run_dirs, output, split='validation', vary=()):
    """Write evidence-backed tables/figures; caller chooses explicit comparison axes."""
    if not run_dirs:
        raise ValueError('At least one run is required')
    if len(set(map(lambda x: str(Path(x).resolve()), run_dirs))) != len(run_dirs):
        raise ValueError('The same run was supplied more than once')
    if split not in ('validation', 'test'):
        raise ValueError('split must be validation or test')
    output = Path(output)
    if output.exists():
        raise ValueError('Report directory exists')
    # Also enforces post-P0 completed status, mapping and non-varying controls.
    compare_runs(run_dirs, vary=vary)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError('Install report support with python -m pip install -e ".[reports]"') from error
    inputs = []
    for run_name in run_dirs:
        run = Path(run_name)
        cfg = yaml.safe_load((run/'resolved_config.yaml').read_text(encoding='utf-8'))
        payload = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
        metric_path = run/('validation.json' if split == 'validation' else 'test.json')
        if not metric_path.is_file():
            raise ValueError(f'{split} evaluation is missing: {metric_path}')
        metrics = _json(metric_path)
        history = _json(run/'history.json')
        if not history:
            raise ValueError(f'Empty training history: {run}')
        for task in payload['mapping']:
            if task not in metrics or 'confusion_matrix' not in metrics[task]:
                raise ValueError(f'{split} metrics missing {task} confusion matrix: {run}')
        condition = _condition(cfg, payload)
        group = hashlib.sha256(json.dumps(condition, sort_keys=True).encode()).hexdigest()[:12]
        inputs.append((run, cfg, payload, metrics, history, group, condition))
    output.mkdir(parents=True)
    rows, groups, examples, missing_predictions = [], {}, [], []
    for index, (run, cfg, payload, metrics, history, group, condition) in enumerate(inputs, 1):
        tasks = payload['mapping']
        for task in tasks:
            value = metrics[task]
            rows.append({'split':split, 'run_dir':str(run), 'experiment':cfg['experiment']['id'],
                         'backbone':cfg['model']['backbone'], 'seed':cfg['training']['seed'],
                         'condition_id':group, 'figure_prefix':f'run-{index:03d}', 'task':task, 'accuracy':value['accuracy'],
                         'macro_f1':value['macro_f1'], 'weighted_f1':value.get('weighted_f1'),
                         'best_epoch':payload['epoch']})
        groups.setdefault(group, {'condition':condition, 'runs':[]})['runs'].append(
            {'seed':cfg['training']['seed'], 'metrics':metrics, 'run_dir':str(run)})

        predictions_path = run/f'{split}_predictions.csv'
        if predictions_path.exists():
            with predictions_path.open(encoding='utf-8', newline='') as stream:
                for prediction in csv.DictReader(stream):
                    for task in tasks:
                        truth, predicted = prediction[f'true_{task}'], prediction[f'pred_{task}']
                        examples.append({'split':split, 'run_dir':str(run), 'image_id':prediction['image_id'],
                                         'image_path':prediction['image_path'], 'task':task,
                                         'true_label':truth, 'predicted_label':predicted,
                                         'confidence':prediction.get(f'confidence_{task}', ''),
                                         'correct':truth == predicted})
        else:
            missing_predictions.append(str(predictions_path))

        epochs = [item['epoch'] for item in history]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for phase in ('train', 'val'):
            axes[0].plot(epochs, [item[phase]['loss'] for item in history], label=phase)
        axes[0].set(title='Loss', xlabel='Epoch', ylabel='Joint loss')
        axes[0].legend()
        for task in tasks:
            axes[1].plot(epochs, [item['val'][task]['macro_f1'] for item in history], label=task)
        axes[1].set(title='Validation macro F1', xlabel='Epoch', ylabel='Macro F1', ylim=(0, 1))
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output/f'run-{index:03d}-curves.png', dpi=120)
        plt.close(fig)
        for task, mapping in tasks.items():
            labels = [name for name, _ in sorted(mapping.items(), key=lambda pair:pair[1])]
            matrix = metrics[task]['confusion_matrix']
            fig, ax = plt.subplots(figsize=(max(4, len(labels)*.7), max(4, len(labels)*.7)))
            display = ax.imshow(matrix, cmap='Blues')
            fig.colorbar(display, ax=ax)
            ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
                   xlabel='Predicted', ylabel='True', title=f'{split}: {task}')
            ax.tick_params(axis='x', labelrotation=45)
            for row_index, row in enumerate(matrix):
                for column_index, count in enumerate(row):
                    ax.text(column_index, row_index, str(count), ha='center', va='center')
            fig.tight_layout()
            fig.savefig(output/f'run-{index:03d}-{task}-confusion.png', dpi=120)
            plt.close(fig)
    _write_csv(output/'results.csv', rows, list(rows[0]))
    _write_csv(output/'examples.csv', examples,
               ['split','run_dir','image_id','image_path','task','true_label','predicted_label','confidence','correct'])
    repeated, skipped = [], []
    for group, values in groups.items():
        runs = values['runs']
        seeds = [run['seed'] for run in runs]
        if len(runs) < 2:
            continue
        if len(set(seeds)) != len(seeds):
            skipped.append({'condition_id':group, 'reason':'Duplicate seeds; independent-seed summary omitted', 'seeds':seeds})
            continue
        for task in values['condition']['mapping']:
            scores = [run['metrics'][task]['macro_f1'] for run in runs]
            repeated.append({'split':split, 'condition_id':group, 'task':task, 'n_seeds':len(scores),
                             'seeds':','.join(str(seed) for seed in seeds),
                             'macro_f1_mean':statistics.mean(scores), 'macro_f1_sample_std':statistics.stdev(scores)})
    fields = ['split','condition_id','task','n_seeds','seeds','macro_f1_mean','macro_f1_sample_std']
    _write_csv(output/'repeated-seeds.csv', repeated, fields)
    result = {'split':split, 'runs':len(inputs), 'vary':list(vary), 'rows':rows,
              'repeated_seed_summary':repeated, 'skipped_summaries':skipped,
              'missing_predictions':missing_predictions,
              'note':'Only recorded results are shown; validation and test are separate reports. '
                     'Comparisons require matching non-varying controls and completed post-P0 status. '
                     'Seed SD is sample standard deviation; no summary is generated for a single seed.'}
    (output/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result
