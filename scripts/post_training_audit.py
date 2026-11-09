"""Audit held-out predictions and run validation-only robustness after the 44 runs."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

from furniture.data import read_rows
from furniture.pipeline import _process_alive
from furniture.robustness import compare_robustness, evaluate_robustness, make_manifest
from furniture.suite import _suite_lock
from scripts import orchestrate_model_release as workflow
from scripts.prediction_audit import audit_closed_set, audit_confusion_csv, audit_external, audit_external_exclusions
from scripts.release_evidence import bundle, sha, validate_bundle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'artifacts/model-only-release'
DATA = ROOT/'data/furniture-v2-conservative'


def status(stage, **extra):
    path = OUTPUT/'post-training-status.json'
    temporary = path.with_name(f'.post-status-{os.getpid()}.tmp')
    temporary.write_text(json.dumps({'stage':stage, **extra},indent=2),encoding='utf-8')
    os.replace(temporary,path)


def wait_for_workflow(pid, retry_demo=False):
    retried = False
    while True:
        current = workflow.read(OUTPUT/'workflow-status.json')
        selection_path = OUTPUT/'selection-frozen.json'
        demo_status = OUTPUT/'selected-room-demo-status.json'
        if selection_path.is_file():
            selected = workflow.read(selection_path)['selected']
            previous = workflow.read(demo_status) if demo_status.exists() else None
            if previous and previous.get('checkpoint_sha256') != selected['checkpoint_sha256']:
                raise ValueError('Room demonstration status belongs to a different selected checkpoint')
            if previous is None or (retry_demo and not retried and previous['status'] == 'failed'):
                retried = True
                try:
                    selected_room_demo(selected)
                    record = {'status':'complete','checkpoint_sha256':selected['checkpoint_sha256'],
                              'note':'Demonstration only; not used for model selection or thresholds.'}
                except Exception as error:
                    record = {'status':'failed','checkpoint_sha256':selected['checkpoint_sha256'],
                              'error':f'{type(error).__name__}: {error}'}
                history = OUTPUT/'selected-room-demo-events'
                workflow.write_once(history/f'{uuid.uuid4().hex}.json',record)
                temporary = OUTPUT/f'.demo-status-{uuid.uuid4().hex}.tmp'
                temporary.write_text(json.dumps(record,indent=2),encoding='utf-8')
                os.replace(temporary,demo_status)
        if current['stage'] == 'complete':
            return current
        if current['stage'] == 'stopped' or not _process_alive(pid):
            raise ValueError(f'Training workflow stopped before completion: {current}')
        status('waiting_for_training', workflow_stage=current['stage'])
        time.sleep(30)


def selected_room_demo(selection):
    suite = workflow.completed_runs(ROOT/'artifacts/runs/suites/furniture-v2-model-only-controlled.json',24)
    if workflow.select_controlled(suite)['selected'] != selection:
        raise ValueError('Frozen selection does not match validation rule')
    checkpoint = Path(selection['run_dir'])/'best.pt'
    if sha(checkpoint) != selection['checkpoint_sha256']:
        raise ValueError('Selected checkpoint hash differs')
    detector = ROOT/'artifacts/detector-baselines/yolov8n-coco-v8.3.0/yolov8n.pt'
    image = ROOT/'artifacts/room-inference-demo/source.jpg'
    config = ROOT/'configs/inference/room-yolov8n-coco-baseline.yaml'
    python = ROOT/('.pip-temp/research-clean-env/Scripts/python.exe' if os.name == 'nt' else '.pip-temp/research-clean-env/bin/python')
    if not all(path.is_file() for path in (detector,image,config,python)):
        raise FileNotFoundError('Detector baseline, demo image, config or separate detector environment is missing')
    context = {'selection_sha256':sha(OUTPUT/'selection-frozen.json'),'checkpoint_sha256':sha(checkpoint),
               'detector_sha256':sha(detector),'image_sha256':sha(image),'config_sha256':sha(config)}
    def build(path):
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'src'),str(ROOT)])
        env['OMP_NUM_THREADS'] = '1'
        env['YOLO_CONFIG_DIR'] = str(ROOT/'.pip-temp')
        subprocess.run([str(python),'-m','scripts.detect_then_classify','--image',str(image),
                        '--image-id','pexels-6312360','--checkpoint',str(checkpoint),'--config',str(config),
                        '--output',str(path/'result'),'--device','cpu','--save-crops'],
                       cwd=ROOT,env=env,check=True,capture_output=True,text=True)
        result = workflow.read(path/'result/result.json')
        if result['provenance']['classifier_sha256'] != context['checkpoint_sha256']:
            raise ValueError('Room demonstration classifier does not match frozen selection')
    bundle(OUTPUT/'selected-room-demo',context,['result/result.json','result/objects.csv','result/annotated.png'],build)


def verify_heldout(selection, controlled, ablation, repeated):
    root, checkpoint, snapshot, context, comparison, protocol, external = workflow.artifact_contexts(
        selection, controlled, ablation, repeated)
    validate_bundle(OUTPUT/'validation-report',comparison,['validation-comparisons.csv','repeated-seeds.csv','controlled-contrasts.csv'])
    synthetic = OUTPUT/'synthetic-test'
    validate_bundle(synthetic,context,['test.json','test_predictions.csv','test_type_confusion.csv','test_style_confusion.csv'])
    external_dir = OUTPUT/'external-reference'
    validate_bundle(external_dir,external,['result/metrics.json','result/predictions.csv','result/missing_or_unseen.csv'])
    synthetic_metrics = workflow.read(synthetic/'test.json')
    external_metrics = workflow.read(external_dir/'result/metrics.json')
    if external_metrics.get('checkpoint_sha256') != selection['checkpoint_sha256'] or external_metrics.get('protocol_sha256') != sha(protocol):
        raise ValueError('External results refer to a different checkpoint or protocol')
    results = {'synthetic':audit_closed_set(root/'test.csv',synthetic/'test_predictions.csv',synthetic_metrics,snapshot['class_mapping']),
               'external':audit_external(root/'real_source_test.csv',external_dir/'result/predictions.csv',external_metrics,snapshot['class_mapping'])}
    for task in ('type','style'):
        audit_confusion_csv(synthetic/f'test_{task}_confusion.csv',snapshot['class_mapping'][task],synthetic_metrics[task])
    audit_external_exclusions(external_dir/'result/predictions.csv',external_dir/'result/missing_or_unseen.csv')
    audit_context = {**context,'synthetic_evidence_sha256':sha(synthetic/'evidence.json'),
                     'external_evidence_sha256':sha(external_dir/'evidence.json')}
    bundle(OUTPUT/'heldout-prediction-audit',audit_context,['audit.json'],
           lambda path:(path/'audit.json').write_text(json.dumps(results,indent=2),encoding='utf-8'))
    return snapshot


def robustness(selection, controlled, snapshot):
    items = {workflow.condition_id(x['config']):x for x in controlled
             if x['config']['model']['backbone'] == selection['backbone'] and x['config']['training']['seed'] == 42}
    conditions = ('C3','C5','C6','C4','C7','C8')
    if not set(conditions) <= set(items):
        raise ValueError('Required controlled robustness checkpoints are missing')
    image_sizes = {x['config']['model']['image_size'] for x in items.values()}
    if len(image_sizes) != 1:
        raise ValueError('Robustness comparison requires one shared image size')
    size = image_sizes.pop()
    val_hash = sha(DATA/'val.csv')
    manifest = make_manifest(read_rows(DATA/'val.csv'),'val',val_hash,size,seed=42)
    manifest_context = {'split_sha256':val_hash,'size':size,'seed':42}
    manifest_dir = bundle(OUTPUT/'robustness-manifest',manifest_context,['manifest.json'],
                          lambda path:(path/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2),encoding='utf-8'))
    paths, source_hashes = {}, {}
    for condition in conditions:
        item = items[condition]
        run = Path(item['attempts'][-1]['run_dir'])
        checkpoint = run/'best.pt'
        checkpoint_hash = sha(checkpoint)
        context = {'condition':condition,'checkpoint_sha256':checkpoint_hash, 'split_hashes':snapshot['split_hashes'],
                   'mapping':snapshot['class_mapping'], 'manifest_sha256':sha(manifest_dir/'manifest.json'),
                   'config_sha256':item['config_sha256']}
        status('robustness',condition=condition)
        folder = bundle(OUTPUT/'robustness'/condition,context,
                        ['result/robustness.json','result/occlusion_manifest.json','result/predictions_clean.csv',
                         'result/predictions_light.csv','result/predictions_strong.csv'],
                        lambda path:evaluate_robustness(DATA,checkpoint,path/'result',split='val',seed=42,manifest=manifest_dir/'manifest.json'))
        result = workflow.read(folder/'result/robustness.json')
        if result['checkpoint_sha256'] != checkpoint_hash or result['split_hashes'] != snapshot['split_hashes'] or result['mapping'] != snapshot['class_mapping']:
            raise ValueError('Robustness result differs from checkpoint or snapshot')
        for level in ('clean','light','strong'):
            audit_closed_set(DATA/'val.csv',folder/f'result/predictions_{level}.csv',result['metrics'][level],snapshot['class_mapping'])
        for level in ('light','strong'):
            for task in ('type','style'):
                for metric in ('accuracy','macro_f1'):
                    expected = result['metrics']['clean'][task][metric]-result['metrics'][level][task][metric]
                    if not math.isclose(result['absolute_drop'][level][task][metric],expected,abs_tol=1e-12):
                        raise ValueError('Robustness drop does not match metrics')
        paths[condition] = folder/'result'
        source_hashes[condition] = sha(folder/'evidence.json')
    context = {'selected_backbone':selection['backbone'],'manifest_sha256':sha(manifest_dir/'manifest.json'),
               'result_evidence_sha256':source_hashes}
    def build(path):
        compare_robustness([paths[k] for k in ('C3','C5','C6')],path/'easy-contrasts.csv',vary=['occlusion'])
        compare_robustness([paths[k] for k in ('C4','C7','C8')],path/'hard-contrasts.csv',vary=['occlusion'])
        fields = ['condition','task','level','accuracy','macro_f1','accuracy_drop_from_clean','macro_f1_drop_from_clean','checkpoint_sha256']
        with (path/'metrics.csv').open('x',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            for condition in conditions:
                result = workflow.read(paths[condition]/'robustness.json')
                for task in ('type','style'):
                    clean = result['metrics']['clean'][task]
                    for level in ('clean','light','strong'):
                        values = result['metrics'][level][task]
                        writer.writerow({'condition':condition,'task':task,'level':level,'accuracy':values['accuracy'],
                            'macro_f1':values['macro_f1'],'accuracy_drop_from_clean':clean['accuracy']-values['accuracy'],
                            'macro_f1_drop_from_clean':clean['macro_f1']-values['macro_f1'],
                            'checkpoint_sha256':result['checkpoint_sha256']})
        (path/'protocol.json').write_text(json.dumps({'split':'validation','seed':42,'conditions':list(conditions),
            'selection_use':'not used for checkpoint selection','note':'Image-area occlusion is not object-area occlusion.'},indent=2),encoding='utf-8')
    bundle(OUTPUT/'robustness-report',context,['easy-contrasts.csv','hard-contrasts.csv','metrics.csv','protocol.json'],build)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workflow-pid',type=int,required=True)
    parser.add_argument('--retry-demo',action='store_true',help='Retry a failed selected-checkpoint room demonstration')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with _suite_lock(OUTPUT/'post-training.lock'):
        try:
            wait_for_workflow(args.workflow_pid,args.retry_demo)
            selection = workflow.read(OUTPUT/'selection-frozen.json')['selected']
            suites = ROOT/'artifacts/runs/suites'
            controlled = workflow.completed_runs(suites/'furniture-v2-model-only-controlled.json',24)
            ablation = workflow.completed_runs(suites/'furniture-v2-model-only-ablation.json',2)
            repeated = workflow.completed_runs(suites/'furniture-v2-model-only-repeated.json',8)
            status('heldout_audit')
            snapshot = verify_heldout(selection,controlled,ablation,repeated)
            robustness(selection,controlled,snapshot)
            status('complete',selected_checkpoint_sha256=selection['checkpoint_sha256'])
        except BaseException as error:
            status('stopped',error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
