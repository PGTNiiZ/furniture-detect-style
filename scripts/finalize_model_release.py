"""Package final results only after all registered model-only evidence exists."""
import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
import yaml

from scripts import orchestrate_model_release as workflow
from scripts.package_research_delivery import package
from scripts.release_evidence import validate_bundle, validate_test_binding
from furniture.pipeline import _process_alive
from furniture.suite import _suite_lock


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'artifacts/model-only-release'
ZIP = ROOT/'artifacts/furniture-project2-classifier-research-complete.zip'


def final_evidence():
    validate_test_binding(ROOT, ROOT/'docs/evidence/crop-consistency-test-binding.json')
    selection = workflow.read(OUTPUT/'selection-frozen.json')['selected']
    checkpoint = Path(selection['run_dir'])/'best.pt'
    if not checkpoint.is_file() or workflow.sha(checkpoint) != selection['checkpoint_sha256']:
        raise ValueError('Selected checkpoint is missing or has changed')
    suites = {}
    for name, count in (('screening',10), ('model-only-controlled',24), ('model-only-ablation',2), ('model-only-repeated',8)):
        suites[name] = workflow.completed_runs(ROOT/f'artifacts/runs/suites/furniture-v2-{name}.json', count)
    source = workflow.source_contract(suites['screening'])
    for name in ('model-only-controlled', 'model-only-ablation', 'model-only-repeated'):
        workflow.source_contract(suites[name], source)
    if workflow.select_controlled(suites['model-only-controlled'])['selected'] != selection:
        raise ValueError('Frozen selection does not match the registered validation rule')
    root, _, snapshot, context, comparison, protocol, external = workflow.artifact_contexts(
        selection, suites['model-only-controlled'], suites['model-only-ablation'], suites['model-only-repeated'])
    validate_bundle(OUTPUT/'validation-report', comparison, ['validation-comparisons.csv','repeated-seeds.csv','controlled-contrasts.csv'])
    test = OUTPUT/'synthetic-test'
    validate_bundle(test, context, ['test.json','test_predictions.csv','test_type_confusion.csv','test_style_confusion.csv'])
    report_context = {**context, 'evaluation_evidence_sha256':workflow.sha(test/'evidence.json')}
    validate_bundle(OUTPUT/'synthetic-test-report', report_context,
                    ['report.json','errors.csv','type-confusion.csv','style-confusion.csv','type-confusion.png','style-confusion.png'])
    validate_bundle(OUTPUT/'external-reference', external, ['result/metrics.json','result/predictions.csv','result/missing_or_unseen.csv'])
    external_metrics = workflow.read(OUTPUT/'external-reference/result/metrics.json')
    if external_metrics.get('checkpoint_sha256') != selection['checkpoint_sha256'] or external_metrics.get('protocol_sha256') != workflow.sha(protocol):
        raise ValueError('External result checkpoint or protocol does not match selection')
    post = workflow.read(OUTPUT/'post-training-status.json')
    if post.get('stage') != 'complete' or post.get('selected_checkpoint_sha256') != selection['checkpoint_sha256']:
        raise ValueError('Held-out audit and robustness workflow are incomplete')
    synthetic = OUTPUT/'synthetic-test'
    audit_context = {**context,'synthetic_evidence_sha256':workflow.sha(synthetic/'evidence.json'),
                     'external_evidence_sha256':workflow.sha(OUTPUT/'external-reference/evidence.json')}
    validate_bundle(OUTPUT/'heldout-prediction-audit',audit_context,['audit.json'])
    controlled = {workflow.condition_id(x['config']):x for x in suites['model-only-controlled']
                  if x['config']['model']['backbone'] == selection['backbone'] and x['config']['training']['seed'] == 42}
    val_manifest = OUTPUT/'robustness-manifest'
    size = next(iter(controlled.values()))['config']['model']['image_size']
    validate_bundle(val_manifest,{'split_sha256':workflow.sha(root/'val.csv'),'size':size,'seed':42},['manifest.json'])
    source_hashes = {}
    for condition in ('C3','C5','C6','C4','C7','C8'):
        item = controlled[condition]
        run = Path(item['attempts'][-1]['run_dir'])
        bundle_context = {'condition':condition,'checkpoint_sha256':workflow.sha(run/'best.pt'),
                          'split_hashes':snapshot['split_hashes'],'mapping':snapshot['class_mapping'],
                          'manifest_sha256':workflow.sha(val_manifest/'manifest.json'),'config_sha256':item['config_sha256']}
        folder = OUTPUT/'robustness'/condition
        validate_bundle(folder,bundle_context,['result/robustness.json','result/occlusion_manifest.json',
                        'result/predictions_clean.csv','result/predictions_light.csv','result/predictions_strong.csv'])
        source_hashes[condition] = workflow.sha(folder/'evidence.json')
    robustness_context = {'selected_backbone':selection['backbone'],'manifest_sha256':workflow.sha(val_manifest/'manifest.json'),
                          'result_evidence_sha256':source_hashes}
    validate_bundle(OUTPUT/'robustness-report',robustness_context,['easy-contrasts.csv','hard-contrasts.csv','metrics.csv','protocol.json'])
    return selection, checkpoint, workflow.read(test/'test.json')


def verify_archive(path, expected_checkpoint):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC check failed')
        hashes = json.loads(archive.read('DELIVERY_SHA256.json'))
        if hashes.get('release/checkpoint/best.pt') != expected_checkpoint:
            raise ValueError('Packaged checkpoint hash differs from selection')
        for name, expected in hashes.items():
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda:stream.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f'ZIP member hash mismatch: {name}')
        # Keep extracted paths short enough for standard Windows installations.
        temp_root = (Path(os.environ['LOCALAPPDATA'])/'Temp').resolve() if os.name == 'nt' and os.environ.get('LOCALAPPDATA') else Path(tempfile.gettempdir()).resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='fmr-', dir=temp_root) as temp:
            extracted = Path(temp).resolve()
            if not extracted.is_relative_to(temp_root):
                raise ValueError('Extraction path is outside the intended temporary root')
            archive.extractall(extracted)
            with (ROOT/'data/furniture-v2-conservative/train.csv').open(encoding='utf-8', newline='') as stream:
                row = next(csv.DictReader(stream))
            image = (ROOT/'data/furniture-v2-conservative'/row['image_path']).resolve()
            if not image.is_relative_to((ROOT/'data/furniture-v2-conservative').resolve()):
                raise ValueError('Training sample path escapes dataset root')
            env = os.environ.copy()
            env['PYTHONPATH'] = os.pathsep.join([str(extracted/'src'), str(extracted)])
            env['PYTHONNOUSERSITE'] = '1'
            env['OMP_NUM_THREADS'] = '1'
            location = subprocess.check_output([sys.executable, '-c', 'import furniture; print(furniture.__file__)'],
                                               cwd=extracted, env=env, text=True).strip()
            if not Path(location).resolve().is_relative_to((extracted/'src').resolve()):
                raise ValueError('Inference did not import the extracted package')
            output = extracted/'verification-prediction.json'
            subprocess.run([sys.executable, '-m', 'scripts.model_inference', '--checkpoint',
                            str(extracted/'release/checkpoint/best.pt'), '--image', str(image),
                            '--image-id', row['image_id'], '--output', str(output), '--device', 'cpu'],
                           cwd=extracted, env=env, check=True, capture_output=True, text=True)
            result = workflow.read(output)
            if len(result) != 1 or result[0].get('status') != 'ok' or set(('type','style')) - set(result[0]):
                raise ValueError('Extracted package inference failed')
            room_result = {'status':'not_checked','reason':'Detector weights, demo image or isolated detector environment unavailable'}
            detector = ROOT/'artifacts/detector-baselines/yolov8n-coco-v8.3.0/yolov8n.pt'
            room_image = ROOT/'artifacts/room-inference-demo/source.jpg'
            detector_python = ROOT/('.pip-temp/research-clean-env/Scripts/python.exe' if os.name == 'nt' else '.pip-temp/research-clean-env/bin/python')
            if detector.is_file() and room_image.is_file() and detector_python.is_file():
                detector_config = yaml.safe_load((extracted/'configs/inference/room-yolov8n-coco-baseline.yaml').read_text(encoding='utf-8'))
                if workflow.sha(detector) != detector_config['detector']['sha256']:
                    raise ValueError('Detector baseline hash differs from the pinned config')
                env['YOLO_CONFIG_DIR'] = str(extracted/'ultralytics-settings')
                detector_location = subprocess.check_output([str(detector_python), '-c', 'import furniture; print(furniture.__file__)'],
                                                            cwd=extracted,env=env,text=True).strip()
                if not Path(detector_location).resolve().is_relative_to((extracted/'src').resolve()):
                    raise ValueError('Room inference did not import the extracted package')
                room_output = extracted/'room-verification'
                subprocess.run([str(detector_python), '-m', 'scripts.detect_then_classify', '--image',str(room_image),
                    '--checkpoint',str(extracted/'release/checkpoint/best.pt'), '--detector-weights',str(detector),
                    '--config',str(extracted/'configs/inference/room-yolov8n-coco-baseline.yaml'),
                    '--output',str(room_output),'--device','cpu'],cwd=extracted,env=env,check=True,capture_output=True,text=True)
                room = workflow.read(room_output/'result.json')
                if (room['provenance']['classifier_sha256'] != expected_checkpoint
                        or room['provenance']['detector_sha256'] != workflow.sha(detector)
                        or not any(item['status']=='ok' for item in room['objects'])):
                    raise ValueError('Extracted room inference did not classify a detector crop')
                room_result = {'status':'passed','detector_sha256':workflow.sha(detector),
                               'classified_objects':sum(item['status']=='ok' for item in room['objects']),
                               'environment':'existing separate research-clean-env; no new installation'}
    return {'checkpoint_sha256':expected_checkpoint, 'sample_id':row['image_id'],
            'split':'train', 'device':'cpu', 'extracted_package_inference':'passed','room_inference':room_result}


def finalize(expected_readme_sha, expected_status_sha):
    selection, checkpoint, metrics = final_evidence()
    readme, release = ROOT/'README.md', ROOT/'RELEASE_STATUS.md'
    old_readme, old_status = readme.read_text(encoding='utf-8'), release.read_text(encoding='utf-8')
    lines = ['# สถานะ model-only release', '',
             'Classifier research complete: การทดลองตาม protocol เสร็จและเลือก checkpoint ด้วย validation ก่อนประเมิน test.',
             'Room inference: มีตัวอย่าง detector baseline จริง; สถานะตรวจแพ็กสุดท้ายอยู่ใน final-delivery.json.',
             'Room end-to-end evaluation: ยังไม่ครบจนมี annotations อิสระและรายงานประเมินจริง.', '',
             f"- Backbone: `{selection['backbone']}`, condition: `{selection['condition']}`",
             f'- Checkpoint บนเครื่อง: `{checkpoint}`', f"- SHA-256: `{selection['checkpoint_sha256']}`",
             '- ZIP มี selected checkpoint ที่ `release/checkpoint/best.pt`; ตรวจ SHA-256 ก่อนใช้',
             '- Selection และ split hashes: `artifacts/model-only-release/selection-frozen.json`',
             '- Controlled, single-task และ repeated validation: `artifacts/model-only-release/validation-report/`',
             '- Robustness validation clean/light/strong: `artifacts/model-only-release/robustness-report/`',
             '- Synthetic test, confusion matrices และ error examples: `artifacts/model-only-release/synthetic-test-report/`',
             '- Real-source test: `artifacts/model-only-release/external-reference/result/metrics.json` แยกจาก synthetic test',
             '- Source annotations ไม่ใช่ expert ground truth; confidence เป็น uncalibrated softmax',
             '- 494 near-duplicate candidates ยังไม่มี human decisions; v2 ใช้ conservative quarantine ตาม protocol',
             '- Detector เป็น optional preprocessing แยกจาก classifier และไม่มีการฝึก detector ใหม่',
             '- Bonn/IKEA/Target เป็น future work ไม่ได้ผสมใน suite นี้', '',
             f"Synthetic test type macro-F1: {metrics['type']['macro_f1']:.6f}; style macro-F1: {metrics['style']['macro_f1']:.6f}.",
             'ดู metrics, counts, confusion matrices และตัวอย่างผิดในไฟล์รายงาน ไม่รวมคะแนนสอง test เข้าด้วยกัน.', '',
             f"บันทึกเมื่อ {datetime.now(timezone.utc).isoformat()} หลังตรวจ completion evidence และ checksum", '']
    if ZIP.exists():
        inference = verify_archive(ZIP, selection['checkpoint_sha256'])
        with zipfile.ZipFile(ZIP) as archive:
            updated_readme = archive.read('README.md').decode('utf-8').replace('\r\n', '\n')
            updated_status = archive.read('RELEASE_STATUS.md').decode('utf-8').replace('\r\n', '\n')
            archived = json.loads(archive.read('artifacts/model-only-release/selection-frozen.json'))['selected']
            if archived != selection:
                raise ValueError('Published ZIP selection differs from current frozen selection')
            file_count = len(archive.namelist())
        result = {'zip':str(ZIP), 'files':file_count, 'sha256':workflow.sha(ZIP)}
    else:
        if workflow.sha(readme) != expected_readme_sha or workflow.sha(release) != expected_status_sha:
            raise ValueError('README or RELEASE_STATUS changed during experiments; preserve user edits and finalize manually')
        if 'ยังไม่ถูกเลือก' not in old_readme:
            raise ValueError('Expected README status marker is missing')
        updated_readme = old_readme.replace('ยังไม่ถูกเลือก', 'ถูกเลือกแล้ว ใช้ release/checkpoint/best.pt และตรวจ SHA-256 ใน selection-frozen.json', 1)
        updated_readme = updated_readme.replace('ข้อมูลภาพเต็มและ checkpoints อาจไม่อยู่ใน ZIP',
                                               'ภาพ dataset เต็มไม่อยู่ใน ZIP; selected checkpoint อยู่ที่ release/checkpoint/best.pt')
        updated_status = '\n'.join(lines)
        attempt = OUTPUT/'_attempts'/f'f-{uuid.uuid4().hex[:12]}'
        attempt.mkdir(parents=True)
        (attempt/'README.md').write_text(updated_readme, encoding='utf-8')
        (attempt/'RELEASE_STATUS.md').write_text(updated_status, encoding='utf-8')
        (attempt/'before-readme.md').write_text(old_readme, encoding='utf-8')
        (attempt/'before-status.md').write_text(old_status, encoding='utf-8')
        candidate = attempt/'candidate.zip'
        result = package(candidate, checkpoint=checkpoint, document_overrides={
            'README.md':attempt/'README.md', 'RELEASE_STATUS.md':attempt/'RELEASE_STATUS.md'})
        inference = verify_archive(candidate, selection['checkpoint_sha256'])
        if workflow.sha(readme) != expected_readme_sha or workflow.sha(release) != expected_status_sha:
            raise ValueError('README or RELEASE_STATUS changed during packaging; preserve user edits')
        os.rename(candidate, ZIP)
        result['zip'] = str(ZIP)
    for path, expected, current, updated in ((readme, expected_readme_sha, old_readme, updated_readme),
                                              (release, expected_status_sha, old_status, updated_status)):
        if current != updated:
            if workflow.sha(path) != expected:
                raise ValueError('README or RELEASE_STATUS changed during experiments; preserve user edits and finalize manually')
            temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
            temp.write_text(updated, encoding='utf-8')
            os.replace(temp, path)
    delivery = {**result, 'selected_checkpoint':str(checkpoint), 'checkpoint_sha256':selection['checkpoint_sha256'],
                'zip_crc_and_sha256':'verified', 'unpacked_inference':inference}
    if (OUTPUT/'final-delivery.json').exists():
        if workflow.read(OUTPUT/'final-delivery.json') != delivery:
            raise ValueError('Existing final-delivery record differs')
    else:
        workflow.write_once(OUTPUT/'final-delivery.json', delivery)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workflow-pid', type=int, required=True)
    parser.add_argument('--post-pid', type=int, required=True)
    parser.add_argument('--readme-sha256', required=True)
    parser.add_argument('--status-sha256', required=True)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with _suite_lock(OUTPUT/'finalizer.lock'):
        while True:
            state = workflow.read(OUTPUT/'workflow-status.json')
            if state['stage'] == 'complete':
                break
            if state['stage'] == 'stopped' or not _process_alive(args.workflow_pid):
                raise ValueError(f'Workflow stopped before finalization: {state}')
            time.sleep(30)
        while True:
            state = workflow.read(OUTPUT/'post-training-status.json')
            if state['stage'] == 'complete':
                break
            if state['stage'] == 'stopped' or not _process_alive(args.post_pid):
                raise ValueError(f'Prediction audit or robustness stopped before finalization: {state}')
            time.sleep(30)
        print(json.dumps(finalize(args.readme_sha256, args.status_sha256), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
