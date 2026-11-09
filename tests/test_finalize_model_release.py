import json
from pathlib import Path
import zipfile

import pytest

from scripts import finalize_model_release as finalizer


def setup_release(tmp_path, monkeypatch):
    output = tmp_path/'artifacts/model-only-release'
    output.mkdir(parents=True)
    readme, status = tmp_path/'README.md', tmp_path/'RELEASE_STATUS.md'
    readme.write_text('checkpoint ยังไม่ถูกเลือก', encoding='utf-8')
    status.write_text('pending', encoding='utf-8')
    checkpoint = tmp_path/'best.pt'
    checkpoint.write_bytes(b'actual checkpoint')
    selection = {'backbone':'resnet18','condition':'C3','checkpoint_sha256':finalizer.workflow.sha(checkpoint)}
    monkeypatch.setattr(finalizer, 'ROOT', tmp_path)
    monkeypatch.setattr(finalizer, 'OUTPUT', output)
    monkeypatch.setattr(finalizer, 'ZIP', tmp_path/'final.zip')
    monkeypatch.setattr(finalizer, 'final_evidence', lambda: (selection, checkpoint,
                        {'type':{'macro_f1':.7},'style':{'macro_f1':.6}}))
    def fake_package(path, checkpoint=None, document_overrides=None):
        with zipfile.ZipFile(path, 'x') as archive:
            archive.writestr('DELIVERY_SHA256.json', json.dumps({}))
            for name, document in document_overrides.items():
                archive.writestr(name, Path(document).read_bytes())
            archive.writestr('artifacts/model-only-release/selection-frozen.json', json.dumps({'selected':selection}))
        return {'zip':str(path),'files':1,'sha256':finalizer.workflow.sha(path)}
    monkeypatch.setattr(finalizer, 'package', fake_package)
    monkeypatch.setattr(finalizer, 'verify_archive', lambda path, expected: {'extracted_package_inference':'passed'})
    return readme, status, output


def test_final_delivery_records_checkpoint_and_preserves_previous_docs(tmp_path, monkeypatch):
    readme, status, output = setup_release(tmp_path, monkeypatch)
    finalizer.finalize(finalizer.workflow.sha(readme), finalizer.workflow.sha(status))
    assert 'ถูกเลือกแล้ว' in readme.read_text(encoding='utf-8')
    assert 'checkpoint' in status.read_text(encoding='utf-8').lower()
    attempts = list((output/'_attempts').iterdir())
    assert (attempts[0]/'before-status.md').read_text(encoding='utf-8') == 'pending'
    record = json.loads((output/'final-delivery.json').read_text(encoding='utf-8'))
    assert record['checkpoint_sha256'] == finalizer.workflow.sha(tmp_path/'best.pt')
    assert record['zip_crc_and_sha256'] == 'verified'


def test_finalizer_refuses_to_overwrite_user_document_changes(tmp_path, monkeypatch):
    readme, status, output = setup_release(tmp_path, monkeypatch)
    original_hash = finalizer.workflow.sha(readme)
    readme.write_text('user changed this', encoding='utf-8')
    with pytest.raises(ValueError, match='preserve user edits'):
        finalizer.finalize(original_hash, finalizer.workflow.sha(status))
    assert readme.read_text(encoding='utf-8') == 'user changed this'
    assert not (tmp_path/'final.zip').exists()


def test_failed_packaging_restores_prior_status(tmp_path, monkeypatch):
    readme, status, output = setup_release(tmp_path, monkeypatch)
    def fail_package(path, checkpoint=None, document_overrides=None):
        raise OSError('archive failed')
    monkeypatch.setattr(finalizer, 'package', fail_package)
    with pytest.raises(OSError, match='archive failed'):
        finalizer.finalize(finalizer.workflow.sha(readme), finalizer.workflow.sha(status))
    assert readme.read_text(encoding='utf-8') == 'checkpoint ยังไม่ถูกเลือก'
    assert status.read_text(encoding='utf-8') == 'pending'


def test_resume_after_publication_without_repacking(tmp_path, monkeypatch):
    readme, status, output = setup_release(tmp_path, monkeypatch)
    original_hashes = finalizer.workflow.sha(readme), finalizer.workflow.sha(status)
    original_write = finalizer.workflow.write_once
    monkeypatch.setattr(finalizer.workflow, 'write_once', lambda *args: (_ for _ in ()).throw(RuntimeError('crash')))
    with pytest.raises(RuntimeError, match='crash'):
        finalizer.finalize(*original_hashes)
    assert (tmp_path/'final.zip').is_file()
    monkeypatch.setattr(finalizer.workflow, 'write_once', original_write)
    monkeypatch.setattr(finalizer, 'package', lambda *args, **kwargs: pytest.fail('must reuse published ZIP'))
    finalizer.finalize(*original_hashes)
    assert (output/'final-delivery.json').is_file()


def test_archive_rejects_wrong_checkpoint_before_inference(tmp_path):
    archive = tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('DELIVERY_SHA256.json',json.dumps({'release/checkpoint/best.pt':'wrong'}))
    with pytest.raises(ValueError,match='checkpoint hash differs'):
        finalizer.verify_archive(archive,'expected')


def test_finalizer_refuses_before_prediction_audit_and_robustness(tmp_path, monkeypatch):
    output=tmp_path/'release'; output.mkdir()
    run=tmp_path/'run'; run.mkdir(); (run/'best.pt').write_bytes(b'checkpoint')
    selected={'run_dir':str(run),'backbone':'resnet18','checkpoint_sha256':finalizer.workflow.sha(run/'best.pt')}
    monkeypatch.setattr(finalizer,'ROOT',tmp_path)
    monkeypatch.setattr(finalizer,'OUTPUT',output)
    monkeypatch.setattr(finalizer,'validate_test_binding',lambda *args:None)
    monkeypatch.setattr(finalizer.workflow,'completed_runs',lambda *args:[{}])
    monkeypatch.setattr(finalizer.workflow,'source_contract',lambda *args:{})
    monkeypatch.setattr(finalizer.workflow,'select_controlled',lambda *args:{'selected':selected})
    monkeypatch.setattr(finalizer.workflow,'artifact_contexts',lambda *args:(tmp_path,run/'best.pt',{}, {}, {}, tmp_path/'protocol', {}))
    monkeypatch.setattr(finalizer,'validate_bundle',lambda *args:None)
    monkeypatch.setattr(finalizer.workflow,'sha',lambda path:'hash')
    selected['checkpoint_sha256']='hash'
    def read(path):
        name=Path(path).name
        if name=='selection-frozen.json': return {'selected':selected}
        if name=='metrics.json': return {'checkpoint_sha256':'hash','protocol_sha256':'hash'}
        if name=='post-training-status.json': return {'stage':'waiting_for_training'}
        raise AssertionError(path)
    monkeypatch.setattr(finalizer.workflow,'read',read)
    with pytest.raises(ValueError,match='audit and robustness workflow are incomplete'):
        finalizer.final_evidence()
