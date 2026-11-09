import hashlib
import json
import copy
import shutil
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from furniture.data import validate_splits
from furniture.config import load_config, digest, training_settings
from furniture.model import AttributeModel
from furniture.robustness import apply_holes, compare_robustness, evaluate_robustness, make_manifest, training_evidence
from furniture.smoke import make_smoke


def mock_verified_run(root, cfg):
    checkpoint = root/'best.pt'
    checkpoint.write_bytes(b'test fixture')
    (root/'status.json').write_text(json.dumps({'status':'completed', 'device':'cpu', 'augmentation_protocol':'post-p0-fix-v1'}))
    (root/'resolved_config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    source = root/'source'
    source.mkdir()
    (source/'model.py').write_text('# source fixture', encoding='utf-8')
    source_hash = hashlib.sha256((source/'model.py').read_bytes()).hexdigest()
    (root/'provenance.json').write_text(json.dumps({'config_sha256':digest(cfg),'code_sha256':{'model.py':source_hash}}))
    (root/'checkpoint.sha256').write_text(hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    (root/'summary.json').write_text(json.dumps({'best_epoch':1}))
    return checkpoint


def test_manifest_deterministic_per_sample_and_all_occluded():
    rows = [{'image_id':str(i), 'pixel_hash':str(i)} for i in range(8)]
    first = make_manifest(rows, 'test', 'hash', 64, 42)
    assert first == make_manifest(list(reversed(rows)), 'test', 'hash', 64, 42)
    assert first != make_manifest(rows, 'test', 'hash', 64, 43)
    source = Image.new('RGB', (64, 64), 'black')
    for sample in first['samples']:
        for level in ('light', 'strong'):
            rectangles = sample['variants'][level]
            assert rectangles
            assert apply_holes(source, rectangles, first['fill_rgb']).tobytes() != source.tobytes()


def test_robustness_reuses_manifest_and_checks_split(tmp_path, monkeypatch):
    import furniture.robustness as robustness
    torch.set_num_threads(1)
    root = tmp_path/'data'
    make_smoke(root)
    splits, mapping = validate_splits(root)
    hashes = {name:hashlib.sha256((root/f'{name}.csv').read_bytes()).hexdigest() for name in splits}
    cfg = load_config(Path(__file__).resolve().parents[1]/'configs/project_smoke.yaml')
    checkpoint = mock_verified_run(tmp_path, cfg)

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            score = x.mean(dim=(1, 2, 3))
            return {key:torch.stack([score, -score], dim=1) for key in mapping}

    monkeypatch.setattr(robustness, 'load_model', lambda *_: (FakeModel(),
                        {'mapping':mapping, 'config':training_settings(cfg), 'split_hashes':hashes, 'epoch':1}))
    first = evaluate_robustness(root, checkpoint, tmp_path/'one', device='cpu')
    assert first['split'] == 'val' and first['training_evidence']['evidence_version'] == 3
    manifest = tmp_path/'one/occlusion_manifest.json'
    second = evaluate_robustness(root, checkpoint, tmp_path/'two', device='cpu', manifest=manifest)
    assert first['manifest_sha256'] == second['manifest_sha256']
    assert first['metrics'] == second['metrics']
    assert set(first['metrics']) == {'clean', 'light', 'strong'}
    assert set(first['metrics']['clean']) == {'type', 'style'}
    assert (tmp_path/'one/predictions_strong.csv').exists()
    comparison = compare_robustness([tmp_path/'one', tmp_path/'two'], tmp_path/'comparison.csv')
    assert len(comparison['rows']) == 4
    assert comparison['split'] == 'val'
    single_mapping = {'type':mapping['type']}
    monkeypatch.setattr(robustness, 'load_model', lambda *_: (FakeModel(),
                        {'mapping':single_mapping, 'config':training_settings(cfg), 'split_hashes':hashes, 'epoch':1}))
    single = evaluate_robustness(root, checkpoint, tmp_path/'single', device='cpu', manifest=manifest)
    assert list(single['metrics']['clean']) == ['type']
    with pytest.raises(ValueError, match='mapping'):
        compare_robustness([tmp_path/'one', tmp_path/'single'], tmp_path/'invalid.csv')
    unverified = tmp_path/'unverified'
    unverified.mkdir()
    (unverified/'fake.pt').write_bytes(b'test fixture')
    evaluate_robustness(root, unverified/'fake.pt', tmp_path/'old', device='cpu', manifest=manifest)
    with pytest.raises(ValueError, match='post-P0'):
        compare_robustness([tmp_path/'one', tmp_path/'old'], tmp_path/'old-comparison.csv')
    with pytest.raises(ValueError, match='manifest differs'):
        evaluate_robustness(root, checkpoint, tmp_path/'other', device='cpu', manifest=manifest, seed=43)
    values = json.loads(manifest.read_text(encoding='utf-8'))
    values['samples'][0]['variants']['strong'] = []
    altered = tmp_path/'altered.json'
    altered.write_text(json.dumps(values), encoding='utf-8')
    with pytest.raises(ValueError, match='manifest differs'):
        evaluate_robustness(root, checkpoint, tmp_path/'other', device='cpu', manifest=altered)
    (root/'test.csv').write_text((root/'test.csv').read_text(encoding='utf-8')+'\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Split manifests differ'):
        evaluate_robustness(root, checkpoint, tmp_path/'changed', device='cpu')


def test_real_checkpoint_robustness_cpu(tmp_path):
    torch.set_num_threads(1)
    root = tmp_path/'data'
    make_smoke(root)
    splits, mapping = validate_splits(root)
    model = AttributeModel(mapping, pretrained=False)
    checkpoint = tmp_path/'random-weights.pt'
    hashes = {name:hashlib.sha256((root/f'{name}.csv').read_bytes()).hexdigest() for name in splits}
    torch.save({'state_dict':model.state_dict(), 'mapping':mapping, 'config':{'image_size':64},
                'split_hashes':hashes}, checkpoint)
    result = evaluate_robustness(root, checkpoint, tmp_path/'result', device='cpu')
    assert set(result['metrics']) == {'clean', 'light', 'strong'}
    assert set(result['metrics']['clean']) == {'type', 'style'}


def test_explicit_axes_allow_cross_backbone_but_guard_other_controls(tmp_path, monkeypatch):
    import furniture.robustness as robustness
    torch.set_num_threads(1)
    root = tmp_path/'data'
    make_smoke(root)
    splits, mapping = validate_splits(root)
    hashes = {name:hashlib.sha256((root/f'{name}.csv').read_bytes()).hexdigest() for name in splits}
    cfg = load_config(Path(__file__).resolve().parents[1]/'configs/project_smoke.yaml')
    checkpoint = mock_verified_run(tmp_path, cfg)
    class FakeModel(torch.nn.Module):
        def forward(self, x):
            score=x.mean(dim=(1,2,3))
            return {key:torch.stack([score,-score],dim=1) for key in mapping}
    monkeypatch.setattr(robustness, 'load_model', lambda *_:(FakeModel(),
                        {'mapping':mapping, 'config':training_settings(cfg), 'split_hashes':hashes, 'epoch':1}))
    evaluate_robustness(root,checkpoint,tmp_path/'base',device='cpu')
    original = json.loads((tmp_path/'base'/'robustness.json').read_text())
    assert original['training_evidence']['controls']['model']['backbone'] == 'resnet18'

    def variant(name, changes):
        destination=tmp_path/name
        shutil.copytree(tmp_path/'base',destination)
        value=copy.deepcopy(original)
        changes(value['training_evidence']['controls'], value)
        evidence=value['training_evidence']
        for key in ('dataset','model','training','augmentation'):
            evidence['resolved_config'][key]=copy.deepcopy(evidence['controls'][key])
        evidence['config_sha256']=digest(evidence['resolved_config'])
        (destination/'robustness.json').write_text(json.dumps(value),encoding='utf-8')
        return destination

    def different_backbone(controls, _):
        controls['model'].update(backbone='resnet50', weights='IMAGENET1K_V2')
    other = variant('other',different_backbone)
    with pytest.raises(ValueError,match='controls'):
        compare_robustness([tmp_path/'base',other],tmp_path/'implicit.csv')
    compared=compare_robustness([tmp_path/'base',other],tmp_path/'explicit.csv',vary=['backbone'])
    assert compared['vary'] == ['backbone']
    assert {row['backbone'] for row in compared['rows']} == {'resnet18','resnet50'}

    def different_lr(controls, value):
        different_backbone(controls,value)
        controls['training']['head_lr'] *= 2
    wrong_lr = variant('wrong-lr', different_lr)
    with pytest.raises(ValueError,match='controls'):
        compare_robustness([tmp_path/'base',wrong_lr],tmp_path/'wrong-lr.csv',vary=['backbone'])

    def copied_data(controls, _):
        controls['dataset']['root'] = 'other-machine/copied-snapshot'
        controls['dataset']['source']['csv_file'] = 'other-machine/source.csv'
        controls['dataset']['source']['image_root'] = 'other-machine/images'
        controls['dataset']['split']['previous_root'] = 'other-machine/previous'
    relocated = variant('relocated',copied_data)
    compare_robustness([tmp_path/'base',relocated],tmp_path/'relocated.csv')

    def different_revision(controls, _):
        controls['dataset']['source']['revision'] = 'other-version'
    changed_revision = variant('different-revision',different_revision)
    with pytest.raises(ValueError,match='controls'):
        compare_robustness([tmp_path/'base',changed_revision],tmp_path/'different-revision.csv')

    def different_split_hash(_, value):
        value['split_hashes']['train'] = 'different'
    changed_split = variant('different-split',different_split_hash)
    with pytest.raises(ValueError,match='split hashes'):
        compare_robustness([tmp_path/'base',changed_split],tmp_path/'different-split.csv',vary=['backbone'])

    def same_backbone_different_weights(controls, _):
        controls['model']['weights'] = 'IMAGENET1K_V2'
    other_weights = variant('other-weights',same_backbone_different_weights)
    with pytest.raises(ValueError,match='explicitly vary weights'):
        compare_robustness([tmp_path/'base',other_weights],tmp_path/'wrong-weights.csv',vary=['backbone'])
    compare_robustness([tmp_path/'base',other_weights],tmp_path/'weights.csv',vary=['weights'])

    def different_seed_and_occlusion(controls, _):
        controls['training']['seed'] += 1
        controls['augmentation']['occlusion'] = {'preset':'light'}
    different = variant('different',different_seed_and_occlusion)
    with pytest.raises(ValueError,match='controls'):
        compare_robustness([tmp_path/'base',different],tmp_path/'seed-only.csv',vary=['seed'])
    compare_robustness([tmp_path/'base',different],tmp_path/'seed-and-occlusion.csv',vary=['seed','occlusion'])

    def different_mapping(controls, value):
        different_backbone(controls,value)
        value['mapping']['type']['chair'] = 999
    wrong_mapping = variant('wrong-mapping',different_mapping)
    with pytest.raises(ValueError,match='mapping'):
        compare_robustness([tmp_path/'base',wrong_mapping],tmp_path/'wrong-mapping.csv',vary=['backbone'])
    old = variant('old-evidence',lambda controls,value:value['training_evidence'].pop('evidence_version'))
    with pytest.raises(ValueError,match='reevaluate'):
        compare_robustness([tmp_path/'base',old],tmp_path/'old-evidence.csv',vary=['backbone'])
    with pytest.raises(ValueError,match='axes'):
        compare_robustness([tmp_path/'base',other],tmp_path/'invalid-axis.csv',vary=['typo'])

    mismatched = tmp_path/'bad-evidence'
    shutil.copytree(tmp_path/'base', mismatched)
    value = json.loads((mismatched/'robustness.json').read_text(encoding='utf-8'))
    value['training_evidence']['controls']['model']['backbone'] = 'resnet50'
    (mismatched/'robustness.json').write_text(json.dumps(value),encoding='utf-8')
    with pytest.raises(ValueError,match='differ from resolved'):
        compare_robustness([tmp_path/'base',mismatched],tmp_path/'bad-evidence.csv',vary=['backbone'])
    value['training_evidence']['checkpoint_sha256'] = 'different'
    (mismatched/'robustness.json').write_text(json.dumps(value),encoding='utf-8')
    with pytest.raises(ValueError,match='Checkpoint selection'):
        compare_robustness([tmp_path/'base',mismatched],tmp_path/'bad-checkpoint.csv',vary=['backbone'])


def test_evidence_requires_original_config_snapshot_and_validation_best(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1]/'configs/project_smoke.yaml')
    checkpoint = mock_verified_run(tmp_path,cfg)
    payload = {'config':training_settings(cfg),'epoch':1}
    assert training_evidence(checkpoint,payload)['checkpoint_selection'] == 'best_validation'
    config_path = tmp_path/'resolved_config.yaml'
    config_text = config_path.read_text(encoding='utf-8')
    changed = copy.deepcopy(cfg)
    changed['model']['backbone'] = 'resnet50'
    config_path.write_text(yaml.safe_dump(changed),encoding='utf-8')
    assert training_evidence(checkpoint,payload)['status'] == 'unknown'
    config_path.write_text(config_text,encoding='utf-8')
    snapshot = tmp_path/'source/model.py'
    snapshot.write_text('# changed source',encoding='utf-8')
    assert training_evidence(checkpoint,payload)['status'] == 'unknown'
    snapshot.write_text('# source fixture',encoding='utf-8')
    checkpoint.write_bytes(b'changed')
    assert training_evidence(checkpoint,payload)['status'] == 'unknown'
    checkpoint.write_bytes(b'test fixture')
    (tmp_path/'summary.json').write_text(json.dumps({'best_epoch':2}))
    assert training_evidence(checkpoint,payload)['status'] == 'unknown'
    (tmp_path/'summary.json').write_text(json.dumps({'best_epoch':1}))
    altered_payload = {'config':{**payload['config'],'finetune_lr':9.0},'epoch':1}
    assert training_evidence(checkpoint,altered_payload)['status'] == 'unknown'
    last = tmp_path/'last.pt'
    last.write_bytes(b'test fixture')
    assert training_evidence(last,payload)['status'] == 'unknown'
