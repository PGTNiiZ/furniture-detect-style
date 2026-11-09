import copy
import csv
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
from PIL import Image

from furniture.config import load_config, merge_known, validate, resolve_path
from furniture.data import transform, read_rows, write_rows, validate_splits
from furniture.prepare import prepare_data, partition
from furniture.model import AttributeModel
from furniture.pipeline import run_experiments, evaluate_run, compare_runs
from furniture.smoke import make_smoke
from furniture.train import initialize_from_checkpoint

CONFIGS = Path(__file__).resolve().parents[1]/'configs'
torch.set_num_threads(1)


def test_experiment_overrides_are_isolated_and_validated():
    e1=load_config(CONFIGS/'project.yaml','E1')
    e2=load_config(CONFIGS/'project.yaml','E2')
    e3=load_config(CONFIGS/'project.yaml','E3')
    assert e1['training']['freeze_epochs']=='all'
    assert e2['training']['freeze_epochs']==2
    assert not e2['augmentation']['enabled']
    assert e3['augmentation']['horizontal_flip_p']==.5
    assert e1['dataset']==e2['dataset']==e3['dataset']
    with pytest.raises(ValueError,match='Unknown config key'):
        merge_known(e2,{'training':{'learing_rate':.1}})
    invalid=copy.deepcopy(e2)
    invalid['augmentation']['horizontal_flip_p']=2
    with pytest.raises(ValueError,match='out of range'):
        validate(invalid)


def test_controlled_experiment_configs():
    expected = [('none','off'), ('none','off'), ('easy','off'), ('hard','off'),
                ('easy','light'), ('easy','strong'), ('hard','light'), ('hard','strong')]
    for index, pair in enumerate(expected, 1):
        cfg = load_config(CONFIGS/'project_controlled.yaml', f'C{index}')
        assert (cfg['augmentation']['preset'], cfg['augmentation']['occlusion']['preset']) == pair
        assert cfg['training']['freeze_epochs'] == ('all' if index == 1 else 2)


def test_suite_dry_run_loads_configs_without_training(monkeypatch, capsys):
    from furniture.cli import main
    monkeypatch.setattr(sys, 'argv', ['furniture', 'experiments', '--config', str(CONFIGS/'project_controlled.yaml'),
                                     '--ids', 'C3', 'C5', '--dry-run'])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result['dry_run'] is True
    assert [cfg['experiment']['id'] for cfg in result['experiments']] == ['C3', 'C5']


def test_failed_run_records_status_and_provenance(tmp_path):
    from furniture.pipeline import train_run
    cfg = load_config(CONFIGS/'project_smoke.yaml')
    cfg['project_root'] = str(tmp_path)
    run = tmp_path/'failure'
    with pytest.raises(FileNotFoundError):
        train_run(cfg, run)
    status = json.loads((run/'status.json').read_text())
    assert status['status'] == 'failed' and 'FileNotFoundError' in status['stop_reason']
    assert (run/'resolved_config.yaml').exists() and (run/'provenance.json').exists()


def test_resume_cli_workflow_keeps_run_identity(tmp_path, monkeypatch):
    import furniture.train as training
    from furniture.pipeline import train_run, resume_run
    make_smoke(tmp_path/'data/smoke')
    cfg = load_config(CONFIGS/'project_smoke.yaml')
    cfg['project_root'] = str(tmp_path)
    cfg['training']['epochs'] = 2
    cfg['training']['patience'] = 2
    original, calls = training.run_epoch, [0]
    def fail_second_epoch(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 3:
            raise RuntimeError('interrupted by test')
        return original(*args, **kwargs)
    monkeypatch.setattr(training, 'run_epoch', fail_second_epoch)
    run = tmp_path/'run'
    with pytest.raises(RuntimeError, match='interrupted by test'):
        train_run(cfg,run)
    assert json.loads((run/'status.json').read_text())['status'] == 'failed'
    monkeypatch.setattr(training, 'run_epoch', original)
    assert resume_run(run)['epochs'] == 2
    assert json.loads((run/'status.json').read_text())['status'] == 'completed'
    with pytest.raises(ValueError, match='failed or interrupted'):
        resume_run(run)


def test_augmentation_changes_training_only_and_respects_disabled():
    image=Image.new('RGB',(32,32),'black')
    for x in range(16):
        for y in range(32):image.putpixel((x,y),(200,70,10))
    baseline=transform(32,False)(image)
    enabled={'enabled':True,'horizontal_flip_p':1.}
    augmented=transform(32,enabled)(image)
    assert torch.equal(augmented,baseline.flip(-1))
    assert torch.equal(transform(32,{'enabled':False,'horizontal_flip_p':1.})(image),baseline)
    assert torch.equal(transform(32,False)(image),baseline)


def test_presets_and_occlusion_are_independent_and_reproducible():
    from furniture.augmentation import settings
    image = Image.new('RGB', (96, 96), 'black')
    clean = settings(yaml.safe_load((CONFIGS/'augmentation/none_off.yaml').read_text()))
    occluded = settings(yaml.safe_load((CONFIGS/'augmentation/none_strong.yaml').read_text()))
    occluded['occlusion']['probability'] = 1.0
    expected = transform(96, clean)(image)
    torch.manual_seed(1)
    first = transform(96, occluded)(image)
    torch.manual_seed(1)
    assert torch.equal(first, transform(96, occluded)(image))
    assert not torch.equal(first, expected)
    with pytest.raises(ValueError, match='ambiguous'):
        settings({'preset':'hard', 'horizontal_flip_p':.5})


def csv_config(tmp_path):
    cfg=load_config(CONFIGS/'project_new_data.yaml')
    cfg['project_root']=str(tmp_path)
    source=cfg['dataset']['source']
    raw=tmp_path/'data/incoming'
    (raw/'images').mkdir(parents=True)
    rows=[]
    for group in range(12):
        for variant in range(2):
            name=f'{group}-{variant}.png'
            Image.new('RGB',(32,32),(group*15,variant*40+20,100)).save(raw/'images'/name)
            rows.append({'filename':name,'category':'chair' if group%2 else 'table',
                         'design_style':'modern' if group%2 else 'rustic','product_id':f'p-{group}',
                         'split':'train' if group<8 else 'val' if group<10 else 'test'})
    with (raw/'labels.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
    return cfg


def test_csv_new_columns_group_integrity_and_versioned_refresh(tmp_path):
    cfg=csv_config(tmp_path)
    audit=prepare_data(cfg)
    assert audit['counts']=={'train':16,'val':4,'test':4}
    first=resolve_path(cfg,cfg['dataset']['root'])
    first_rows={s:read_rows(first/f'{s}.csv') for s in ['train','val','test']}
    cfg['dataset']['split']['previous_root']=cfg['dataset']['root']
    cfg['dataset']['root']='data/my-furniture-v2'
    cfg['dataset']['id']='my-furniture-v2'
    cfg['dataset']['source']['columns']['split']=None
    prepare_data(cfg)
    second=resolve_path(cfg,cfg['dataset']['root'])
    for split in first_rows:
        assert {r['pixel_hash'] for r in first_rows[split]}=={r['pixel_hash'] for r in read_rows(second/f'{split}.csv')}
    with pytest.raises(ValueError,match='exists'):
        prepare_data(cfg)


def test_group_bridge_across_explicit_splits_is_rejected():
    rows=[{'type':'chair','style':'modern','pixel_hash':str(i),'group_id':'same-product',
           '_split':'train' if i==0 else 'test'} for i in range(2)]
    with pytest.raises(ValueError,match='bridges'):
        partition(rows,{'seed':42,'val_fraction':.15,'test_fraction':.15})


def test_joint_loss_backpropagates_into_both_heads_and_shared_backbone():
    model=AttributeModel({'type':{'chair':0,'table':1},'style':{'modern':0,'rustic':1}},pretrained=False)
    out=model(torch.randn(2,3,64,64))
    loss=sum(torch.nn.functional.cross_entropy(x,torch.tensor([0,1])) for x in out.values())
    loss.backward()
    for head in model.heads.values():
        assert head.weight.grad.abs().sum()>0
    assert model.backbone.conv1.weight.grad.abs().sum()>0


def test_config_driven_three_runs_and_new_dataset_initialization(tmp_path):
    cfgdir=tmp_path/'configs'
    shutil.copytree(CONFIGS,cfgdir)
    project=yaml.safe_load((cfgdir/'project_smoke.yaml').read_text())
    project.update(default_experiment='E2',experiments=['experiments/E1.yaml','experiments/E2.yaml','experiments/E3.yaml'])
    project_path=cfgdir/'test-suite.yaml'
    project_path.write_text(yaml.safe_dump(project))
    e1=yaml.safe_load((cfgdir/'experiments/E1.yaml').read_text())
    e1['overrides']['training']['freeze_epochs']=1
    (cfgdir/'experiments/E1.yaml').write_text(yaml.safe_dump(e1))
    make_smoke(tmp_path/'data/smoke')
    suite=run_experiments(project_path,['E1','E2','E3'])
    assert len(suite['comparison']['rows'])==3
    assert Path(suite['comparison_csv']).exists()
    runs=[Path(x['run_dir']) for x in suite['runs']]
    assert len(set(runs))==3
    assert not any((run/'test.json').exists() for run in runs)
    assert [r['shared_backbone_update_epochs'] for r in suite['runs']]==[0,1,1]
    root=runs[1]
    history=json.loads((root/'history.json').read_text())
    assert history[-1]['learning_rates']['heads'] > history[-1]['learning_rates']['backbone']
    saved=(root/'resolved_config.yaml').read_text()
    (cfgdir/'training/smoke.yaml').write_text('changed after run')
    assert (root/'resolved_config.yaml').read_text()==saved
    evaluated=evaluate_run(root,device='cpu')
    assert 'style' in evaluated and (root/'test_predictions.csv').exists()
    assert (root/'test_type_confusion.csv').exists()
    with pytest.raises(ValueError,match='already recorded'):
        evaluate_run(root,device='cpu')
    payload=torch.load(root/'best.pt',weights_only=True,map_location='cpu')
    splits,mapping=validate_splits(tmp_path/'data/smoke')
    init={'initialization':'checkpoint','pretrained':False,'init_checkpoint':str(root/'best.pt'),'checkpoint_scope':'full'}
    model=AttributeModel(mapping,pretrained=False)
    report=initialize_from_checkpoint(model,mapping,splits,init,torch.device('cpu'))
    assert report['mode']=='warm_start'
    new_mapping=copy.deepcopy(mapping)
    new_mapping['type']['cabinet']=2
    new_model=AttributeModel(new_mapping,pretrained=False)
    with pytest.raises(ValueError,match='identical label mapping'):
        initialize_from_checkpoint(new_model,new_mapping,splits,init,torch.device('cpu'))
    init['checkpoint_scope']='backbone'
    assert initialize_from_checkpoint(new_model,new_mapping,splits,init,torch.device('cpu'))['scope']=='backbone'
    moved=copy.deepcopy(splits)
    moved['test'].append(moved['train'].pop())
    with pytest.raises(ValueError,match='leakage'):
        initialize_from_checkpoint(new_model,new_mapping,moved,init,torch.device('cpu'))
    # A changed validation policy must not silently enter a controlled comparison.
    changed=yaml.safe_load((runs[2]/'resolved_config.yaml').read_text())
    changed['training']['seed']=9
    (runs[2]/'resolved_config.yaml').write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError,match='different'):
        compare_runs(runs)


def test_detector_uses_its_own_box_aware_augmentation(tmp_path,monkeypatch):
    from furniture.detection import train_detector
    cfg=tmp_path/'detector.yaml'
    value=yaml.safe_load((CONFIGS/'inference/room_detector.yaml').read_text())
    value['project_root']='.'
    cfg.write_text(yaml.safe_dump(value))
    (tmp_path/'configs').mkdir()
    shutil.copy(CONFIGS/'detection.example.yaml',tmp_path/'configs/detection.example.yaml')
    for split in ['train','val']:(tmp_path/'data/rooms/images'/split).mkdir(parents=True)
    captured={}
    class FakeYOLO:
        def __init__(self,weights):captured['weights']=weights
        def train(self,**kwargs):captured.update(kwargs)
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=FakeYOLO))
    result=train_detector(cfg)
    assert captured['fliplr']==.5 and captured['mosaic']==0
    assert captured['workers']==0
    assert Path(result['run_dir'],'resolved_data.yaml').exists()


def test_warm_start_preserves_all_ancestor_exposure(tmp_path):
    mapping={'type':{'chair':0},'style':{'modern':0}}
    model=AttributeModel(mapping,pretrained=False)
    current={s:{'pixel_hashes':[s], 'group_ids':[]} for s in ['train','val','test']}
    inherited=copy.deepcopy(current)
    inherited['train']['pixel_hashes'].append('ancestor-training-image')
    path=tmp_path/'parent.pt'
    torch.save({'state_dict':model.state_dict(),'mapping':mapping,'data_membership':current,
                'exposure_membership':inherited},path)
    new_splits={s:[{'pixel_hash':s}] for s in current}
    new_splits['test'].append({'pixel_hash':'ancestor-training-image'})
    cfg=dict(initialization='checkpoint',pretrained=False,init_checkpoint=str(path),checkpoint_scope='full')
    with pytest.raises(ValueError,match='leakage'):
        initialize_from_checkpoint(model,mapping,new_splits,cfg,torch.device('cpu'))
