import pytest
from PIL import Image
import torch

from furniture.data import image_hash, make_splits, validate_splits
from furniture.recommend import decide, replacements
from furniture.smoke import make_smoke
from furniture.train import fit, evaluate, load_model
from furniture.room import classify_crop


def test_pixel_hash_preserves_dimensions():
    assert image_hash(Image.new('RGB',(2,4))) != image_hash(Image.new('RGB',(4,2)))


def test_exact_duplicates_stay_together():
    rows = []
    for i in range(10):
        rows.append({'pixel_hash':str(i),'type':'chair','style':'modern'})
    rows.append(dict(rows[0]))
    first,_ = make_splits(rows)
    second,_ = make_splits(rows)
    assert first == second
    sets = [{r['pixel_hash'] for r in v} for v in first.values()]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]


def test_conflicting_duplicate_labels_fail():
    with pytest.raises(ValueError, match='conflicting'):
        make_splits([{'pixel_hash':'a','type':'chair','style':'modern'},
                     {'pixel_hash':'a','type':'chair','style':'rustic'}])


def test_target_style_and_abstention():
    assert decide({'modern':.85,'rustic':.15},'modern')['action'] == 'keep_candidate'
    assert decide({'modern':.85,'rustic':.15},'rustic')['action'] == 'replace_candidate'
    assert decide({'modern':.51,'rustic':.49},'rustic')['action'] == 'review'
    with pytest.raises(ValueError, match='Unsupported'):
        decide({'modern':1},'unknown')


def test_replacement_is_same_category():
    rows = [{'image_id':'a','image_path':'a.png','type':'table','style':'modern'},
            {'image_id':'b','image_path':'b.png','type':'sofa','style':'modern'}]
    assert replacements(rows,'couch','modern')[0]['image_id'] == 'b'
    assert replacements(rows,'bed','modern') == []


def test_train_checkpoint_evaluate_and_crop(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    data, run = tmp_path/'data', tmp_path/'run'
    make_smoke(data)
    validate_splits(data)
    config = dict(seed=42,image_size=64,batch_size=4,epochs=1,freeze_epochs=0,
                  head_lr=.001,finetune_lr=.0001,weight_decay=.0001,patience=1,
                  pretrained=False,augment=False,device='cpu')
    result = fit(data,run,config)
    assert result['epochs'] == 1
    report = evaluate(data,run/'best.pt',tmp_path/'metrics.json',device='cpu')
    assert 'style' in report and len(report['type']['confusion_matrix']) == 2
    model,payload = load_model(run/'best.pt',torch.device('cpu'))
    scores = classify_crop(model,payload,Image.new('RGB',(50,90)),torch.device('cpu'))
    assert sum(scores['style'].values()) == pytest.approx(1)
    from furniture.room import classify_image
    source = tmp_path/'single.png'
    Image.new('RGB',(50,90),'white').save(source)
    single = classify_image(source,run/'best.pt',tmp_path/'single.json',device='cpu')
    assert set(single['predictions']) == {'type','style'}
    # Mock only the detector boundary; exercise actual crops, checkpoint, decisions and report.
    import sys
    from types import SimpleNamespace
    from furniture.room import analyze
    box = SimpleNamespace(cls=torch.tensor(0), conf=torch.tensor(.9),
                          xyxy=torch.tensor([[0.,0.,25.,30.]]))
    class FakeYOLO:
        def __init__(self, weights):
            pass
        def predict(self, **kwargs):
            return [SimpleNamespace(boxes=[box], names={0:'chair'})]
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=FakeYOLO))
    room_path = tmp_path/'room.png'
    Image.new('RGB',(50,50),'white').save(room_path)
    report = analyze(room_path, run/'best.pt', 'mock.pt', 'modern', tmp_path/'room-output',
                     data/'train.csv', device='cpu')
    assert len(report['objects']) == 1
    assert (tmp_path/'room-output/annotated.jpg').exists()
    with pytest.raises(ValueError, match='Target style unavailable'):
        analyze(room_path, run/'best.pt', 'mock.pt', 'unknown', tmp_path/'unsupported', device='cpu')
    # Tampering after training must be detected before evaluation.
    Image.new('RGB',(32,32),'red').save(data/'images/test-0.png')
    with pytest.raises(ValueError,match='Changed image'):
        evaluate(data,run/'best.pt',tmp_path/'invalid.json',device='cpu')


@pytest.mark.parametrize('task', ['type', 'style'])
def test_single_task_training_and_prediction_exports(tmp_path, task):
    import csv
    torch.set_num_threads(1)
    data, run = tmp_path/'data', tmp_path/'run'
    make_smoke(data)
    config = dict(seed=42,image_size=64,batch_size=4,epochs=1,freeze_epochs=0,
                  head_lr=.001,finetune_lr=.0001,weight_decay=.0001,patience=1,
                  pretrained=False,augment=False,device='cpu', tasks=[task], selection_metric='mean_macro_f1')
    fit(data,run,config)
    report = evaluate(data,run/'best.pt',tmp_path/'metrics.json',device='cpu')
    assert task in report and ({'type','style'} - {task}).isdisjoint(report)
    with (tmp_path/'metrics_predictions.csv').open(encoding='utf-8', newline='') as stream:
        fields = csv.DictReader(stream).fieldnames
    assert f'true_{task}' in fields and f'pred_{task}' in fields
    assert f'true_{next(iter({"type","style"}-{task}))}' not in fields


def test_resume_at_epoch_boundary_matches_uninterrupted_cpu(tmp_path, monkeypatch):
    import furniture.train as training
    torch.set_num_threads(1)
    data = tmp_path/'data'
    make_smoke(data)
    config = dict(seed=42,image_size=64,batch_size=4,epochs=2,freeze_epochs=0,
                  head_lr=.001,finetune_lr=.0001,weight_decay=.0001,patience=3,
                  pretrained=False,augment=False,device='cpu')
    training.fit(data,tmp_path/'whole',config)
    original = training.run_epoch
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('simulated interruption')
        return original(*args, **kwargs)
    monkeypatch.setattr(training, 'run_epoch', interrupt)
    with pytest.raises(RuntimeError, match='simulated'):
        training.fit(data,tmp_path/'interrupted',config)
    monkeypatch.setattr(training, 'run_epoch', original)
    training.fit(data,tmp_path/'interrupted',config,run_prepared=True,resume=True)
    whole = torch.load(tmp_path/'whole'/'last.pt',map_location='cpu',weights_only=True)
    resumed = torch.load(tmp_path/'interrupted'/'last.pt',map_location='cpu',weights_only=True)
    assert whole['epoch'] == resumed['epoch'] == 2
    assert all(torch.equal(value, resumed['state_dict'][key]) for key,value in whole['state_dict'].items())


def test_prepare_exports_hf_rows_without_prompt_leakage(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from furniture.data import prepare, read_rows
    samples = []
    for i in range(12):
        samples.append({'image': Image.new('RGB',(20,20),(i*10,40,60)),
                        'type':'chair', 'style':'modern', 'prompt':'label-bearing text'})
    monkeypatch.setitem(sys.modules,'datasets',SimpleNamespace(load_dataset=lambda *a,**kw: {'train':samples}))
    out = tmp_path/'prepared'
    audit = prepare(out,'mock-dataset','a'*40)
    assert audit['rows'] == 12
    splits, mapping = validate_splits(out)
    assert set(mapping['type']) == {'chair'}
    assert 'prompt' not in read_rows(out/'train.csv')[0]
    assert sum(len(rows) for rows in splits.values()) == 12
