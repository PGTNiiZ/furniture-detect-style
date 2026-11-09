import csv
import json
from pathlib import Path

from furniture.robustness import task_metrics
from furniture.suite import plan_suite
from scripts import post_training_audit as post
from scripts.release_evidence import sha


def test_robustness_reuses_one_validation_manifest_and_checks_predictions(tmp_path, monkeypatch):
    data,output=tmp_path/'data',tmp_path/'output'
    data.mkdir(); output.mkdir()
    rows=[{'image_id':'a','image_path':'a.png','pixel_hash':'hash-a','type':'chair','style':'modern'}]
    with (data/'val.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    mapping={'type':{'chair':0},'style':{'modern':0}}
    snapshot={'split_hashes':{'train':'x','val':sha(data/'val.csv'),'test':'z'},'class_mapping':mapping}
    plans=plan_suite('configs/project_v2_conservative.yaml','controlled',['resnet18'])
    items=[]
    for i,p in enumerate(plans):
        run=tmp_path/f'r{i}'; run.mkdir(); (run/'best.pt').write_bytes(str(i).encode())
        items.append({'config':p['config'],'config_sha256':f'cfg-{i}','attempts':[{'run_dir':str(run)}]})
    monkeypatch.setattr(post,'DATA',data); monkeypatch.setattr(post,'OUTPUT',output)
    used_manifests=[]
    def fake_evaluate(root,checkpoint,destination,split,seed,manifest):
        used_manifests.append(Path(manifest).read_text(encoding='utf8'))
        destination=Path(destination); destination.mkdir()
        metric={'type':task_metrics([0],[0],['chair']),'style':task_metrics([0],[0],['modern'])}
        result={'checkpoint_sha256':sha(checkpoint),'split_hashes':snapshot['split_hashes'],'mapping':mapping,
                'metrics':{level:metric for level in ('clean','light','strong')},
                'absolute_drop':{level:{task:{'accuracy':0,'macro_f1':0} for task in mapping} for level in ('light','strong')}}
        (destination/'robustness.json').write_text(json.dumps(result))
        (destination/'occlusion_manifest.json').write_text(Path(manifest).read_text(encoding='utf8'))
        record={'image_id':'a','image_path':'a.png','true_type':'chair','pred_type':'chair',
                'true_style':'modern','pred_style':'modern'}
        for level in ('clean','light','strong'):
            with (destination/f'predictions_{level}.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(record)); writer.writeheader(); writer.writerow(record)
    def fake_compare(paths,output,vary):
        assert vary == ['occlusion'] and len(paths)==3
        Path(output).write_text('comparison')
    monkeypatch.setattr(post,'evaluate_robustness',fake_evaluate)
    monkeypatch.setattr(post,'compare_robustness',fake_compare)
    post.robustness({'backbone':'resnet18'},items,snapshot)
    assert len(used_manifests)==6 and len(set(used_manifests))==1
    assert (output/'robustness-report/metrics.csv').is_file()


def test_selected_demo_failure_is_recorded_and_explicit_retry_keeps_history(tmp_path, monkeypatch):
    monkeypatch.setattr(post,'OUTPUT',tmp_path)
    (tmp_path/'workflow-status.json').write_text(json.dumps({'stage':'complete'}))
    (tmp_path/'selection-frozen.json').write_text(json.dumps({'selected':{'checkpoint_sha256':'fixed'}}))
    calls=[]
    def run_demo(selection):
        calls.append(selection['checkpoint_sha256'])
        if len(calls)==1:
            raise RuntimeError('detector unavailable')
    monkeypatch.setattr(post,'selected_room_demo',run_demo)
    post.wait_for_workflow(123)
    assert json.loads((tmp_path/'selected-room-demo-status.json').read_text())['status']=='failed'
    post.wait_for_workflow(123)
    assert len(calls)==1
    post.wait_for_workflow(123,retry_demo=True)
    assert len(calls)==2 and json.loads((tmp_path/'selected-room-demo-status.json').read_text())['status']=='complete'
    assert len(list((tmp_path/'selected-room-demo-events').glob('*.json')))==2
