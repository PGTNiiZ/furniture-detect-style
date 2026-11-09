import json
from pathlib import Path
import sys

import pytest

from scripts import orchestrate_model_release as workflow


def test_controlled_selection_is_finetuned_validation_only(tmp_path, monkeypatch):
    items, payloads = [], {}
    for backbone in ('alpha', 'beta', 'gamma'):
        for index in range(1, 9):
            condition = f'C{index}'
            root = tmp_path/f'{backbone}-{condition}'
            root.mkdir()
            (root/'best.pt').write_bytes(f'{backbone}-{condition}'.encode())
            items.append({'config': {'experiment': {'id':f'controlled-{backbone}-{condition}-42-type-style'},
                                     'model': {'backbone':backbone}},
                          'attempts':[{'run_dir':str(root)}]})
            # C1 has the highest score but must not be eligible for the shared-backbone model.
            score = 1. if index == 1 else (.9 if condition == 'C3' else .5)
            payloads[str(root/'best.pt')] = {'val':{'type':{'macro_f1':score},'style':{'macro_f1':score}},
                                              'shared_backbone_update_epochs':0 if index == 1 else 1,
                                              'split_hashes':{'train':'train-hash'}}
    monkeypatch.setattr(workflow.torch, 'load', lambda path, **kwargs: payloads[str(path)])
    result = workflow.select_controlled(items)
    assert result['selected']['condition'] == 'C3'
    assert result['selected']['backbone'] == 'alpha'  # stable tie-break
    assert result['test_used_for_selection'] is False
    assert len(result['candidates']) == 21
    payloads[str(tmp_path/'alpha-C3'/'best.pt')]['shared_backbone_update_epochs'] = 0
    result = workflow.select_controlled(items)
    assert result['selected']['backbone'] == 'beta'
    assert len(result['excluded']) == 1


def test_validation_table_identifies_single_task_condition(tmp_path):
    run = tmp_path/'run'
    run.mkdir()
    (run/'best.pt').write_bytes(b'checkpoint')
    (run/'summary.json').write_text(json.dumps({'epochs':3,'best_epoch':2,'shared_backbone_update_epochs':1}))
    values = {'accuracy':.1,'macro_precision':.2,'macro_recall':.3,'macro_f1':.4,'weighted_f1':.5}
    (run/'validation.json').write_text(json.dumps({'type':values}))
    item = {'config':{'experiment':{'id':'ablation-resnet18-C3-42-type'},'model':{'backbone':'resnet18'},
                       'training':{'seed':42,'tasks':['type']}},'attempts':[{'run_dir':str(run)}]}
    output = tmp_path/'table.csv'
    workflow.write_validation_table(output, {'single-task':[item]})
    text = output.read_text(encoding='utf-8')
    assert 'single-task,resnet18,C3,42,type' in text
    with pytest.raises(FileExistsError):
        workflow.write_validation_table(output, {'single-task':[item]})


def test_completed_evidence_required_before_advancing(tmp_path):
    manifest = tmp_path/'suite.json'
    manifest.write_text(json.dumps({'runs':[{'status':'planned'}]}))
    with pytest.raises(ValueError, match='incomplete'):
        workflow.completed_runs(manifest, expected=1)


def test_workflow_freezes_selection_before_held_out_evaluation(tmp_path, monkeypatch):
    output = tmp_path/'release'
    monkeypatch.setattr(workflow, 'OUTPUT', output)
    monkeypatch.setattr(workflow, 'ROOT', tmp_path)
    screening = []
    for name in ('a', 'b', 'c'):
        run = tmp_path/name
        run.mkdir()
        (run/'best.pt').write_bytes(name.encode())
        screening.append({'config':{'model':{'backbone':name}}, 'attempts':[{'run_dir':str(run)}]})
    selected = [{'backbone':x['config']['model']['backbone'], 'run_dir':x['attempts'][0]['run_dir']} for x in screening]
    screening_selection = {'selected':selected, 'checkpoint_sha256':{x['run_dir']:workflow.sha(Path(x['run_dir'])/'best.pt') for x in selected}}
    final = {'backbone':'a', 'run_dir':selected[0]['run_dir'], 'checkpoint_sha256':screening_selection['checkpoint_sha256'][selected[0]['run_dir']]}
    controlled = [screening[0]]
    monkeypatch.setattr(workflow, 'wait_for_screening', lambda pid: screening)
    monkeypatch.setattr(workflow, 'source_contract', lambda items: {'source':'same'})
    monkeypatch.setattr(workflow, 'validate_suite_runs', lambda items:None)
    monkeypatch.setattr(workflow, 'screen_selection', lambda items: screening_selection)
    monkeypatch.setattr(workflow, 'run_stage', lambda stage, *args, **kwargs: controlled if stage == 'controlled' else [])
    monkeypatch.setattr(workflow, 'select_controlled', lambda items: {'selected':final})
    calls = []
    def test_artifacts(selected, controlled, ablation, repeated):
        assert (output/'selection-frozen.json').is_file()
        calls.append('reports_after_frozen_selection')
    monkeypatch.setattr(workflow, 'complete_artifacts', test_artifacts)
    monkeypatch.setattr(sys, 'argv', ['workflow', '--screening-pid', '123'])
    workflow.main()
    assert calls == ['reports_after_frozen_selection']
    assert json.loads((output/'workflow-status.json').read_text())['stage'] == 'complete'
