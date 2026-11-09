from pathlib import Path
import pytest

from furniture.suite import plan_suite, describe_plan


PROJECT = Path(__file__).resolve().parents[1]/'configs'/'project_controlled.yaml'


def test_suite_plans_screening_controlled_and_repeats_without_training():
    screening = plan_suite(PROJECT, 'screening')
    assert describe_plan(screening)['runs'] == 10
    assert len({p['run_dir'] for p in screening}) == 10
    assert all(p['config']['training']['epochs'] == 10 for p in screening)
    assert all(p['config']['augmentation']['preset'] == 'easy' for p in screening)
    controlled = plan_suite(PROJECT, 'controlled', ['resnet18', 'resnet50', 'densenet121'])
    assert len(controlled) == 24
    repeat = plan_suite(PROJECT, 'repeated', ['resnet18'], [123, 2026], ['C1','C2','C3','C5'])
    assert len(repeat) == 8
    ablations = plan_suite(PROJECT, 'ablation', ['resnet18'])
    assert [p['config']['training']['tasks'] for p in ablations] == [['type'],['style']]
    with pytest.raises(ValueError, match='distinct'):
        plan_suite(PROJECT, 'controlled', ['resnet18','resnet18'])
