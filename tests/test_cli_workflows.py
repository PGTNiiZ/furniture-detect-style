import json
import sys
import pytest

from furniture.cli import main


def test_cli_forwards_recovery_and_comparison_axes(monkeypatch, capsys):
    import furniture.pipeline as pipeline
    import furniture.robustness as robustness
    calls = []
    monkeypatch.setattr(pipeline, 'resume_run', lambda path, recover_running=False: calls.append((path,recover_running)) or {'ok':True})
    monkeypatch.setattr(sys, 'argv', ['furniture','resume','--run-dir','run','--recover-running'])
    main()
    assert calls == [('run',True)] and json.loads(capsys.readouterr().out)['ok']
    monkeypatch.setattr(robustness, 'compare_robustness', lambda paths, output, vary: {'vary':vary})
    monkeypatch.setattr(sys, 'argv', ['furniture','compare-robustness','--results','a','b','--output','r.csv','--vary','backbone'])
    main()
    assert json.loads(capsys.readouterr().out)['vary'] == ['backbone']


def test_cli_continues_suite_with_explicit_flags(monkeypatch, capsys):
    import furniture.suite as suite
    captured = []
    monkeypatch.setattr(suite, 'run_suite', lambda *args: captured.append(args) or {'ok':True})
    monkeypatch.setattr(sys, 'argv', ['furniture','suite','--continue-manifest','suite.json',
                                      '--retry-failed','--recover-running','--fail-fast'])
    main()
    assert captured == [(None,'suite.json',True,True,True,False)]
    assert json.loads(capsys.readouterr().out)['ok']


def test_cli_requires_explicit_opt_in_for_incomplete_labels(monkeypatch, capsys):
    import furniture.cli as cli
    import furniture.prepare as prepare
    monkeypatch.setattr(cli, 'load_config', lambda *args: {'example':True})
    monkeypatch.setattr(prepare, 'prepare_data', lambda cfg, *, exclude_incomplete_labels: {'excluded':exclude_incomplete_labels})
    monkeypatch.setattr(sys, 'argv', ['furniture','prepare-data','--exclude-incomplete-labels'])
    main()
    assert json.loads(capsys.readouterr().out)['excluded'] is True


def test_suite_cli_signals_incomplete_runs(monkeypatch, capsys):
    import furniture.suite as suite
    monkeypatch.setattr(suite, 'run_suite', lambda *args: {'counts':{'failed':1}})
    monkeypatch.setattr(sys, 'argv', ['furniture','suite','--continue-manifest','suite.json'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)['counts']['failed'] == 1


def test_external_protocol_cli_does_not_infer(monkeypatch, capsys):
    import furniture.external_evaluation as external
    called=[]
    monkeypatch.setattr(external,'create_external_protocol',lambda a,b,c:called.append((a,b,c)) or {'locked':True})
    monkeypatch.setattr(external,'evaluate_external',lambda *args:pytest.fail('unexpected inference'))
    monkeypatch.setattr(sys,'argv',['furniture','external-protocol','--train-manifest','train.csv',
                                    '--external-manifest','external.csv','--output','protocol.json'])
    main()
    assert called==[('train.csv','external.csv','protocol.json')]
    assert json.loads(capsys.readouterr().out)['locked'] is True
