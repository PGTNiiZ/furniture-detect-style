"""Public entry point. See docs/02-config-and-workflow.md for stage-by-stage use."""
import argparse
import json
from pathlib import Path
from .config import load_config,resolve_path


def main():
    parser=argparse.ArgumentParser(description='Furniture Project 2: configuration-driven training')
    commands=parser.add_subparsers(dest='command',required=True)
    for name,help_text in [
        ('check-config','Show the merged configuration; no training'),
        ('prepare-data','Import images and labels, save dataset snapshot'),
        ('audit-data','Check hashes, labels and split leakage'),
        ('preview-augmentation','Render examples before training'),
        ('preview-occlusion','Compare augmentation and occlusion for training images'),
        ('inspect-dataset','Create a browsable dataset and augmentation gallery'),
        ('train','Train one experiment; save a new run'),
        ('experiments','Run controlled E1/E2/E3 comparisons'),
        ('room','Detect furniture and assess target style')]:
        item=commands.add_parser(name,help=help_text)
        item.add_argument('--config',default='configs/project.yaml')
        if name!='experiments':
            item.add_argument('--experiment',help='Default from project config')
        if name=='train':
            item.add_argument('--run-dir',help='Optional new output directory')
        elif name=='prepare-data':
            item.add_argument('--exclude-incomplete-labels',action='store_true',help='Explicitly exclude and report rows with missing type/style')
        elif name=='preview-augmentation':
            item.add_argument('--image',required=True)
            item.add_argument('--output',required=True)
        elif name=='preview-occlusion':
            item.add_argument('--output',required=True)
            item.add_argument('--count',type=int,default=8)
        elif name=='inspect-dataset':
            item.add_argument('--output',required=True,help='New gallery directory')
            item.add_argument('--variants',type=int,default=3,help='Augmented examples per training image')
        elif name=='experiments':
            item.add_argument('--ids',nargs='+',default=['E1','E2','E3'])
            item.add_argument('--dry-run',action='store_true',help='Validate and show selected configs without training')
        elif name=='room':
            item.add_argument('--image',required=True)
            item.add_argument('--checkpoint',required=True)
            item.add_argument('--target',required=True)
            item.add_argument('--catalog')
            item.add_argument('--output',required=True)
    commands.add_parser('check-device',help='Show available CUDA GPU and selected device')
    compare=commands.add_parser('compare',help='Compare validation metrics from compatible runs')
    compare.add_argument('--runs',nargs='+',required=True)
    compare.add_argument('--output',required=True)
    compare.add_argument('--vary',nargs='*',choices=['backbone','augmentation','occlusion','training_mode','tasks','seed'],default=[])
    evaluate=commands.add_parser('evaluate-run',help='Evaluate held-out test with saved run config')
    evaluate.add_argument('--run-dir',required=True)
    evaluate.add_argument('--split',choices=['val','test'],default='test')
    evaluate.add_argument('--device',default='auto')
    resume=commands.add_parser('resume',help='Resume an interrupted run from its last completed epoch')
    resume.add_argument('--run-dir',required=True)
    resume.add_argument('--recover-running',action='store_true',help='Recover an orphaned local running run after safety checks')
    image_inference=commands.add_parser('classify-image',help='Classify a single furniture crop')
    image_inference.add_argument('--image',required=True)
    image_inference.add_argument('--checkpoint',required=True)
    image_inference.add_argument('--output',required=True)
    image_inference.add_argument('--device',default='auto')
    external_protocol=commands.add_parser('external-protocol',help='Freeze external source annotation protocol without inference')
    external_protocol.add_argument('--train-manifest',required=True)
    external_protocol.add_argument('--external-manifest',required=True)
    external_protocol.add_argument('--output',required=True)
    external_eval=commands.add_parser('evaluate-external',help='Explicitly evaluate external annotations using a frozen protocol')
    external_eval.add_argument('--protocol',required=True)
    external_eval.add_argument('--train-manifest',required=True)
    external_eval.add_argument('--external-manifest',required=True)
    external_eval.add_argument('--image-root',required=True)
    external_eval.add_argument('--checkpoint',required=True)
    external_eval.add_argument('--output',required=True)
    external_eval.add_argument('--device',default='auto')
    robustness=commands.add_parser('evaluate-robustness',help='Evaluate clean and deterministic occlusion conditions')
    robustness.add_argument('--data',required=True)
    robustness.add_argument('--checkpoint',required=True)
    robustness.add_argument('--output',required=True)
    robustness.add_argument('--split',choices=['val','test'],default='val')
    robustness.add_argument('--device',default='auto')
    robustness.add_argument('--seed',type=int,default=42)
    robustness.add_argument('--manifest')
    benchmark=commands.add_parser('benchmark',help='Measure checkpoint inference latency on synthetic tensors')
    benchmark.add_argument('--checkpoint',required=True)
    benchmark.add_argument('--output',required=True)
    benchmark.add_argument('--device',default='auto')
    benchmark.add_argument('--batch-size',type=int,default=1)
    benchmark.add_argument('--warmup',type=int,default=5)
    benchmark.add_argument('--repetitions',type=int,default=30)
    suite=commands.add_parser('suite',help='Plan or run screening, controlled, ablation or repeated seeds')
    suite.add_argument('--config',default='configs/project_controlled.yaml')
    suite_action=suite.add_mutually_exclusive_group(required=True)
    suite_action.add_argument('--stage',choices=['screening','controlled','ablation','repeated'])
    suite_action.add_argument('--continue-manifest',help='Continue a previously created suite manifest')
    suite.add_argument('--backbones',nargs='+')
    suite.add_argument('--seeds',nargs='+',type=int)
    suite.add_argument('--conditions',nargs='+')
    suite.add_argument('--dry-run',action='store_true')
    suite.add_argument('--manifest',help='Path for a new persistent suite manifest')
    suite.add_argument('--fail-fast',action='store_true',help='Stop the suite after the first failure')
    suite.add_argument('--retry-failed',action='store_true',help='Retry failed or interrupted attempts')
    suite.add_argument('--recover-running',action='store_true',help='Check and recover orphaned running attempts')
    suite.add_argument('--restart-failed',action='store_true',help='Start a new attempt for failed runs without altering their old checkpoints')
    selection=commands.add_parser('select-screening',help='Select backbones from compatible completed screening runs')
    selection.add_argument('--runs',nargs='+',required=True)
    selection.add_argument('--count',type=int,default=3)
    robust_compare=commands.add_parser('compare-robustness',help='Compare matching robustness evaluations')
    robust_compare.add_argument('--results',nargs='+',required=True)
    robust_compare.add_argument('--output',required=True)
    robust_compare.add_argument('--vary',nargs='*',choices=['backbone','weights','augmentation','occlusion','training_mode','tasks','seed','device'],default=[])
    report=commands.add_parser('report',help='Generate tables and figures from completed runs')
    report.add_argument('--runs',nargs='+',required=True)
    report.add_argument('--output',required=True)
    report.add_argument('--split',choices=['validation','test'],default='validation')
    report.add_argument('--vary',nargs='*',choices=['backbone','augmentation','occlusion','training_mode','tasks','seed'],default=[])
    room_eval=commands.add_parser('evaluate-room',help='Evaluate reference and detector crops and count misses')
    room_eval.add_argument('--annotations',required=True)
    room_eval.add_argument('--detections',required=True)
    room_eval.add_argument('--image-root',required=True)
    room_eval.add_argument('--checkpoint',required=True)
    room_eval.add_argument('--class-mapping',required=True)
    room_eval.add_argument('--output',required=True)
    room_eval.add_argument('--device',default='auto')
    room_eval.add_argument('--confidence',type=float,default=.25)
    room_eval.add_argument('--min-crop-size',type=int,default=16)
    room_eval.add_argument('--iou-threshold',type=float,default=.5)
    room_eval.add_argument('--split',default='test')
    room_detect=commands.add_parser('export-room-detections',help='Run detector on annotated room images and save boxes')
    room_detect.add_argument('--annotations',required=True)
    room_detect.add_argument('--image-root',required=True)
    room_detect.add_argument('--weights',required=True)
    room_detect.add_argument('--output',required=True)
    room_detect.add_argument('--device',default='auto')
    room_detect.add_argument('--confidence',type=float,default=.05)
    room_detect.add_argument('--split',default='test')
    detector=commands.add_parser('train-detector',help='Train YOLO using a separate box-aware config')
    detector.add_argument('--config',default='configs/inference/room_detector.yaml')
    smoke=commands.add_parser('make-smoke-data',help='Artificial fixtures, not scientific data')
    smoke.add_argument('--output',default='data/smoke')
    args=parser.parse_args()
    if args.command=='check-device':
        import torch
        from .train import choose_device
        available = torch.cuda.is_available()
        result = {'selected_device': str(choose_device('auto')), 'cuda_available': available,
                  'gpu_name': torch.cuda.get_device_name(0) if available else None,
                  'pytorch_version': torch.__version__, 'pytorch_cuda_version': torch.version.cuda,
                  'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory if available else None,
                  'gpu_capability':torch.cuda.get_device_capability(0) if available else None,
                  'tensor_operation_result':torch.ones(4,device=choose_device('auto')).sum().item()}
        if available:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                amp_result = torch.ones((2,2),device='cuda') @ torch.ones((2,2),device='cuda')
            result['amp_float16_test'] = {'dtype':str(amp_result.dtype), 'sum':amp_result.sum().item()}
        else:
            result['amp_float16_test'] = None
    elif args.command=='make-smoke-data':
        from .smoke import make_smoke
        result=make_smoke(args.output)
    elif args.command=='train-detector':
        from .detection import train_detector
        result=train_detector(args.config)
    elif args.command=='compare':
        from .pipeline import compare_runs
        result=compare_runs(args.runs,args.output,args.vary)
    elif args.command=='evaluate-run':
        from .pipeline import evaluate_run
        result=evaluate_run(args.run_dir,args.split,args.device)
    elif args.command=='resume':
        from .pipeline import resume_run
        result=resume_run(args.run_dir,recover_running=args.recover_running)
    elif args.command=='classify-image':
        from .room import classify_image
        result=classify_image(args.image,args.checkpoint,args.output,args.device)
    elif args.command=='external-protocol':
        from .external_evaluation import create_external_protocol
        result=create_external_protocol(args.train_manifest,args.external_manifest,args.output)
    elif args.command=='evaluate-external':
        from .external_evaluation import evaluate_external
        result=evaluate_external(args.protocol,args.train_manifest,args.external_manifest,args.image_root,
                                 args.checkpoint,args.output,args.device)
    elif args.command=='evaluate-robustness':
        from .robustness import evaluate_robustness
        result=evaluate_robustness(args.data,args.checkpoint,args.output,args.split,args.device,args.seed,args.manifest)
    elif args.command=='benchmark':
        from .benchmark import benchmark
        result=benchmark(args.checkpoint,args.output,args.device,args.batch_size,args.warmup,args.repetitions)
    elif args.command=='suite':
        from .suite import plan_suite, describe_plan, run_suite
        if args.continue_manifest:
            if args.dry_run or args.manifest or args.backbones or args.seeds or args.conditions:
                parser.error('--continue-manifest cannot be combined with planning options')
            result=run_suite(None,args.continue_manifest,args.fail_fast,args.retry_failed,args.recover_running,args.restart_failed)
        else:
            if args.retry_failed or args.recover_running or args.restart_failed:
                parser.error('--retry-failed, --restart-failed and --recover-running require --continue-manifest')
            if args.dry_run and (args.manifest or args.fail_fast):
                parser.error('--dry-run cannot be combined with --manifest or --fail-fast')
            plans=plan_suite(args.config,args.stage,args.backbones,args.seeds,args.conditions)
            result=describe_plan(plans) if args.dry_run else run_suite(plans,args.manifest,args.fail_fast,args.retry_failed,args.recover_running)
    elif args.command=='select-screening':
        from .suite import select_screening
        result=select_screening(args.runs,args.count)
    elif args.command=='compare-robustness':
        from .robustness import compare_robustness
        result=compare_robustness(args.results,args.output,args.vary)
    elif args.command=='report':
        from .reports import generate_report
        result=generate_report(args.runs,args.output,args.split,args.vary)
    elif args.command=='evaluate-room':
        from .room_evaluation import evaluate_room_domain
        result=evaluate_room_domain(args.annotations,args.detections,args.image_root,args.checkpoint,args.output,
                                    args.class_mapping,args.device,args.confidence,args.min_crop_size,args.iou_threshold,args.split)
    elif args.command=='export-room-detections':
        from .room_evaluation import export_room_detections
        result=export_room_detections(args.annotations,args.image_root,args.weights,args.output,
                                      args.device,args.confidence,args.split)
    elif args.command=='experiments':
        if args.dry_run:
            if len(set(args.ids)) != len(args.ids):
                parser.error('Duplicate experiment IDs')
            result={'dry_run':True,'experiments':[load_config(args.config, name) for name in args.ids],
                    'note':'Configuration validation only; data and model weights were not checked'}
        else:
            from .pipeline import run_experiments
            result=run_experiments(args.config,args.ids)
    else:
        cfg=load_config(args.config,args.experiment)
        if args.command=='check-config':
            result=cfg
        elif args.command=='prepare-data':
            from .prepare import prepare_data
            result=prepare_data(cfg,exclude_incomplete_labels=args.exclude_incomplete_labels)
        elif args.command=='audit-data':
            from .prepare import audit_data
            result=audit_data(cfg)
        elif args.command=='preview-augmentation':
            from .augmentation import preview
            result=preview(args.image,args.output,cfg['model']['image_size'],cfg['augmentation'],seed=cfg['training']['seed'])
        elif args.command=='preview-occlusion':
            from .augmentation import preview_dataset
            result=preview_dataset(resolve_path(cfg,cfg['dataset']['root']),args.output,cfg['model']['image_size'],
                                   cfg['training']['seed'],args.count,Path(cfg['project_root'])/'configs'/'augmentation')
        elif args.command=='inspect-dataset':
            from .gallery import create_gallery
            dataset_root=resolve_path(cfg,cfg['dataset']['root'])
            missing=[str(dataset_root / f'{split}.csv') for split in ('train','val','test')
                     if not (dataset_root / f'{split}.csv').is_file()]
            if missing:
                parser.error('Dataset is not prepared. Missing: ' + ', '.join(missing) +
                             '. Run prepare-data with this config first. For a demo, use '
                             '--config configs/project_smoke.yaml after make-smoke-data.')
            result=create_gallery(dataset_root,args.output,
                                  cfg['model']['image_size'],cfg['augmentation'],
                                  cfg['training']['seed'],args.variants)
        elif args.command=='train':
            from .pipeline import train_run
            result=train_run(cfg,args.run_dir)
        else:
            from .room import analyze
            settings=cfg['inference']
            weights=settings['detector_weights']
            if len(Path(weights).parts)>1:
                weights=str(resolve_path(cfg,weights))
            result=analyze(args.image,args.checkpoint,weights,args.target,args.output,args.catalog,
                           cfg['training']['device'],settings['minimum_confidence'],settings['detector_confidence'],
                           settings['keep_threshold'],settings['replace_threshold'],settings['top_k'])
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if args.command == 'suite' and not args.dry_run:
        if any(result.get('counts', {}).get(state, 0) for state in ('planned','running','failed','interrupted')):
            raise SystemExit(1)


if __name__=='__main__':
    main()
