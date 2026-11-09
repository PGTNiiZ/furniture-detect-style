"""Package the verified workspace, optionally including the selected checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'artifacts/furniture-project2-research-followup-verified.zip'
SNAPSHOT = ROOT / 'data/furniture-v2-conservative'


def files_under(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f'Required delivery directory is missing: {directory}')
    return (p for p in directory.rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and not any(part.endswith('.egg-info') for part in p.parts)
            and not p.name.endswith(('.pyc', '.pyo')))


def package(output, checkpoint=None, document_overrides=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite delivery: {output}')
    entries = {}

    def add(path):
        path = Path(path)
        name = path.relative_to(ROOT).as_posix()
        if name in entries:
            raise ValueError(f'Duplicate package path: {name}')
        entries[name] = path

    for filename in ('.gitignore', 'CHANGELOG.md', 'README.md', 'RELEASE_STATUS.md', 'pyproject.toml'):
        add(ROOT / filename)
    for name in ('configs', 'docs', 'examples', 'notebooks', 'scripts', 'src', 'tests'):
        for path in files_under(ROOT / name):
            add(path)
    for name, replacement in (document_overrides or {}).items():
        if name not in ('README.md', 'RELEASE_STATUS.md') or not Path(replacement).is_file():
            raise ValueError('Invalid release document override')
        entries[name] = Path(replacement)

    # Manifests and audit evidence are useful independently of the full images.
    for path in SNAPSHOT.iterdir():
        if path.is_file():
            add(path)
    room_intake = ROOT/'data/room-independent-review-v1'
    if room_intake.is_dir():
        for path in files_under(room_intake):
            add(path)
    for name in ('source-train-cross-split-review', 'source-train-duplicate-review'):
        for path in files_under(ROOT / 'artifacts' / name):
            add(path)
    detector_evidence = ROOT/'artifacts/detector-baselines/yolov8n-coco-v8.3.0/verification.json'
    if detector_evidence.is_file():
        add(detector_evidence)
    demo = ROOT/'artifacts/room-inference-demo'
    if (demo/'source-provenance.json').is_file():
        add(demo/'source-provenance.json')
    if (demo/'result-v3').is_dir():
        for path in files_under(demo/'result-v3'):
            add(path)
    for name in ('source-train-near-audit.csv', 'source-train-near-audit.json',
                 'conservative-quarantine-proposal.json', 'furniture-v2-conservative-near-audit.csv',
                 'furniture-v2-conservative-near-audit.json', 'furniture-v2-pretrained-steps.json'):
        add(ROOT / 'artifacts' / name)
    for name in ('protocols', 'furniture-v2-training-benchmarks',
                 'furniture-v2-training-benchmarks-matched'):
        folder = ROOT / 'artifacts' / name
        if folder.exists():
            for path in files_under(folder):
                if path.suffix.lower() in ('.json', '.csv', '.log'):
                    add(path)
    suites = ROOT / 'artifacts/runs/suites'
    if suites.exists():
        for path in suites.glob('furniture-v2-*.json'):
            add(path)
            # Preserve small run evidence, never package model or optimizer checkpoints.
            manifest = json.loads(path.read_text(encoding='utf-8'))
            for item in manifest.get('runs', []):
                if item.get('status') != 'completed':
                    continue
                run = Path(item['attempts'][-1]['run_dir'])
                if not run.resolve().is_relative_to((ROOT/'artifacts/runs').resolve()):
                    raise ValueError(f'Suite run outside workspace: {run}')
                for name in ('status.json', 'summary.json', 'history.json', 'provenance.json',
                             'resolved_config.yaml', 'checkpoint.sha256', 'validation.json',
                             'validation_predictions.csv', 'validation_type_confusion.csv',
                             'validation_style_confusion.csv', 'test.json', 'test_predictions.csv',
                             'test_type_confusion.csv', 'test_style_confusion.csv'):
                    file = run/name
                    if file.is_file():
                        add(file)
    workflow = ROOT / 'artifacts/model-only-release'
    if workflow.exists():
        for path in files_under(workflow):
            if '_attempts' not in path.parts and path.suffix.lower() in ('.json', '.csv', '.png', '.md'):
                add(path)
    if checkpoint is not None:
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file() or not checkpoint.is_relative_to((ROOT/'artifacts/runs').resolve()):
            raise ValueError('Selected checkpoint must be an existing run checkpoint inside the workspace')
        entries['release/checkpoint/best.pt'] = checkpoint

    required = {'train.csv', 'validation.csv', 'synthetic_test.csv', 'real_source_test.csv',
                'dataset_metadata.json', 'near_duplicate_review.csv', 'quarantine.csv'}
    missing = sorted(p for p in required if not (SNAPSHOT / p).is_file())
    if missing:
        raise FileNotFoundError(f'Incomplete dataset evidence: {missing}')
    with zipfile.ZipFile(BASELINE) as original:
        missing = sorted(set(original.namelist()) - set(entries))
    if missing:
        raise ValueError(f'Files from the verified base are missing: {missing}')

    delivery_note = (
        'Research delivery based on furniture-project2-research-followup-verified.zip.\n'
        'Current scope is model-only. See RELEASE_STATUS.md for unfinished experiments.\n'
        'The new manifests and audit evidence are under data/furniture-v2-conservative/.\n'
        + ('The selected checkpoint is at release/checkpoint/best.pt.\n' if checkpoint else
           'Model weights/checkpoints are not included.\n')
        + 'The full training dataset images are not included; provisional room-review images may be included.\n'
        'Review the 494 source-train pairs using artifacts/source-train-cross-split-review/index.html.\n'
        'The earlier exploratory benchmarks in artifacts/furniture-v2-training-benchmarks/ used\n'
        'different sampled examples and must not be used for cross-model comparison.\n'
        'Manifests and screening status are a point-in-time copy; see docs and evidence.\n'
    ).encode('utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    created = False
    try:
        with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                             allowZip64=True) as archive:
            created = True
            for name, path in sorted(entries.items()):
                digest = hashlib.sha256()
                with path.open('rb') as source, archive.open(name, 'w', force_zip64=True) as target:
                    for data in iter(lambda: source.read(1024 * 1024), b''):
                        digest.update(data)
                        target.write(data)
                hashes[name] = digest.hexdigest()
            archive.writestr('DELIVERY_README.txt', delivery_note)
            hashes['DELIVERY_README.txt'] = hashlib.sha256(delivery_note).hexdigest()
            archive.writestr('DELIVERY_SHA256.json', json.dumps(hashes, indent=2, sort_keys=True))
    except BaseException:
        # Only remove the file exclusively created by this function.
        if created:
            output.unlink(missing_ok=True)
        raise
    digest = hashlib.sha256()
    with output.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'zip': str(output), 'files': len(hashes) + 1, 'sha256': digest.hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='New ZIP path (never overwrites an existing file)')
    args = parser.parse_args()
    print(json.dumps(package(args.output), indent=2))
