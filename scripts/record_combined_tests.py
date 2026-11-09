"""Bind one complete pytest invocation to the code tested before and after it."""
import argparse
import json
from pathlib import Path
import re

from scripts.release_evidence import code_fingerprint, sha


ROOT = Path(__file__).resolve().parents[1]


def capture(root, output):
    with Path(output).open('x', encoding='utf-8') as stream:
        json.dump(code_fingerprint(root), stream, indent=2)


def bind(root, before_path, log_path, output, exit_code):
    root, log = Path(root).resolve(), Path(log_path).resolve()
    if exit_code != 0 or not log.is_relative_to(root):
        raise ValueError('Combined pytest must finish successfully with a workspace log')
    before = json.loads(Path(before_path).read_text(encoding='utf-8'))
    if before != code_fingerprint(root):
        raise ValueError('Code or configs changed while combined tests were running')
    raw = log.read_bytes()
    text = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
    matches = re.findall(r'(\d+) passed((?:, [^\r\n]+?)?) in ([\d.]+)s', text)
    if not matches:
        raise ValueError('Combined test summary is missing')
    passed, extras, seconds = matches[-1]
    skipped = re.search(r'(\d+) skipped', extras)
    result = {'result':'passed', 'passed':int(passed), 'skipped':int(skipped.group(1) if skipped else 0),
              'seconds':float(seconds), 'log':log.relative_to(root).as_posix(), 'log_sha256':sha(log),
              'command':'python -m pytest -q (entire suite)', 'code_files':before}
    with Path(output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    before = sub.add_parser('capture')
    before.add_argument('--output', required=True)
    after = sub.add_parser('bind')
    after.add_argument('--before', required=True)
    after.add_argument('--log', required=True)
    after.add_argument('--output', required=True)
    after.add_argument('--exit-code', type=int, required=True)
    args = parser.parse_args()
    if args.action == 'capture':
        capture(ROOT, args.output)
    else:
        result = bind(ROOT, args.before, args.log, args.output, args.exit_code)
        print(json.dumps({k:v for k,v in result.items() if k != 'code_files'}, indent=2))


if __name__ == '__main__':
    main()
