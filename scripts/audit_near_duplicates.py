"""Exhaustively enumerate dHash candidates within source-train only, without test metrics."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from furniture.data import read_rows


def audit(root, output, threshold=4):
    root, output = Path(root), Path(output)
    if not isinstance(threshold, int) or not 0 <= threshold <= 8:
        raise ValueError('threshold must be an integer in [0, 8]')
    report_path, pairs_path = output.with_suffix('.json'), output.with_suffix('.csv')
    if report_path.exists() or pairs_path.exists():
        raise ValueError('Audit output already exists')
    rows, hashes = [], {}
    for split in ('train','val','test'):
        path = root/f'{split}.csv'
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.extend({**row, '_split':split} for row in read_rows(path) if row.get('source_split') == 'train')
    if not rows:
        raise ValueError('No source train rows; refusing to inspect external holdout')
    try:
        values = np.array([int(row['dhash'],16) for row in rows], dtype=np.uint64)
    except (KeyError, ValueError, OverflowError) as error:
        raise ValueError('Invalid dHash in source train manifests') from error
    if any(len(row['dhash']) != 16 for row in rows):
        raise ValueError('dHash must have 16 hexadecimal characters')
    lookup = np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)
    output.parent.mkdir(parents=True,exist_ok=True)
    count, cross, examined, same_pixels = 0, 0, 0, 0
    by_split = Counter()
    with pairs_path.open('x',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream, ['image_id_a','split_a','image_id_b','split_b','distance','exact_pixels'])
        writer.writeheader()
        for start in range(0,len(rows),256):
            stop = min(start+256,len(rows))
            xor = np.bitwise_xor(values[start:stop,None],values[None,:])
            counts = lookup[xor.view(np.uint8).reshape(stop-start,len(rows),8)].sum(axis=2)
            for local,i in enumerate(range(start,stop)):
                candidates = np.flatnonzero(counts[local, i+1:] <= threshold) + i + 1
                examined += len(rows)-i-1
                for j in candidates.tolist():
                    a,b = rows[i],rows[j]
                    exact = a['pixel_hash'] == b['pixel_hash']
                    writer.writerow({'image_id_a':a['image_id'],'split_a':a['_split'],
                                     'image_id_b':b['image_id'],'split_b':b['_split'],
                                     'distance':int(counts[local,j]),'exact_pixels':exact})
                    count += 1
                    cross += a['_split'] != b['_split']
                    same_pixels += exact
                    by_split['/'.join(sorted((a['_split'],b['_split'])))] += 1
    report = {'source_split':'train', 'rows':len(rows), 'method':'64-bit dHash, exhaustive Hamming distance',
              'threshold':threshold, 'examined_pairs':examined, 'candidate_pairs':count,
              'cross_split_pairs':cross, 'exact_pixel_pairs':same_pixels, 'candidates_by_split':dict(by_split),
              'truncated':False, 'manifest_sha256':hashes, 'candidates_csv':str(pairs_path),
              'note':'Candidates require human review; dHash proximity does not prove shared product identity. External source test was excluded from pair comparison.'}
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True,help='New output prefix for JSON and CSV')
    parser.add_argument('--threshold',type=int,default=4)
    args=parser.parse_args()
    print(json.dumps(audit(args.data,args.output,args.threshold),indent=2))
