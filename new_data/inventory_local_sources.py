"""Read-only inventory of a complete local sources ZIP; does not extract files."""
import argparse
import collections
import csv
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

IMAGE_EXTENSIONS={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}
ANNOTATION_EXTENSIONS={'.csv','.json','.jsonl','.xml','.yaml','.yml','.txt','.parquet','.mat'}
ARCHIVE_EXTENSIONS={'.zip','.tar','.gz','.7z'}


def inventory(archive, output, expected_size=None, expected_sha256=None, verify_crc=True):
    archive=Path(archive).resolve(); output=Path(output).resolve()
    before=archive.stat()
    if expected_size is not None and before.st_size!=expected_size:
        raise ValueError(f'Archive size mismatch: expected {expected_size}, got {before.st_size}')
    digest=hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): digest.update(block)
    actual=digest.hexdigest()
    if expected_sha256 and actual.lower()!=expected_sha256.lower():
        raise ValueError('Archive SHA-256 differs from the supplied checksum')
    stats={}; seen=set(); duplicates=[]
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if item.is_dir(): continue
            parts=PurePosixPath(item.filename).parts
            source=parts[1] if len(parts)>1 else '(root)'
            row=stats.setdefault(source,{'files':0,'image_extension_entries':0,'other_files':0,
                'uncompressed_bytes':0,'compressed_bytes':0,'nested_archives':0,
                'annotation_extension_entries':0,'annotation_examples':[]})
            ext=PurePosixPath(item.filename).suffix.lower()
            row['files']+=1; row['uncompressed_bytes']+=item.file_size; row['compressed_bytes']+=item.compress_size
            row['image_extension_entries']+=int(ext in IMAGE_EXTENSIONS)
            row['other_files']+=int(ext not in IMAGE_EXTENSIONS)
            row['nested_archives']+=int(ext in ARCHIVE_EXTENSIONS)
            if ext in ANNOTATION_EXTENSIONS:
                row['annotation_extension_entries']+=1
                if len(row['annotation_examples'])<30: row['annotation_examples'].append(item.filename)
            if item.filename in seen: duplicates.append(item.filename)
            seen.add(item.filename)
        bad=z.testzip() if verify_crc else None
        if bad is not None: raise zipfile.BadZipFile(f'CRC verification failed: {bad}')
    after=archive.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('Archive changed during inspection')
    output.mkdir(parents=True,exist_ok=False)
    fields=['source','files','image_extension_entries','other_files','uncompressed_bytes','compressed_bytes',
            'nested_archives','annotation_extension_entries']
    with (output/'sources.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for source,row in sorted(stats.items()): writer.writerow({'source':source,**{k:row[k] for k in fields[1:]}})
    result={'archive':str(archive),'archive_bytes':before.st_size,'archive_sha256':actual,
            'trusted_sha256_supplied':bool(expected_sha256),'full_member_crc_verified':verify_crc,
            'file_entries':sum(r['files'] for r in stats.values()),
            'image_extension_entries':sum(r['image_extension_entries'] for r in stats.values()),
            'duplicate_member_names':duplicates,'sources':stats,
            'limitations':['Image extensions are not independent-image counts.',
              'Nested archives are counted as files and are not expanded or inspected.',
              'Annotation names do not establish schema, labels, licensing, or train eligibility.']}
    (output/'inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',required=True); p.add_argument('--output',required=True)
    p.add_argument('--expected-size',type=int); p.add_argument('--expected-sha256')
    p.add_argument('--skip-crc',action='store_true',help='Central directory only; record that CRC was not checked')
    a=p.parse_args(); report=inventory(a.archive,a.output,a.expected_size,a.expected_sha256,not a.skip_crc)
    print(json.dumps({k:report[k] for k in ('archive','archive_bytes','archive_sha256','full_member_crc_verified','file_entries','image_extension_entries')},indent=2))


if __name__=='__main__': main()
