"""Import new labeled image data, preserving group boundaries and previous splits."""
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
from .config import resolve_path
from .data import rgb, image_hash, dhash, write_rows, read_rows, validate_splits


def near_duplicate_candidates(splits, threshold=4, max_comparisons=250000, max_pairs=2000):
    """Generate bounded dHash candidates for human review, never merge them automatically."""
    if not 0 <= threshold <= 8:
        raise ValueError('dHash threshold must be between 0 and 8')
    entries = [(split, row) for split, rows in splits.items() for row in rows]
    blocks = threshold + 1
    buckets, seen, pairs = defaultdict(list), set(), []
    comparisons = 0
    truncated = False
    for i, (split, row) in enumerate(entries):
        try:
            value = int(row['dhash'], 16)
            if len(row['dhash']) != 16:
                raise ValueError('invalid hash length')
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Invalid dHash for {row.get('image_id', i)}") from None
        for block in range(blocks):
            start, end = 64 * block // blocks, 64 * (block + 1) // blocks
            key = (block, (value >> start) & ((1 << (end-start)) - 1))
            for j, other_value in buckets[key]:
                if (j, i) in seen:
                    continue
                seen.add((j, i))
                comparisons += 1
                if comparisons > max_comparisons:
                    truncated = True
                    break
                other_split, other = entries[j]
                distance = (value ^ other_value).bit_count()
                if distance <= threshold and row['pixel_hash'] != other['pixel_hash']:
                    pairs.append({'image_id_a': other['image_id'], 'image_id_b': row['image_id'],
                                  'path_a':other['image_path'], 'path_b':row['image_path'],
                                  'split_a':other_split, 'split_b':split, 'distance':distance,
                                  'group_a':other.get('group_id', ''), 'group_b':row.get('group_id', '')})
                    if len(pairs) >= max_pairs:
                        truncated = True
                        break
            if truncated:
                break
            buckets[key].append((i, value))
        if truncated:
            break
    return {'method':'64-bit dHash, Hamming distance', 'threshold':threshold,
            'comparisons':min(comparisons,max_comparisons), 'truncated':truncated,
            'max_comparisons':max_comparisons, 'max_pairs':max_pairs, 'pairs':pairs,
            'note':'Candidates require human review; only confirmed group_id assignments constrain split.'}


def split_summary(splits, mapping):
    all_rows = [r for rows in splits.values() for r in rows]
    return {'type_style_counts':{f'{t} / {s}':n for (t,s),n in sorted(Counter((r['type'],r['style']) for r in all_rows).items())},
            'missing_holdout_classes':{split:{key:sorted(set(mapping[key])-set(r[key] for r in rows))
                                        for key in mapping} for split,rows in splits.items() if split != 'train'},
            'exact_duplicate_groups':sum(n > 1 for n in Counter(r['pixel_hash'] for r in all_rows).values()),
            'provided_groups':len({r['group_id'] for r in all_rows if r.get('group_id')})}


def partition(rows, split_config, previous_root=None):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i,j):
        parent[find(i)] = find(j)
    keys, image_labels = {}, {}
    for i,row in enumerate(rows):
        h = row['pixel_hash']
        label = (row['type'],row['style'])
        if image_labels.setdefault(h,label) != label:
            raise ValueError('Identical pixels have conflicting labels')
        for kind,value in [('image',h),('group',row.get('group_id'))]:
            if value:
                key = (kind,value)
                if key in keys:
                    union(i,keys[key])
                keys[key] = i
    groups = defaultdict(list)
    for i,row in enumerate(rows):
        groups[find(i)].append(row)
    previous = {}
    if previous_root:
        old_splits,_ = validate_splits(previous_root)
        for split,items in old_splits.items():
            for row in items:
                for kind,value in [('image',row['pixel_hash']),('group',row.get('group_id'))]:
                    if value:
                        previous[(kind,value)] = split
    strata = defaultdict(list)
    for group in groups.values():
        fixed = set()
        for row in group:
            if row.get('_split'):
                fixed.add(row['_split'])
            for kind,value in [('image',row['pixel_hash']),('group',row.get('group_id'))]:
                if (kind,value) in previous:
                    fixed.add(previous[(kind,value)])
        if not fixed <= {'train','val','test'}:
            raise ValueError(f'Invalid split labels: {fixed}')
        if len(fixed) > 1:
            raise ValueError('Group/duplicate bridges different splits; review source assignments')
        label_signature = tuple(sorted({(r['type'],r['style']) for r in group}))
        strata[label_signature].append((group,next(iter(fixed),None)))
    result = {s:[] for s in ['train','val','test']}
    rng = random.Random(split_config['seed'])
    rare = []
    for signature,groups in sorted(strata.items()):
        groups.sort(key=lambda pair:min(r['pixel_hash'] for r in pair[0]))
        fixed_count = Counter(split for _,split in groups if split)
        unfixed = [g for g,s in groups if not s]
        rng.shuffle(unfixed)
        n = len(groups)
        v = max(1,round(n*split_config['val_fraction'])) if n >= 3 else 0
        t = max(1,round(n*split_config['test_fraction'])) if n >= 3 else 0
        if v+t >= n and n >= 3:
            t = n-v-1
        if n < 3:
            rare.append({'labels':[list(x) for x in signature],'groups':n})
        for group,split in groups:
            if split:
                result[split].extend(group)
        for group in unfixed:
            if fixed_count['val'] < v:
                split = 'val'
            elif fixed_count['test'] < t:
                split = 'test'
            else:
                split = 'train'
            fixed_count[split] += 1
            result[split].extend(group)
    if any(not values for values in result.values()):
        raise ValueError('Not enough independent groups for non-empty train/val/test')
    # _split is an ingestion instruction, not an exported feature.
    clean = {s:[{k:v for k,v in row.items() if k != '_split'} for row in rs] for s,rs in result.items()}
    return clean, rare


def prepare_data(cfg, *, exclude_incomplete_labels=False):
    if not isinstance(exclude_incomplete_labels, bool):
        raise ValueError('exclude_incomplete_labels must be true/false')
    data = cfg['dataset']
    root = resolve_path(cfg,data['root'])
    source = data['source']
    if source['kind'] == 'prepared':
        if exclude_incomplete_labels:
            raise ValueError('Cannot apply import exclusions to an already prepared snapshot')
        return audit_data(cfg)
    if root.exists():
        raise ValueError('Dataset snapshot exists; choose a new dataset id/root')
    if source['kind'] == 'huggingface':
        revision = source['revision']
        if not isinstance(revision,str) or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision.lower()):
            raise ValueError('Set source.revision to a real 40-character Hugging Face commit SHA')
        from datasets import load_dataset
        dataset = load_dataset(source['dataset_id'], revision=revision, streaming=True)
        def records():
            for name, samples in dataset.items():
                for i,sample in enumerate(samples):
                    yield f'{name}-{i:07d}',name,sample
        iterator = records()
    else:
        with resolve_path(cfg,source['csv_file']).open(encoding='utf-8-sig',newline='') as stream:
            samples = list(csv.DictReader(stream))
        iterator = ((f'csv-{i:07d}','csv',sample) for i,sample in enumerate(samples))
    root.mkdir(parents=True)
    (root/'images').mkdir()
    rows,errors,excluded = [],[],[]
    source_counts = Counter()
    columns = source['columns']
    for image_id,source_split,sample in iterator:
        source_counts[source_split] += 1
        try:
            if source['kind'] == 'csv':
                path = Path(sample[columns['image_path']])
                path = path if path.is_absolute() else resolve_path(cfg,source['image_root'])/path
                with Image.open(path) as original:
                    image = rgb(original)
            else:
                image = rgb(sample[columns['image_path']])
            values = {key:sample[columns[key]] for key in ['type','style']}
            missing = [key for key,value in values.items() if value is None or (isinstance(value,str) and not value.strip())]
            if any(value is not None and not isinstance(value,str) for value in values.values()):
                raise TypeError('type/style label must be text or null')
            labels = {key:value.strip().lower() for key,value in values.items() if key not in missing}
            group = sample[columns['group_id']].strip() if columns['group_id'] else ''
            selected_split = sample[columns['split']].strip().lower() if columns['split'] else ''
            if selected_split and selected_split not in ('train','val','test'):
                raise ValueError(f'Invalid split label: {selected_split}')
            if source['kind'] == 'huggingface' and source_split.lower() == 'test':
                if selected_split and selected_split != 'test':
                    raise ValueError('Source test cannot be assigned to train or validation')
                selected_split = 'test'
            pixel_hash, perceptual_hash = image_hash(image), dhash(image)
            if missing:
                if not exclude_incomplete_labels:
                    raise ValueError(f'Empty type/style label: {", ".join(missing)}')
                excluded.append({'image_id':image_id, 'source_split':source_split,
                                 'reason':'missing_required_label', 'missing_labels':missing})
                continue
            row = {'image_id':image_id,'image_path':f'images/{image_id}.png',**labels,
                   'source_split':source_split,'pixel_hash':pixel_hash,'dhash':perceptual_hash,
                   'group_id':group,'_split':selected_split}
            image.save(root/row['image_path'])
            rows.append(row)
        except (KeyError,ValueError,OSError,AttributeError,TypeError) as error:
            errors.append({'image_id':image_id,'source_split':source_split,'error':str(error)})
    exclusions = {'policy':'exclude_incomplete_labels' if exclude_incomplete_labels else 'fail_on_incomplete_labels',
                  'source_counts':dict(source_counts), 'excluded_count':len(excluded),
                  'excluded_by_source_split':dict(Counter(row['source_split'] for row in excluded)),
                  'excluded_by_label':dict(Counter(label for row in excluded for label in row['missing_labels'])),
                  'excluded':excluded}
    (root/'excluded_samples.json').write_text(json.dumps(exclusions,indent=2,ensure_ascii=False),encoding='utf-8')
    if errors:
        (root/'import_errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
        raise ValueError(f'{len(errors)} import errors; see import_errors.json; correct source and use a new root')
    previous = data['split']['previous_root']
    splits,rare = partition(rows,data['split'],resolve_path(cfg,previous) if previous else None)
    for split,items in splits.items():
        write_rows(root/f'{split}.csv',items)
    splits,mapping = validate_splits(root)
    split_hashes = {s:hashlib.sha256((root/f'{s}.csv').read_bytes()).hexdigest() for s in splits}
    duplicate_review = near_duplicate_candidates(splits)
    (root/'near_duplicate_candidates.json').write_text(json.dumps(duplicate_review,indent=2),encoding='utf-8')
    old_mapping = validate_splits(resolve_path(cfg,previous))[1] if previous else None
    changes = {k:{'added':sorted(set(mapping[k])-set(old_mapping[k])),
                  'removed':sorted(set(old_mapping[k])-set(mapping[k]))} for k in mapping} if old_mapping else None
    metadata = {'dataset_id':data['id'], 'source':source, 'split_hashes':split_hashes, 'import_exclusions':exclusions,
                'class_mapping':mapping, 'class_changes':changes,
                'class_mapping_sha256':hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()}
    (root/'dataset_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    report = {'dataset_id':data['id'],'source':source,'split_policy':data['split'],
              'rows':len(rows),'counts':{s:len(rs) for s,rs in splits.items()},'rare_groups':rare,
              'import_exclusions':exclusions,
              'labels':{key:dict(Counter(r[key] for r in rows)) for key in ['type','style']},
              **split_summary(splits,mapping), 'split_hashes':split_hashes,
              'class_changes':changes, 'near_duplicate_review':'See near_duplicate_candidates.json; confirm candidates manually and populate group_id',
              'split_policy_note':'Group-aware approximate fractions; explicit and previous assignments take precedence; Hugging Face source test stays test'}
    (root/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def audit_data(cfg):
    root = resolve_path(cfg,cfg['dataset']['root'])
    splits,mapping = validate_splits(root)
    hashes = {s:hashlib.sha256((root/f'{s}.csv').read_bytes()).hexdigest() for s in splits}
    exclusion_file = root/'excluded_samples.json'
    exclusions = json.loads(exclusion_file.read_text(encoding='utf-8')) if exclusion_file.exists() else None
    return {'dataset_id':cfg['dataset']['id'],'counts':{s:len(rs) for s,rs in splits.items()},
            'mapping':mapping,'split_hashes':hashes,
            'import_exclusions':exclusions,
            'label_counts':{s:{k:dict(Counter(r[k] for r in rs)) for k in mapping} for s,rs in splits.items()},
            **split_summary(splits,mapping), 'near_duplicate_candidates':near_duplicate_candidates(splits),
            'near_duplicate_review':'Exact duplicates and provided group_id checked; visually similar candidates require review'}
