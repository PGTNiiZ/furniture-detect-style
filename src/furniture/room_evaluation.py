"""Evaluate annotated room objects and externally recorded detector predictions."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image

from .data import read_rows, rgb
from .room import classify_crop
from .train import choose_device, load_model, metrics_for


def _box(row, width, height):
    try:
        box = tuple(float(row[k]) for k in ('x1', 'y1', 'x2', 'y2'))
    except (KeyError, ValueError, TypeError):
        raise ValueError('Bounding boxes require numeric x1,y1,x2,y2') from None
    if not all(math.isfinite(v) for v in box) or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError(f'Invalid bounding box {box} for image {width}x{height}')
    return box


def _iou(a, b):
    intersection = max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0, min(a[3],b[3])-max(a[1],b[1]))
    return intersection / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)


def _crop(image, box):
    return image.crop((math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])))


def export_room_detections(annotations_csv, image_root, weights, output_csv, device='auto',
                           confidence=.05, split='test'):
    """Run Ultralytics on each annotated image and export original-coordinate boxes."""
    import numpy as np
    from ultralytics import YOLO
    if not 0 <= confidence <= 1:
        raise ValueError('confidence must be between 0 and 1')
    output = Path(output_csv)
    metadata_path = output.with_suffix(output.suffix+'.metadata.json')
    if output.exists() or metadata_path.exists():
        raise ValueError('Detection output already exists')
    root = Path(image_root).resolve()
    images, room_splits = {}, {}
    for row in read_rows(annotations_csv):
        if not all(row.get(k) for k in ('image_id','image_path','room_id','split')):
            raise ValueError('Annotations require image_id, image_path, room_id and split')
        if room_splits.setdefault(row['room_id'], row['split']) != row['split']:
            raise ValueError('Room group crosses splits')
        if row['split'] == split:
            if images.setdefault(row['image_id'], row['image_path']) != row['image_path']:
                raise ValueError('Image ID has inconsistent paths')
    if not images:
        raise ValueError(f'No annotated images in split {split}')
    paths = {key:(root/value).resolve() for key,value in images.items()}
    if any(not p.is_relative_to(root) or not p.is_file() for p in paths.values()):
        raise ValueError('Room image path must be an existing file inside image_root')
    selected_device = choose_device(device)
    detector = YOLO(weights)
    records = []
    for image_id,path in paths.items():
        with Image.open(path) as raw:
            image = rgb(raw)
        predictions = detector.predict(source=np.asarray(image)[:,:,::-1].copy(), conf=confidence,
                                       device=str(selected_device), verbose=False)[0]
        for box in predictions.boxes:
            x1,y1,x2,y2 = [float(v) for v in box.xyxy[0].cpu().tolist()]
            records.append({'image_id':image_id,'detector_type':predictions.names[int(box.cls.item())],
                            'confidence':float(box.conf.item()),'x1':x1,'y1':y1,'x2':x2,'y2':y2})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,['image_id','detector_type','confidence','x1','y1','x2','y2'])
        writer.writeheader()
        writer.writerows(records)
    result = {'output_csv':str(output), 'images':len(images), 'detections':len(records),
              'weights':str(weights), 'weights_sha256':hashlib.sha256(Path(weights).read_bytes()).hexdigest() if Path(weights).is_file() else None,
              'annotations_sha256':hashlib.sha256(Path(annotations_csv).read_bytes()).hexdigest(),
              'detections_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
              'confidence':confidence, 'split':split, 'device':str(selected_device)}
    metadata_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def evaluate_room_domain(annotations_csv, detections_csv, image_root, checkpoint, output,
                         class_mapping_json, device='auto', confidence=.25, min_crop_size=16,
                         iou_threshold=.5, split='test'):
    """Evaluate known boxes and matched detector boxes; unmatched detections are false positives.

    Input detections are produced separately by the chosen detector with the original image
    coordinates. Matching is class-aware greedy matching by descending IoU.
    """
    output = Path(output)
    if output.exists():
        raise ValueError('Output already exists')
    if not (0 <= confidence <= 1 and 0 < iou_threshold <= 1 and min_crop_size >= 1):
        raise ValueError('Invalid detector confidence, IoU or minimum crop size')
    annotations = read_rows(annotations_csv)
    detections = read_rows(detections_csv)
    if not annotations:
        raise ValueError('No room annotations')
    sidecar = Path(detections_csv).with_suffix(Path(detections_csv).suffix+'.metadata.json')
    detector_metadata = json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.exists() else None
    if detector_metadata:
        expected_annotations = hashlib.sha256(Path(annotations_csv).read_bytes()).hexdigest()
        expected_detections = hashlib.sha256(Path(detections_csv).read_bytes()).hexdigest()
        if detector_metadata.get('annotations_sha256') != expected_annotations or detector_metadata.get('detections_sha256') != expected_detections:
            raise ValueError('Detection metadata hashes differ from input files')
        if detector_metadata.get('split') != split or confidence < detector_metadata.get('confidence', 1):
            raise ValueError('Detection export split or confidence is incompatible with evaluation')
    class_mapping = json.loads(Path(class_mapping_json).read_text(encoding='utf-8'))
    if not isinstance(class_mapping, dict) or not class_mapping or not all(isinstance(k,str) and k and isinstance(v,str) and v for k,v in class_mapping.items()):
        raise ValueError('Detector class mapping must be a non-empty string-to-string object')
    device = choose_device(device)
    model, payload = load_model(checkpoint, device)
    mapping = payload['mapping']
    if set(mapping) != {'type','style'} or not set(class_mapping.values()) <= set(mapping['type']):
        raise ValueError('Room evaluation requires both heads and mapped detector types from checkpoint')
    room_splits, room_paths, image_rooms, seen_objects = {}, {}, {}, set()
    for row in annotations:
        if not all(row.get(k) for k in ('room_id','image_id','image_path','object_id','split','type','style')):
            raise ValueError('Each annotation needs room/image/object IDs, path, split and type/style')
        if room_splits.setdefault(row['room_id'], row['split']) != row['split']:
            raise ValueError('Room group crosses splits')
        if room_paths.setdefault(row['image_id'], row['image_path']) != row['image_path']:
            raise ValueError('Image ID has inconsistent paths')
        if image_rooms.setdefault(row['image_id'], row['room_id']) != row['room_id']:
            raise ValueError('Image ID belongs to multiple rooms')
        if row['object_id'] in seen_objects:
            raise ValueError('Duplicate object_id')
        seen_objects.add(row['object_id'])
        if row['type'] not in mapping['type'] or row['style'] not in mapping['style']:
            raise ValueError('Annotation labels not in checkpoint mapping')
    selected = [r for r in annotations if r['split'] == split]
    if not selected:
        raise ValueError(f'No annotations in split {split}')
    selected_ids = {r['image_id'] for r in selected}
    if any(r['image_id'] in selected_ids and r['split'] != split for r in annotations):
        raise ValueError('Image group crosses splits')
    by_image = defaultdict(list)
    for row in selected:
        by_image[row['image_id']].append(row)
    detector_by_image = defaultdict(list)
    for row in detections:
        if not row.get('image_id') or row['image_id'] not in room_paths:
            raise ValueError('Detection image_id is absent from annotations')
        if row['image_id'] in selected_ids:
            detector_by_image[row['image_id']].append(row)
    truth = {k:[] for k in mapping}
    pred_reference = {k:[] for k in mapping}
    matched_truth = {k:[] for k in mapping}
    pred_detector = {k:[] for k in mapping}
    predictions, misses, false_positives = [], [], []
    excluded = {'low_confidence':0, 'unmapped_class':0, 'small_crop':0}
    tp = 0
    for image_id, objects in by_image.items():
        root = Path(image_root).resolve()
        path = (root/room_paths[image_id]).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Room image path must be inside image_root')
        with Image.open(path) as raw:
            image = rgb(raw)
        reference = [(r, _box(r, *image.size)) for r in objects]
        detected = []
        for row in detector_by_image[image_id]:
            box = _box(row, *image.size)
            try:
                score = float(row['confidence'])
            except (ValueError, KeyError, TypeError):
                raise ValueError('Detection confidence must be numeric') from None
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Detection confidence must be between 0 and 1')
            label = row.get('detector_type', '')
            if label not in class_mapping:
                excluded['unmapped_class'] += 1
            elif score < confidence:
                excluded['low_confidence'] += 1
            elif box[2]-box[0] < min_crop_size or box[3]-box[1] < min_crop_size:
                excluded['small_crop'] += 1
            else:
                detected.append((row, box, class_mapping[label], score))
        candidates = sorted(((_iou(b, db), i, j) for i,(r,b) in enumerate(reference)
                             for j,(_,db,label,_) in enumerate(detected) if r['type'] == label), reverse=True)
        assignments, used = {}, set()
        for overlap, i, j in candidates:
            if overlap >= iou_threshold and i not in assignments and j not in used:
                assignments[i] = (j, overlap)
                used.add(j)
        for i, (row, box) in enumerate(reference):
            ref_scores = classify_crop(model, payload, _crop(image,box), device)
            ref_prediction = {k:max(ref_scores[k],key=ref_scores[k].get) for k in mapping}
            result = {'image_id':image_id,'object_id':row['object_id'],'true_type':row['type'],'true_style':row['style'],
                      **{f'reference_{k}':ref_prediction[k] for k in mapping},
                      **{f'reference_{k}_confidence':ref_scores[k][ref_prediction[k]] for k in mapping},
                      'matched':i in assignments}
            for k in mapping:
                truth[k].append(mapping[k][row[k]])
                pred_reference[k].append(mapping[k][ref_prediction[k]])
            if i in assignments:
                j, overlap = assignments[i]
                detection, detector_box, _, detector_score = detected[j]
                scores = classify_crop(model,payload,_crop(image,detector_box),device)
                result.update({'detector_type':detection['detector_type'], 'detector_confidence':detector_score,
                               'detector_box':list(detector_box), 'iou':overlap,
                               **{f'detected_{k}':max(scores[k],key=scores[k].get) for k in mapping},
                               **{f'detected_{k}_confidence':max(scores[k].values()) for k in mapping}})
                for k in mapping:
                    matched_truth[k].append(mapping[k][row[k]])
                    pred_detector[k].append(mapping[k][result[f'detected_{k}']])
                tp += 1
            else:
                misses.append({'image_id':image_id,'object_id':row['object_id'],'type':row['type'],'bbox':list(box)})
            predictions.append(result)
        for j,(row,box,_,score) in enumerate(detected):
            if j not in used:
                false_positives.append({'image_id':image_id,'detector_type':row['detector_type'],'confidence':score,'bbox':list(box)})
    report = {'split':split, 'objects':len(selected), 'images':len(by_image),
              'reference_crops':metrics_for(truth,pred_reference,mapping),
              'matched_detector_crops':metrics_for(matched_truth,pred_detector,mapping) if tp else None,
              'detection':{'matched':tp,'misses':len(misses),'false_positives':len(false_positives),
                           'recall':tp/len(selected), 'precision':tp/(tp+len(false_positives)) if tp+len(false_positives) else None,
                           'excluded':excluded,'matching':'class-aware greedy descending IoU'},
              'misses':misses,'false_positives':false_positives,'class_mapping':class_mapping,
              'detector_export_metadata':detector_metadata,
              'settings':{'iou_threshold':iou_threshold,'confidence':confidence,'min_crop_size':min_crop_size,
                          'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                          'annotations_sha256':hashlib.sha256(Path(annotations_csv).read_bytes()).hexdigest(),
                          'detections_sha256':hashlib.sha256(Path(detections_csv).read_bytes()).hexdigest()},
              'limitations':['Detector-crop scores include matched objects only; misses are separate.',
                             'Unsupported classes are excluded and counted; this is not mAP.',
                             'All evaluated images must be annotated exhaustively for false-positive counts.',
                             'Images without annotated objects are not represented or evaluated.',
                             'External detection CSV without metadata cannot verify export threshold or provenance.']}
    output.mkdir(parents=True,exist_ok=False)
    (output/'room_metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    fields = ['image_id','object_id','true_type','true_style','reference_type','reference_style',
              'reference_type_confidence','reference_style_confidence','matched','detector_type',
              'detector_confidence','detector_box','iou','detected_type','detected_style',
              'detected_type_confidence','detected_style_confidence']
    with (output/'room_predictions.csv').open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fields)
        writer.writeheader()
        writer.writerows(predictions)
    return report
