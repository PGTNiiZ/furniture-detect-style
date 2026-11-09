"""Evaluate independent room annotations, ground-truth crops, and detector crops."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image
from furniture.data import image_hash, rgb
from furniture.room_evaluation import _box, _iou
from furniture.robustness import task_metrics
from scripts.model_inference import FurniturePredictor
from scripts.release_evidence import bundle, sha
from scripts.room_crop import prepare_crop


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def evaluate(images, objects, detections, image_root, predictor, protocol, split='final'):
    if (protocol.get('version') != 'room-eval-v1' or protocol.get('status') != 'locked'
            or not protocol.get('locked_at_utc') or not protocol.get('development_reference')
            or protocol.get('independent_from_classifier_training') is not True
            or protocol.get('matching') != 'class-agnostic greedy one-to-one by descending IoU'):
        raise ValueError('A locked room-eval-v1 protocol is required')
    confidence, iou = protocol['confidence'], protocol['iou_threshold']
    padding, mapping, min_size = protocol['crop_padding_fraction'], protocol['detector_to_category'], protocol['min_crop_size']
    if not (0 <= confidence <= 1 and 0 < iou <= 1 and 0 <= padding <= 1) or not isinstance(min_size,int) or isinstance(min_size,bool) or min_size < 1:
        raise ValueError('Invalid room evaluation thresholds or padding')
    if not isinstance(mapping, dict) or not set(mapping.values()) <= set(predictor.mapping['type']):
        raise ValueError('Detector category mapping is inconsistent with classifier labels')
    exposure = getattr(predictor, 'exposure_membership', None)
    if not isinstance(exposure, dict) or any(not isinstance(exposure.get(s, {}).get('pixel_hashes'), list) for s in ('train','val','test')):
        raise ValueError('Classifier checkpoint lacks exposure membership for room independence checks')
    exposed_hashes = {h for s in ('train','val','test') for h in exposure[s]['pixel_hashes']}
    image_rows, rooms, contents = {}, {}, {}
    for row in images:
        if not all(row.get(k) for k in ('image_id','image_path','room_id','split','sha256')) or row['image_id'] in image_rows:
            raise ValueError('Image manifest requires unique IDs, paths, room IDs, split and SHA-256')
        if rooms.setdefault(row['room_id'],row['split']) != row['split']:
            raise ValueError('Room crosses development/final splits')
        if contents.setdefault(row['sha256'],row['split']) != row['split']:
            raise ValueError('Exact image crosses development/final splits')
        image_rows[row['image_id']] = row
    selected = {k:v for k,v in image_rows.items() if v['split'] == split}
    if not selected:
        raise ValueError('No images in requested split')
    grouped, seen = defaultdict(list), set()
    for row in objects:
        key = (row.get('image_id'),row.get('object_id'))
        if key[0] not in image_rows or not key[1] or key in seen or not row.get('type') or 'style' not in row:
            raise ValueError('Invalid, duplicate or unlinked room object annotation')
        seen.add(key)
        if key[0] in selected:
            grouped[key[0]].append(row)
    detected = defaultdict(list)
    for row in detections:
        if row.get('image_id') not in image_rows:
            raise ValueError('Detection image ID is absent from image manifest')
        if row['image_id'] in selected:
            detected[row['image_id']].append(row)
    totals = {'matches':0,'misses':0,'false_positives':0,'unmapped_detections':0,
              'invalid_or_small_crops':0,'matched_category_disagreements':0}
    samples, misses, false_positives, rejected_detections = [], [], [], []
    for image_id, info in selected.items():
        root = Path(image_root).resolve()
        path = (root/info['image_path']).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError('Image path or SHA-256 does not match manifest')
        with Image.open(path) as raw:
            image = rgb(raw)
        if image_hash(image) in exposed_hashes:
            raise ValueError('Room image overlaps classifier dataset exposure')
        refs = [(row,_box(row,*image.size)) for row in grouped[image_id]]
        dets = []
        for row in detected[image_id]:
            score = float(row['confidence'])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid detector confidence')
            if score < confidence:
                continue
            if row['detector_type'] not in mapping:
                totals['unmapped_detections'] += 1
                continue
            try:
                raw = [float(row[k]) for k in ('x1','y1','x2','y2')]
                clipped, crop_box, crop_image = prepare_crop(image,raw,padding_fraction=padding,min_size=min_size)
            except (KeyError, TypeError, ValueError) as error:
                totals['invalid_or_small_crops'] += 1
                rejected_detections.append({'image_id':image_id,'detector_type':row.get('detector_type'),
                                            'raw_bbox_xyxy':[row.get(k) for k in ('x1','y1','x2','y2')],
                                            'error':str(error)})
                continue
            dets.append((row,tuple(clipped),crop_box,crop_image))
        candidates = sorted(((_iou(rb,db),i,j) for i,(_,rb) in enumerate(refs)
                             for j,(_,db,_,_) in enumerate(dets)), reverse=True)
        matches, used = {}, set()
        for overlap,i,j in candidates:
            if overlap >= iou and i not in matches and j not in used:
                matches[i] = (j,overlap); used.add(j)
        for index,(row,ref_box) in enumerate(refs):
            _, reference_crop_box, reference_crop = prepare_crop(image,ref_box,padding_fraction=padding,min_size=1)
            ref = predictor.predict_image(reference_crop)
            record = {'image_id':image_id,'object_id':row['object_id'],'true_type':row['type'],
                      'true_style':row['style'],'reference_type':ref['type']['label'],
                      'reference_style':ref['style']['label'],'reference_bbox_xyxy':list(ref_box),
                      'reference_crop_xyxy':reference_crop_box,'matched':index in matches}
            if index in matches:
                j,overlap = matches[index]
                pred = predictor.predict_image(dets[j][3])
                disagreement = mapping[dets[j][0]['detector_type']] != row['type']
                totals['matched_category_disagreements'] += int(disagreement)
                record.update(iou=overlap, detector_type=dets[j][0]['detector_type'],
                              detector_category_disagreement=disagreement,
                              detector_bbox_xyxy=list(dets[j][1]),
                              detector_crop_xyxy=dets[j][2],
                              detected_type=pred['type']['label'], detected_style=pred['style']['label'])
            else:
                misses.append({'image_id':image_id,'object_id':row['object_id'],'bbox_xyxy':list(ref_box)})
            samples.append(record)
        for j,(row,bounds,crop_box,_) in enumerate(dets):
            if j not in used:
                false_positives.append({'image_id':image_id,'detector_type':row['detector_type'],
                                        'confidence':float(row['confidence']),'bbox_xyxy':list(bounds),'crop_xyxy':crop_box})
        totals['matches'] += len(matches)
        totals['misses'] += len(refs)-len(matches)
        totals['false_positives'] += len(dets)-len(used)
    total_objects = totals['matches']+totals['misses']
    def task_report(task, detector_crop):
        classes = predictor.mapping[task]
        names = [k for k,_ in sorted(classes.items(),key=lambda x:x[1])]
        eligible = [r for r in samples if r[f'true_{task}'] in classes and (r['matched'] or not detector_crop)]
        truth = [classes[r[f'true_{task}']] for r in eligible]
        predicted = [classes[r[f'{"detected" if detector_crop else "reference"}_{task}']] for r in eligible]
        known_total = sum(r[f'true_{task}'] in classes for r in samples)
        return {'eligible':len(eligible),'denominator_objects':total_objects,
                'coverage_of_objects':len(eligible)/total_objects if total_objects else None,
                'known_reference_total':known_total,'coverage_of_known':len(eligible)/known_total if known_total else None,
                'missing_labels':sum(not r[f'true_{task}'] for r in samples),
                'unseen_labels':sum(bool(r[f'true_{task}']) and r[f'true_{task}'] not in classes for r in samples),
                'metrics':task_metrics(truth,predicted,names) if eligible else None}
    return {'split':split,'images':len(selected),'objects':total_objects,'detection':{**totals,
            'recall':totals['matches']/total_objects if total_objects else None,
            'precision':totals['matches']/(totals['matches']+totals['false_positives']) if totals['matches']+totals['false_positives'] else None,
            'matching':'class-agnostic greedy one-to-one by descending IoU'},
            'reference_crops':{t:task_report(t,False) for t in ('type','style')},
            'matched_detector_crops':{t:task_report(t,True) for t in ('type','style')},
            'predictions':samples,'misses':misses,'false_positives':false_positives,
            'rejected_detections':rejected_detections,'protocol':protocol,
            'limitations':['Matched-crop accuracy excludes missed objects; see detection recall and coverage.',
                           'Unmapped labels and invalid/small boxes are excluded and counted separately; this is not mAP.',
                           'Annotations must be independent and exhaustive; model predictions are not ground truth.']}


def main():
    parser = argparse.ArgumentParser(description='Evaluate separately annotated room images')
    for option in ('images','objects','detections','image-root','checkpoint','detector-checkpoint','protocol','output'):
        parser.add_argument('--'+option, required=True)
    parser.add_argument('--split', choices=['development','final'], default='final')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text(encoding='utf-8'))
    sidecar = Path(args.detections + '.metadata.json')
    metadata = json.loads(sidecar.read_text(encoding='utf-8'))
    if (metadata.get('detections_sha256') != sha(args.detections) or metadata.get('weights_sha256') != sha(args.detector_checkpoint)
            or metadata.get('images_sha256') != sha(args.images) or metadata.get('split') != args.split
            or metadata.get('confidence',1) > protocol['confidence']):
        raise ValueError('Detector predictions do not match detector checkpoint and metadata')
    predictor = FurniturePredictor(args.checkpoint,args.device)
    context = {'image_manifest_sha256':sha(args.images),'object_manifest_sha256':sha(args.objects),
               'detections_sha256':sha(args.detections),'detector_sha256':sha(args.detector_checkpoint),
               'classifier_sha256':sha(args.checkpoint),'protocol_sha256':sha(args.protocol),'split':args.split}
    def build(path):
        result = evaluate(read(args.images),read(args.objects),read(args.detections),args.image_root,predictor,protocol,args.split)
        result['evidence'] = context
        with (path/'metrics.json').open('x',encoding='utf-8') as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2)
        fields = ['image_id','object_id','true_type','true_style','reference_type','reference_style','reference_bbox_xyxy',
                  'reference_crop_xyxy','matched','iou','detector_type','detector_bbox_xyxy','detector_crop_xyxy',
                  'detector_category_disagreement','detected_type','detected_style']
        with (path/'object-predictions.csv').open('x',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            writer.writerows(result['predictions'])
    bundle(args.output,context,['metrics.json','object-predictions.csv'],build)
    print(json.dumps({'output':args.output,'status':'complete'}))


if __name__ == '__main__':
    main()
