"""Room inference: an existing detector proposes boxes for one shared classifier."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import yaml

from furniture.data import rgb
from furniture.recommend import canonical_type
from scripts.model_inference import FurniturePredictor
from scripts.release_evidence import bundle
from scripts.room_crop import prepare_crop


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def detect_and_classify(image_path, detector, predictor, confidence=.25, top_k=3, *, image_id=None,
                        class_mapping=None, padding_fraction=.05, min_size=2, imgsz=640,
                        provenance=None, crop_dir=None):
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Detector confidence must be between 0 and 1')
    if not math.isfinite(padding_fraction) or not 0 <= padding_fraction <= 1 or min_size < 1:
        raise ValueError('Invalid crop padding or minimum size')
    image_id = str(image_id or Path(image_path).stem)
    report = {'image_id':image_id, 'image_path':str(image_path), 'objects':[],
              'provenance':provenance or {}, 'crop_padding_fraction':padding_fraction,
              'coordinate_system':'xyxy in EXIF-oriented image; right/bottom exclusive',
              'note':'Detector and classifier scores are separate; classifier softmax is not calibrated compatibility.'}
    try:
        with Image.open(image_path) as source:
            image = rgb(source)
    except (OSError, UnidentifiedImageError, ValueError) as error:
        report.update(status='image_error', error=f'{type(error).__name__}: {error}', image_size=None)
        return report
    report['image_size'] = {'width':image.width, 'height':image.height}
    if class_mapping is None:
        # Legacy callers use the existing alias function; CLI supplies the explicit mapping config.
        class_mapping = {canonical_type(name):canonical_type(name) for name in predictor.mapping['type']}
        class_mapping.update({'couch':'sofa','dining table':'table'})
    if not isinstance(class_mapping, dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in class_mapping.items()):
        raise ValueError('Detector-to-category mapping must be a string-to-string object')
    results = detector.predict(source=np.asarray(image)[:, :, ::-1].copy(), conf=confidence,
                               device=str(predictor.device), verbose=False, imgsz=imgsz)
    boxes = results[0].boxes
    report['status'] = 'no_detections' if len(boxes) == 0 else 'ok'
    if crop_dir is not None:
        Path(crop_dir).mkdir(parents=True, exist_ok=True)
    for index, box in enumerate(boxes):
        label = results[0].names[int(box.cls.item())]
        raw = [float(v) for v in box.xyxy[0].cpu().tolist()]
        item = {'image_id':image_id, 'object_id':str(index), 'image_size':report['image_size'],
                'detector_label':label, 'detector_confidence':float(box.conf.item()),
                'detector_type':label, 'raw_bbox_xyxy':[v if math.isfinite(v) else None for v in raw], 'bbox_xyxy':None, 'crop_xyxy':None,
                'category':None, 'category_confidence':None, 'category_top_k':[],
                'style':None, 'style_confidence':None, 'style_top_k':[], 'provenance':report['provenance']}
        report['objects'].append(item)
        try:
            detector_box, crop_box, crop = prepare_crop(image,raw,padding_fraction=padding_fraction,min_size=min_size)
            item['bbox_xyxy'] = detector_box
            item['crop_xyxy'] = crop_box
            mapped = class_mapping.get(label)
            item['mapped_detector_category'] = mapped
            if mapped not in predictor.mapping['type']:
                item.update(status='unmapped_detector_label', error='Detector label has no supported category mapping')
                continue
            if crop_dir is not None:
                crop_name = f'{index:04d}.png'
                crop.save(Path(crop_dir)/crop_name)
                item['crop_file'] = f'crops/{crop_name}'
            prediction = predictor.predict_image(crop, top_k)
            item['type'] = prediction['type']  # Backward compatible with prior callers.
            item['style_prediction'] = prediction['style']
            item.update(status='ok', category=prediction['type']['label'], category_confidence=prediction['type']['confidence'],
                        category_top_k=prediction['type']['top_k'], style=prediction['style']['label'],
                        style_confidence=prediction['style']['confidence'], style_top_k=prediction['style']['top_k'],
                        category_disagreement=(mapped != prediction['type']['label']))
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            item.update(status='crop_error', error=f'{type(error).__name__}: {error}')
    return report


def export_room(report, image_path, output, save_annotated=True):
    output = Path(output)
    (output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    fields = ['image_id','object_id','status','error','image_size','bbox_xyxy','crop_xyxy','detector_label',
              'detector_confidence','mapped_detector_category','category','category_confidence','category_top_k',
              'category_disagreement','style','style_confidence','style_top_k','provenance']
    with (output/'objects.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in report['objects']:
            row = {k:item.get(k, '') for k in fields}
            for key in ('image_size','bbox_xyxy','crop_xyxy','category_top_k','style_top_k','provenance'):
                row[key] = json.dumps(row[key], ensure_ascii=False)
            writer.writerow(row)
    if save_annotated and report['status'] != 'image_error':
        with Image.open(image_path) as source:
            image = rgb(source)
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype('arial.ttf', max(12, image.width//75))
        except OSError:
            font = ImageFont.load_default()
        for item in report['objects']:
            if item['bbox_xyxy']:
                color = 'gray' if item['status'] != 'ok' else 'red' if item.get('category_disagreement') else 'lime'
                draw.rectangle(item['bbox_xyxy'], outline=color, width=3)
                label = f"{item['object_id']}: {item.get('category') or item['detector_label']} / {item.get('style') or item['status']}"
                x,y = item['bbox_xyxy'][:2]
                bounds = draw.textbbox((x,y), label, font=font)
                draw.rectangle(bounds, fill='black')
                draw.text((x,y), label, fill='white', font=font)
        image.save(output/'annotated.png')


def main():
    parser = argparse.ArgumentParser(description='Room image: detector -> crops -> multi-task classifier')
    parser.add_argument('--image', required=True)
    parser.add_argument('--image-id')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--detector-weights', help='Existing local detector checkpoint; never automatically downloaded')
    parser.add_argument('--config', default='configs/inference/room-yolov8n-coco-baseline.yaml')
    parser.add_argument('--output', required=True, help='New output directory')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--save-crops', action='store_true')
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    weights = Path(args.detector_weights or config['detector']['weights'])
    if not weights.is_file():
        raise FileNotFoundError('Provide an existing local detector checkpoint')
    detector_hash = file_hash(weights)
    if not args.detector_weights and detector_hash != config['detector']['sha256']:
        raise ValueError('Configured detector checkpoint hash differs')
    from ultralytics import YOLO
    predictor = FurniturePredictor(args.checkpoint, args.device)
    detector = YOLO(str(weights))
    provenance = {'detector_checkpoint':str(weights.resolve()), 'detector_sha256':detector_hash,
                  'classifier_checkpoint':str(Path(args.checkpoint).resolve()), 'classifier_sha256':file_hash(args.checkpoint),
                  'config':str(config_path.resolve()), 'config_sha256':file_hash(config_path), 'config_version':config['version']}
    image_hash = file_hash(args.image) if Path(args.image).is_file() else None
    context = {'image_id':args.image_id or Path(args.image).stem, 'image_path':str(Path(args.image).resolve()),
               'image_sha256':image_hash, 'provenance':provenance,
               'top_k':args.top_k, 'save_crops':args.save_crops, 'device':str(predictor.device)}
    def build(path):
        result = detect_and_classify(args.image, detector, predictor, config['detector']['confidence'], args.top_k,
            image_id=context['image_id'], class_mapping=config['detector_to_category'],
            padding_fraction=config['crop']['padding_fraction'], min_size=config['crop']['min_size'],
            imgsz=config['detector']['imgsz'], provenance=provenance, crop_dir=path/'crops' if args.save_crops else None)
        export_room(result, args.image, path)
    bundle(args.output, context, ['result.json','objects.csv'], build)
    result = json.loads((Path(args.output)/'result.json').read_text(encoding='utf-8'))
    print(json.dumps({'output':args.output, 'status':result['status'], 'detections':len(result['objects'])}, ensure_ascii=False))
    if result['status'] == 'image_error':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
