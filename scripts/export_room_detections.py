"""Export detector boxes for every image, including images with no annotated objects."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from furniture.data import rgb
from scripts.release_evidence import bundle, sha
from scripts.room_evaluation import read


def export(images, image_root, detector, output, *, split, confidence, imgsz, device):
    root = Path(image_root).resolve()
    seen, records = set(), []
    for row in images:
        if row.get('split') != split:
            continue
        image_id = row.get('image_id')
        if not image_id or image_id in seen:
            raise ValueError('Image IDs must be unique')
        seen.add(image_id)
        path = (root/row['image_path']).resolve()
        if not path.is_relative_to(root) or sha(path) != row['sha256']:
            raise ValueError('Image path or hash differs from manifest')
        with Image.open(path) as source:
            image = rgb(source)
        result = detector.predict(source=np.asarray(image)[:,:,::-1].copy(), conf=confidence,
                                  imgsz=imgsz, device=device, verbose=False)[0]
        for box in result.boxes:
            coords = [float(v) for v in box.xyxy[0].cpu().tolist()]
            records.append({'image_id':image_id,'detector_type':result.names[int(box.cls.item())],
                            'confidence':float(box.conf.item()), **dict(zip(('x1','y1','x2','y2'),coords))})
    if not seen:
        raise ValueError('No images in selected split')
    with Path(output).open('x',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['image_id','detector_type','confidence','x1','y1','x2','y2'])
        writer.writeheader()
        writer.writerows(records)
    return {'images':len(seen),'detections':len(records)}


def main():
    parser = argparse.ArgumentParser()
    for name in ('images','image-root','weights','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--split',choices=['development','final'],required=True)
    parser.add_argument('--confidence',type=float,default=.05)
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1 or args.imgsz < 1 or not Path(args.weights).is_file():
        raise ValueError('Invalid detector weights, confidence or image size')
    from ultralytics import YOLO
    detector = YOLO(args.weights)
    context = {'images_sha256':sha(args.images),'weights_sha256':sha(args.weights),'split':args.split,
               'confidence':args.confidence,'imgsz':args.imgsz,'device':args.device}
    def build(path):
        output = path/'detections.csv'
        result = export(read(args.images),args.image_root,detector,output,split=args.split,
                        confidence=args.confidence,imgsz=args.imgsz,device=args.device)
        result.update(context,detections_sha256=sha(output))
        (path/'detections.csv.metadata.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    bundle(args.output,context,['detections.csv','detections.csv.metadata.json'],build)
    print(json.dumps({'output':args.output,'status':'complete'}))


if __name__ == '__main__':
    main()
