"""Load one trained classifier and classify images without modifying training code."""
import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms

from furniture.data import Square
from furniture.train import choose_device, load_model


class FurniturePredictor:
    """Inference for the two-head checkpoint format produced by furniture.train."""

    def __init__(self, checkpoint, device='auto'):
        self.device = choose_device(device)
        self.model, payload = load_model(checkpoint, self.device)
        self.mapping = payload.get('mapping', {})
        self.preprocessing = payload.get('preprocessing')
        self.exposure_membership = payload.get('exposure_membership')
        if set(self.mapping) != {'type', 'style'}:
            raise ValueError('A checkpoint with both type and style heads is required')
        for task, mapping in self.mapping.items():
            if not mapping or sorted(mapping.values()) != list(range(len(mapping))):
                raise ValueError(f'Invalid checkpoint class mapping for {task}')
        p = self.preprocessing
        if (not isinstance(p, dict) or p.get('color') != 'EXIF-corrected RGB'
                or p.get('resize') != 'aspect-preserving PIL bicubic pad'
                or p.get('pad_rgb') != [255, 255, 255]
                or p.get('evaluation_order') != 'resize/pad, tensor, normalize'
                or p.get('image_size') != payload.get('config', {}).get('image_size')
                or not isinstance(p.get('image_size'), int) or p['image_size'] < 1):
            raise ValueError('Checkpoint preprocessing metadata is missing or unsupported')
        mean, std = p.get('mean'), p.get('std')
        if (not isinstance(mean, list) or not isinstance(std, list) or len(mean) != 3 or len(std) != 3
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in mean + std)
                or any(v <= 0 for v in std)):
            raise ValueError('Checkpoint normalization metadata is invalid')
        self.transform = transforms.Compose([Square(p['image_size']), transforms.ToTensor(), transforms.Normalize(mean, std)])

    def predict_image(self, image, top_k=3):
        """Classify one PIL image or crop using the already-loaded model."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError('top_k must be a positive integer')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            outputs = self.model(tensor)
        result = {}
        for task, mapping in self.mapping.items():
            labels = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
            if outputs[task].shape != (1, len(labels)) or not torch.isfinite(outputs[task]).all():
                raise ValueError(f'Invalid {task} classifier output')
            values, indices = outputs[task].softmax(dim=1)[0].topk(min(top_k, len(labels)))
            choices = [{'label': labels[i], 'confidence': float(value)}
                       for value, i in zip(values.cpu().tolist(), indices.cpu().tolist())]
            result[task] = {'label': choices[0]['label'], 'confidence': choices[0]['confidence'], 'top_k': choices}
        return result

    def predict(self, images, top_k=3):
        """Return one record per (sample_id, image_path); unreadable images retain their ID and error."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError('top_k must be a positive integer')
        records, seen = [], set()
        for sample_id, image_path in images:
            sample_id, path = str(sample_id).strip(), Path(image_path)
            if not sample_id or sample_id in seen:
                raise ValueError(f'Sample IDs must be nonempty and unique: {sample_id!r}')
            seen.add(sample_id)
            record = {'sample_id': sample_id, 'image_path': str(path)}
            try:
                with Image.open(path) as image:
                    record.update(self.predict_image(image, top_k))
                record['status'] = 'ok'
            except (OSError, UnidentifiedImageError, ValueError) as error:
                record.update(status='error', error=f'{type(error).__name__}: {error}')
            records.append(record)
        return records


def input_items(image=None, folder=None, csv_file=None, image_id=None):
    if sum(x is not None for x in (image, folder, csv_file)) != 1:
        raise ValueError('Select exactly one input source')
    if image is not None:
        path = Path(image)
        return [(image_id or path.stem, path)]
    if image_id is not None:
        raise ValueError('--image-id is only valid with --image')
    if folder is not None:
        directory = Path(folder)
        if not directory.is_dir():
            raise ValueError(f'Input folder does not exist: {directory}')
        paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'})
        return [(p.name, p) for p in paths]
    source = Path(csv_file)
    with source.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'image_id', 'image_path'} <= set(reader.fieldnames or []):
            raise ValueError('CSV must contain image_id and image_path columns')
        rows = list(reader)
    if any(not r['image_id'].strip() or not r['image_path'].strip() for r in rows):
        raise ValueError('CSV image_id and image_path must be nonempty')
    return [(r['image_id'], source.parent / r['image_path']) for r in rows]


def write_results(output, records):
    path = Path(output)
    if path.suffix.lower() not in ('.json', '.csv'):
        raise ValueError('Output extension must be .json or .csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as stream:
        if path.suffix.lower() == '.json':
            json.dump(records, stream, ensure_ascii=False, indent=2)
        else:
            fields = ['sample_id', 'image_path', 'status', 'error', 'type', 'type_confidence', 'type_top_k', 'style', 'style_confidence', 'style_top_k']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in records:
                row = {key:record.get(key, '') for key in fields}
                for task in ('type', 'style'):
                    value = record.get(task, {})
                    row[task] = value.get('label', '')
                    row[task+'_confidence'] = value.get('confidence', '')
                    row[task+'_top_k'] = json.dumps(value.get('top_k', []), ensure_ascii=False)
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Furniture type/style classifier; confidence is uncalibrated classifier softmax')
    parser.add_argument('--checkpoint', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image')
    source.add_argument('--folder')
    source.add_argument('--csv', dest='csv_file')
    parser.add_argument('--image-id')
    parser.add_argument('--output', required=True, help='New .json or .csv file')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--top-k', type=int, default=3)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError('Output exists; choose a new path')
    items = input_items(args.image, args.folder, args.csv_file, args.image_id)
    if not items:
        raise ValueError('No input images found')
    predictor = FurniturePredictor(args.checkpoint, args.device)
    records = predictor.predict(items, args.top_k)
    write_results(args.output, records)
    print(json.dumps({'output':args.output, 'images':len(records), 'errors':sum(r['status'] == 'error' for r in records)}, ensure_ascii=False))
    if any(r['status'] == 'error' for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
