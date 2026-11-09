"""CPU-only adapter smoke on v2-train matches; not a research run or pilot."""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageStat
from furniture.model import AttributeModel
from new_data.zip_dataset import loader_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    digest = hashlib.sha256()
    with Path(args.archive).open('rb') as archive_file:
        for block in iter(lambda: archive_file.read(4*1024*1024), b''):
            digest.update(block)
    if digest.hexdigest() != 'dce23f8f54b1317d31805564a4d58ab0eff2ed3ce00e06656df188440563f79a':
        raise ValueError('Archive differs from the CRC-verified source')
    root = Path('data/furniture-v2-conservative')
    with (root/'train.csv').open(encoding='utf-8', newline='') as f:
        v2 = list(csv.DictReader(f))
    mapping = {key:{value:i for i,value in enumerate(sorted({r[key] for r in v2}))} for key in ('type','style')}
    lookup = {row['image_id']:row for row in v2}
    with Path('artifacts/small-v2-index-overlap-candidates.csv').open(encoding='utf-8', newline='') as f:
        candidates = [r for r in csv.DictReader(f) if r['incoming_split']=='train' and r['v2_split']=='train'
                      and r['exact_label_match']=='True'][:8]
    rows = []
    with zipfile.ZipFile(args.archive) as z:
        for candidate in candidates:
            ref = lookup[candidate['v2_image_id']]
            with Image.open(root/ref['image_path']) as image:
                original = image.convert('RGB').resize((256,256), Image.Resampling.LANCZOS)
            with Image.open(io.BytesIO(z.read('furniture_dataset/data/processed/'+candidate['incoming_path']))) as image:
                actual = image.convert('RGB')
            mae = sum(ImageStat.Stat(ImageChops.difference(original,actual)).mean)/3
            if mae > 5:
                raise ValueError('Sample differs from v2 training source')
            rows.append({'image_id':ref['image_id'], 'image_path':candidate['incoming_path'],
                         'type':ref['type'], 'style':ref['style']})
    if len(rows) != 8:
        raise ValueError('Expected eight v2 training samples')
    manifest = out/'train.csv'
    with manifest.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    config = {'storage_mode':'zip', 'archive_path':str(Path(args.archive).resolve()),
              'archive_prefix':'furniture_dataset/data/processed/', 'manifest_path':str(manifest.resolve()),
              'snapshot_id':'smoke-only-v2-train-matches', 'corrupt_image_policy':'error',
              'num_workers':0, 'batch_size':4, 'image_size':64, 'seed':42, 'shuffle':True}
    (out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (out/'mapping.json').write_text(json.dumps(mapping, indent=2), encoding='utf-8')
    loader = loader_from_config(out/'config.json', mapping)
    model = AttributeModel(mapping, pretrained=False, backbone='resnet18')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    images, labels = next(iter(loader))
    outputs = model(images)
    loss = sum(F.cross_entropy(outputs[task], labels[task]) for task in mapping)
    loss.backward()
    gradient = model.backbone.conv1.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
        raise ValueError('Missing or invalid shared backbone gradient')
    optimizer.step()
    result = {'status':'passed', 'scope':'CPU integration smoke only; no research metric or selected checkpoint',
              'images_in_manifest':len(rows), 'images_in_optimizer_step':len(images),
              'snapshot':'v2 train matches only; JPEG comparison <= 5 MAE; no holdout examples',
              'train_manifest_sha256':hashlib.sha256((root/'train.csv').read_bytes()).hexdigest(),
              'archive_sha256':digest.hexdigest(),
              'shared_backbone_gradient_nonzero':True, 'device':'cpu'}
    (out/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
