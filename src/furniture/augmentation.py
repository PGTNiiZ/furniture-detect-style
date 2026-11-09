"""Image-only training transforms; object detection uses YOLO's box-aware pipeline."""
import random
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T
from .config import validate_augmentation

DEFAULT = dict(enabled=False, horizontal_flip_p=0., rotation_degrees=0.,
               brightness=0., contrast=0., saturation=0., hue=0.)


def settings(value):
    if isinstance(value, bool):  # compatibility with original starter
        return {**DEFAULT, 'enabled': value, 'horizontal_flip_p': .5 if value else 0.}
    if not isinstance(value, dict):
        raise ValueError('augmentation must be a boolean or mapping')
    result = value.copy() if value.get('backend') else {**DEFAULT, **value}
    validate_augmentation(result)
    return result


def pil_transforms(value):
    a = settings(value)
    if a.get('backend') == 'albumentations':
        return [AlbumentationsTransform(a)]
    steps = []
    if a['enabled']:
        if a['horizontal_flip_p']:
            steps.append(T.RandomHorizontalFlip(a['horizontal_flip_p']))
        if a['rotation_degrees']:
            steps.append(T.RandomRotation(a['rotation_degrees'], fill=(255,255,255)))
        if any(a[k] for k in ['brightness','contrast','saturation','hue']):
            steps.append(T.ColorJitter(**{k:a[k] for k in ['brightness','contrast','saturation','hue']}))
    return steps


class AlbumentationsTransform:
    def __init__(self, config):
        import os
        os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
        import albumentations as A
        import cv2
        p, o = config['params'], config['occlusion']
        steps = [
            A.HorizontalFlip(p=p['horizontal_flip_p']),
            A.Affine(rotate=(-p['rotate_degrees'], p['rotate_degrees']),
                     translate_percent=(-p['translate_fraction'], p['translate_fraction']),
                     scale=tuple(p['scale']), interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                     fill=(255,255,255), p=p['affine_p']),
            A.RandomBrightnessContrast(brightness_limit=p['brightness_limit'], contrast_limit=p['contrast_limit'],
                                       brightness_by_max=True, p=p['brightness_contrast_p']),
            A.GaussianBlur(blur_limit=(p['blur_kernel'],p['blur_kernel']), p=p['blur_p']),
            A.CoarseDropout(num_holes_range=tuple(o['holes']), hole_height_range=tuple(o['height_fraction']),
                            hole_width_range=tuple(o['width_fraction']), fill=o['fill'], p=o['probability']),
        ]
        self.pipeline = A.Compose(steps, seed=int(torch.randint(0, 2**31, ()).item()))

    def __call__(self, image):
        import numpy as np
        return Image.fromarray(self.pipeline(image=np.asarray(image))['image'])


def preview(image_path, output, size, augmentation, count=6, seed=42):
    from .data import Square
    random.seed(seed)
    torch.manual_seed(seed)
    with Image.open(image_path) as image:
        base = Square(size)(image)
    canvas = Image.new('RGB', ((count+1)*size, size+28), 'white')
    draw = ImageDraw.Draw(canvas)
    transform = T.Compose(pil_transforms(augmentation))
    for i in range(count+1):
        canvas.paste(base if i == 0 else transform(base), (i*size, 28))
        draw.text((i*size+6, 6), 'original' if i == 0 else f'augmented {i}', fill='black')
    path = Path(output)
    if path.exists():
        raise ValueError('Preview output exists; choose a new path')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return {'preview': str(path), 'augmentation':settings(augmentation), 'seed':seed}


def preview_dataset(root, output, size, seed, count, preset_dir):
    """Select training samples across label pairs and save eight comparable panels each."""
    import json
    import yaml
    from .data import read_rows, Square
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError('count must be a positive integer')
    root, output, preset_dir = Path(root), Path(output), Path(preset_dir)
    if output.exists():
        raise ValueError('Preview output exists; choose a new directory')
    rows = read_rows(root/'train.csv')
    if not rows:
        raise ValueError('No training images')
    rng = random.Random(seed)
    groups = {}
    for row in rows:
        groups.setdefault((row['type'], row['style']), []).append(row)
    for group in groups.values():
        rng.shuffle(group)
    keys = sorted(groups)
    selected = []
    while keys and len(selected) < count:
        for key in list(keys):
            selected.append(groups[key].pop())
            if not groups[key]:
                keys.remove(key)
            if len(selected) == count:
                break
    names = ['none_off', 'easy_off', 'hard_off', 'easy_light', 'easy_strong', 'hard_light', 'hard_strong']
    configs = {name: yaml.safe_load((preset_dir/f'{name}.yaml').read_text(encoding='utf-8')) for name in names}
    for config in configs.values():
        validate_augmentation(config)
    output.mkdir(parents=True)
    torch.manual_seed(seed)
    transforms = {name:T.Compose(pil_transforms(config)) for name,config in configs.items()}
    results = []
    for i, row in enumerate(selected):
        with Image.open(root/row['image_path']) as image:
            base = Square(size)(image)
        canvas = Image.new('RGB', (4*size, 2*(size+24)), 'white')
        draw = ImageDraw.Draw(canvas)
        for j,name in enumerate(['original', *names]):
            x, y = (j % 4)*size, (j // 4)*(size+24)
            canvas.paste(base if name == 'original' else transforms[name](base), (x, y+24))
            draw.text((x+4,y+4), name, fill='black')
        filename = f'sample-{i+1:03d}.png'
        canvas.save(output/filename)
        results.append({'sample_id':row['image_id'], 'type':row['type'], 'style':row['style'], 'file':filename})
    report = {'split':'train', 'seed':seed, 'samples':results, 'augmentation_configs':configs,
              'selection':'balanced across available type/style pairs; no test rows used'}
    (output/'preview.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return {'output':str(output), 'samples':len(results), 'report':str(output/'preview.json')}
