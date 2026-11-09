"""Compose small YAML files and reject misspellings before expensive work."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import yaml


def read_yaml(path):
    with Path(path).open(encoding='utf-8') as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a YAML mapping')
    return value


def exact_keys(value, expected, location):
    unknown, missing = set(value) - set(expected), set(expected) - set(value)
    if unknown or missing:
        raise ValueError(f'{location}: unknown keys={sorted(unknown)}, missing keys={sorted(missing)}')


def merge_known(base, overrides, prefix=''):
    result = deepcopy(base)
    for key, value in overrides.items():
        if key not in base:
            raise ValueError(f'Unknown config key: {prefix}{key}')
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f'{prefix}{key} must be a mapping')
            result[key] = merge_known(base[key], value, prefix + key + '.')
        else:
            result[key] = value
    return result


def number(value, name, low=0, high=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if integer and not isinstance(value, int):
        raise ValueError(f'{name} must be an integer')
    if value < low or (high is not None and value > high):
        raise ValueError(f'{name} out of range [{low}, {high}]')


def validate_augmentation(a):
    if not isinstance(a, dict):
        raise ValueError('augmentation must be a mapping')
    if a.get('backend') == 'albumentations':
        exact_keys(a, ['backend','preset','params','occlusion'], 'augmentation')
        if a['preset'] not in ('none','easy','hard','custom'):
            raise ValueError('augmentation.preset must be none, easy, hard or custom')
        p = a['params']
        if not isinstance(p, dict) or not isinstance(a['occlusion'], dict):
            raise ValueError('augmentation.params and occlusion must be mappings')
        exact_keys(p, ['horizontal_flip_p','affine_p','rotate_degrees','translate_fraction','scale',
                       'brightness_contrast_p','brightness_limit','contrast_limit','blur_p','blur_kernel'], 'augmentation.params')
        for key in ('horizontal_flip_p','affine_p','brightness_contrast_p','blur_p'):
            number(p[key], 'augmentation.params.'+key, 0, 1)
        number(p['rotate_degrees'], 'rotate_degrees', 0, 30)
        number(p['translate_fraction'], 'translate_fraction', 0, .5)
        for key in ('brightness_limit','contrast_limit'):
            number(p[key], key, 0, 1)
        def interval(value, name, low, high, integer=False):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f'{name} must contain two values')
            for item in value:
                number(item, name, low, high, integer=integer)
            if value[0] > value[1]:
                raise ValueError(f'{name} lower bound exceeds upper bound')
        interval(p['scale'], 'scale', .1, 2)
        number(p['blur_kernel'], 'blur_kernel', 3, 99, integer=True)
        if p['blur_kernel'] % 2 == 0:
            raise ValueError('blur_kernel must be odd')
        if a['preset'] == 'none' and any(p[k] for k in ('horizontal_flip_p','affine_p','brightness_contrast_p','blur_p')):
            raise ValueError('preset none requires zero augmentation probabilities')
        o = a['occlusion']
        exact_keys(o, ['preset','probability','holes','height_fraction','width_fraction','fill'], 'augmentation.occlusion')
        if o['preset'] not in ('off','light','strong','custom'):
            raise ValueError('occlusion.preset must be off, light, strong or custom')
        number(o['probability'], 'occlusion.probability', 0, 1)
        if o['preset'] == 'off' and o['probability'] != 0:
            raise ValueError('occlusion off requires zero probability')
        interval(o['holes'], 'occlusion.holes', 1, 100, integer=True)
        interval(o['height_fraction'], 'occlusion.height_fraction', .001, 1)
        interval(o['width_fraction'], 'occlusion.width_fraction', .001, 1)
        number(o['fill'], 'occlusion.fill', 0, 255, integer=True)
        return
    if 'backend' in a:
        raise ValueError('Unsupported augmentation backend')
    if 'preset' in a or 'occlusion' in a:
        raise ValueError('Old preset config is ambiguous; migrate to configs/augmentation/*_*.yaml')
    required = ['enabled','horizontal_flip_p','rotation_degrees','brightness','contrast','saturation','hue']
    unknown = set(a) - set(required)
    missing = set(required) - set(a)
    if unknown or missing:
        raise ValueError(f'augmentation: unknown keys={sorted(unknown)}, missing keys={sorted(missing)}')
    if not isinstance(a['enabled'], bool):
        raise ValueError('augmentation.enabled must be true/false')
    number(a['horizontal_flip_p'], 'horizontal_flip_p', 0, 1)
    number(a['rotation_degrees'], 'rotation_degrees', 0, 30)
    for key in ['brightness', 'contrast', 'saturation']:
        number(a[key], key, 0, 1)
    number(a['hue'], 'hue', 0, .5)


def validate(cfg):
    data, model, train, aug, inference = (cfg[k] for k in ['dataset','model','training','augmentation','inference'])
    exact_keys(data, ['id','root','source','split'], 'dataset')
    exact_keys(data['source'], ['kind','dataset_id','revision','csv_file','image_root','columns'], 'dataset.source')
    exact_keys(data['source']['columns'], ['image_path','type','style','group_id','split'], 'dataset.source.columns')
    if data['source']['kind'] not in ['huggingface','csv','prepared']:
        raise ValueError('source.kind must be huggingface, csv or prepared')
    if not data['id'] or not data['root']:
        raise ValueError('dataset id and root are required')
    exact_keys(data['split'], ['seed','val_fraction','test_fraction','previous_root'], 'dataset.split')
    number(data['split']['seed'], 'split.seed', 0, integer=True)
    for key in ['val_fraction','test_fraction']:
        number(data['split'][key], key, .001, .49)
    if data['split']['val_fraction'] + data['split']['test_fraction'] >= 1:
        raise ValueError('Need a non-empty training fraction')
    exact_keys(model, ['backbone','initialization','checkpoint','checkpoint_scope','image_size','dropout','weights'], 'model')
    from .model import BACKBONES
    if model['backbone'] not in BACKBONES:
        raise ValueError(f"Unsupported backbone: {model['backbone']}")
    from torchvision import models
    if not isinstance(model['weights'], str) or model['weights'] not in models.get_model_weights(model['backbone']).__members__ or model['weights'] == 'DEFAULT':
        raise ValueError('model.weights must be an explicit torchvision weights enum member')
    if model['initialization'] not in ['imagenet','scratch','checkpoint']:
        raise ValueError('initialization: imagenet, scratch or checkpoint')
    if model['checkpoint_scope'] not in ['full','backbone']:
        raise ValueError('checkpoint_scope: full or backbone')
    if model['initialization'] == 'checkpoint' and not model['checkpoint']:
        raise ValueError('checkpoint path is required')
    number(model['image_size'], 'image_size', 32, 1024, integer=True)
    if model['backbone'] == 'vit_b_16' and model['image_size'] != 224:
        raise ValueError('vit_b_16 requires image_size 224')
    number(model['dropout'], 'dropout', 0, .9)
    exact_keys(train, ['seed','batch_size','epochs','freeze_epochs','head_lr','finetune_lr','weight_decay','patience','device','loss_weights','selection_metric','tasks','accumulation_steps','amp'], 'training')
    if train['tasks'] not in (['type', 'style'], ['type'], ['style']):
        raise ValueError('training.tasks must be [type, style], [type], or [style]')
    number(train['accumulation_steps'], 'accumulation_steps', 1, 1024, integer=True)
    if not isinstance(train['amp'], bool):
        raise ValueError('training.amp must be true/false')
    for key, low in [('seed',0),('batch_size',2),('epochs',1),('patience',1)]:
        number(train[key], key, low, integer=True)
    if train['freeze_epochs'] != 'all':
        number(train['freeze_epochs'], 'freeze_epochs', 0, train['epochs'], integer=True)
    for key in ['head_lr','finetune_lr']:
        number(train[key], key, 1e-10, 1)
    number(train['weight_decay'], 'weight_decay', 0, 1)
    exact_keys(train['loss_weights'], ['type','style'], 'loss_weights')
    for key, weight in train['loss_weights'].items():
        number(weight, f'loss_weights.{key}', 1e-8, 100)
    if train['selection_metric'] not in ['mean_macro_f1','style_macro_f1']:
        raise ValueError('selection_metric: mean_macro_f1 or style_macro_f1')
    if train['selection_metric'] == 'style_macro_f1' and 'style' not in train['tasks']:
        raise ValueError('style_macro_f1 requires the style task')
    validate_augmentation(aug)
    exact_keys(inference, ['detector_weights','detector_confidence','minimum_confidence','keep_threshold','replace_threshold','top_k'], 'inference')
    for key in ['detector_confidence','minimum_confidence','keep_threshold','replace_threshold']:
        number(inference[key], key, 0, 1)
    if inference['replace_threshold'] >= inference['keep_threshold']:
        raise ValueError('replace_threshold must be less than keep_threshold')
    number(inference['top_k'], 'top_k', 1, 100, integer=True)


def load_config(project_file, experiment=None):
    path = Path(project_file).resolve()
    project = read_yaml(path)
    exact_keys(project, ['schema_version','project_root','output_root','default_experiment','dataset','model','training','augmentation','inference','experiments'], 'project')
    if project['schema_version'] != 1:
        raise ValueError('Unsupported schema_version')
    selected = experiment or project['default_experiment']
    cfg = {}
    source_files = [path]
    for key in ['dataset','model','training','augmentation','inference']:
        child = (path.parent / project[key]).resolve()
        cfg[key] = read_yaml(child)
        source_files.append(child)
    cfg['training'].setdefault('tasks', ['type', 'style'])
    cfg['training'].setdefault('accumulation_steps', 1)
    cfg['training'].setdefault('amp', False)
    from .model import WEIGHT_NAMES
    cfg['model'].setdefault('weights', WEIGHT_NAMES.get(cfg['model'].get('backbone')))
    definitions = {}
    for name in project['experiments']:
        child = (path.parent / name).resolve()
        spec = read_yaml(child)
        exact_keys(spec, ['id','description','overrides'], f'experiment {child}')
        if spec['id'] in definitions:
            raise ValueError('Duplicate experiment id')
        definitions[spec['id']] = (spec, child)
    if selected not in definitions:
        raise ValueError(f'Unknown experiment {selected}; choose {list(definitions)}')
    spec, child = definitions[selected]
    cfg = merge_known(cfg, spec['overrides'])
    source_files.append(child)
    cfg['experiment'] = {key: spec[key] for key in ['id','description']}
    cfg['project_root'] = str((path.parent / project['project_root']).resolve())
    cfg['output_root'] = project['output_root']
    validate(cfg)
    cfg['source_configs'] = {str(f.relative_to(path.parent)): f.read_text(encoding='utf-8') for f in source_files}
    return cfg


def resolve_path(cfg, path):
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (Path(cfg['project_root']) / p).resolve()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def training_settings(cfg):
    return {**deepcopy(cfg['training']), 'freeze_epochs':cfg['training']['epochs'] if cfg['training']['freeze_epochs']=='all' else cfg['training']['freeze_epochs'], 'image_size': cfg['model']['image_size'],
            'pretrained': cfg['model']['initialization'] == 'imagenet',
            'dropout': cfg['model']['dropout'], 'backbone': cfg['model']['backbone'], 'weights_name':cfg['model']['weights'], 'augment': deepcopy(cfg['augmentation']),
            'initialization': cfg['model']['initialization'],
            'init_checkpoint': str(resolve_path(cfg, cfg['model']['checkpoint'])) if cfg['model']['checkpoint'] else None,
            'checkpoint_scope': cfg['model']['checkpoint_scope']}
