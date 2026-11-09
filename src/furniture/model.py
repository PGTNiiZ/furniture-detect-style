import torch.nn as nn
from torchvision import models


BACKBONES = {
    'resnet18': ('fc',), 'resnet50': ('fc',),
    'densenet121': ('classifier',),
    'efficientnet_b0': ('classifier', -1), 'efficientnet_b2': ('classifier', -1),
    'mobilenet_v3_large': ('classifier', -1), 'regnet_y_400mf': ('fc',),
    'convnext_tiny': ('classifier', -1), 'swin_t': ('head',),
    'vit_b_16': ('heads', 'head'),
}
WEIGHT_NAMES = {name: ('IMAGENET1K_V2' if name in ('resnet50', 'mobilenet_v3_large', 'regnet_y_400mf')
                       else 'IMAGENET1K_V1') for name in BACKBONES}


class AttributeModel(nn.Module):
    def __init__(self, mapping, pretrained=True, dropout=0.0, backbone='resnet18', weights_name=None):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f'Unsupported backbone: {backbone}')
        weights = models.get_model_weights(backbone)[weights_name or WEIGHT_NAMES[backbone]] if pretrained else None
        self.backbone = models.get_model(backbone, weights=weights)
        parent = self.backbone
        *path, last = BACKBONES[backbone]
        for part in path:
            parent = parent[part] if isinstance(part, int) else getattr(parent, part)
        layer = parent[last] if isinstance(last, int) else getattr(parent, last)
        self.feature_dim = layer.in_features
        if isinstance(last, int):
            parent[last] = nn.Identity()
        else:
            setattr(parent, last, nn.Identity())
        self.backbone_name = backbone
        self.weights_identifier = str(weights) if weights else None
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({key + '_head': nn.Linear(self.feature_dim, len(values)) for key, values in mapping.items()})

    def forward(self, images):
        features = self.dropout(self.backbone(images))
        return {key.removesuffix('_head'): head(features) for key, head in self.heads.items()}

    def freeze_backbone(self, frozen):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not frozen)
