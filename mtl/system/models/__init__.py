from .mtl import (
    BranchingVGGa,
    BranchingVGG9a,
    BranchingVGG11a,
    CrossStitchVGGa,
    CrossStitchVGG9a,
    CrossStitchVGG11a,
)
from .resnet import ResNet, ResNet18a
from .vgg import VGGa, VGG9a, VGG11a


__all__ = [
    'BranchingVGGa',
    'BranchingVGG9a',
    'BranchingVGG11a',
    'CrossStitchVGGa',
    'CrossStitchVGG9a',
    'CrossStitchVGG11a',
    'ResNet',
    'ResNet18a',
    'VGGa',
    'VGG9a',
    'VGG11a',
    'create_model',
]


def create_model(model_name, n_channels, n_classes, **kwargs):
    model_classes = {
        'vgg9a': VGG9a,
        'vgg11a': VGG11a,
        'resnet18a': ResNet18a,
        'vgg9a_branch': BranchingVGG9a,
        'vgg11a_branch': BranchingVGG11a,
        'vgg9a_stitch': CrossStitchVGG9a,
        'vgg11a_stitch': CrossStitchVGG11a,
    }

    try:
        model_class = model_classes[model_name]
        model = model_class(n_channels, n_classes, **kwargs)
    except KeyError:
        raise ValueError(f'Unrecognized model type: {model_name}')

    return model
