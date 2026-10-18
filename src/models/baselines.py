"""Standard torchvision baselines with grayscale input and three-class output."""
from torch import nn
from torchvision import models

BASELINES = ('resnet18', 'resnet50', 'densenet121', 'mobilenet_v3_small', 'efficientnet_b0')
DISPLAY_NAMES = dict(zip(BASELINES, ['ResNet18', 'ResNet50', 'DenseNet121', 'MobileNetV3-Small', 'EfficientNet-B0']))
INPUT_LAYERS = dict(zip(BASELINES, ['conv1', 'conv1', 'features.conv0', 'features.0.0', 'features.0.0']))
OUTPUT_LAYERS = dict(zip(BASELINES, ['fc', 'fc', 'classifier', 'classifier.3', 'classifier.1']))


def build_baseline(model_name: str, num_classes: int = 3, input_channels: int = 1,
                   pretrained: bool = False) -> nn.Module:
    """Construct a from-scratch baseline, preserving all deeper architecture.

    The native constructor initializes the three-class classifier. Only the first
    Conv2d is replaced, retaining its spatial settings and torchvision family's
    initialization rule. Pretrained experiments are deliberately unsupported.
    """
    if model_name not in BASELINES:
        raise ValueError(f'Unsupported baseline {model_name!r}; choose one of {BASELINES}')
    if pretrained is not False:
        raise ValueError('The primary experiment requires pretrained=False and weights=None')
    if num_classes != 3 or input_channels != 1:
        raise ValueError('The frozen protocol requires one input channel and three classes')
    model = getattr(models, model_name)(weights=None, num_classes=num_classes)
    path = INPUT_LAYERS[model_name]
    original = model.get_submodule(path)
    replacement = nn.Conv2d(input_channels, original.out_channels, original.kernel_size,
                            original.stride, original.padding, original.dilation, original.groups,
                            original.bias is not None, original.padding_mode)
    # DenseNet uses fan_in; the other four torchvision families use fan_out.
    nn.init.kaiming_normal_(replacement.weight,
                           mode='fan_in' if model_name == 'densenet121' else 'fan_out',
                           nonlinearity='relu')
    if replacement.bias is not None:
        nn.init.zeros_(replacement.bias)
    parent_path, _, child = path.rpartition('.')
    parent = model.get_submodule(parent_path) if parent_path else model
    setattr(parent, child, replacement)
    return model
