import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resnet


class WideBottleneck(resnet.Bottleneck):
    expansion = 2


def resnet50(k, **kwargs):
    """Constructs a ResNet-50 model.
    """
    inplanes = [x * k for x in resnet.get_inplanes()]
    model = resnet.ResNet(WideBottleneck, [3, 4, 6, 3], inplanes, **kwargs)
    return model
