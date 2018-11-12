import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import resnet


class ResNeXtBottleneck(resnet.Bottleneck):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super().__init__()

        mid_planes = cardinality * planes // 32
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)


class ResNeXt(resnet.ResNet):

    def __init__(self,
                 block,
                 layers,
                 inplanes,
                 sample_size,
                 sample_duration,
                 conv1_t_size=7,
                 shortcut_type='B',
                 cardinality=32,
                 n_classes=400):
        super().__init__(block, layers, inplanes, sample_size, sample_duration,
                         conv1_t_size, shortcut_type, n_classes)

        block = partial(block, cardinality=cardinality)

        self.layer1 = self._make_layer(block, inplanes[0] * 2, layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(
            block, inplanes[1] * 2, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, inplanes[2] * 2, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, inplanes[3] * 2, layers[3], shortcut_type, stride=2)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnext50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], resnet.get_inplanes(),
                    **kwargs)
    return model


def resnext101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], resnet.get_inplanes(),
                    **kwargs)
    return model


def resnext152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], resnet.get_inplanes(),
                    **kwargs)
    return model
