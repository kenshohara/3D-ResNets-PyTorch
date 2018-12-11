import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import conv1x1x1, Bottleneck, ResNet
from utils import partialclass


def get_inplanes():
    return [128, 256, 512, 1024]


class ResNeXtBottleneck(Bottleneck):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super().__init__(inplanes, planes, stride, downsample)

        mid_planes = cardinality * planes // 32
        self.conv1 = conv1x1x1(inplanes, mid_planes)
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
        self.conv3 = conv1x1x1(mid_planes, planes * self.expansion)


class ResNeXt(ResNet):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 shortcut_type='B',
                 cardinality=32,
                 n_classes=400):
        block = partialclass(block, cardinality=cardinality)
        super().__init__(block, layers, block_inplanes, sample_size,
                         conv1_t_size, conv1_t_stride, shortcut_type, n_classes)

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
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    return model


def resnext101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    return model


def resnext152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    return model
