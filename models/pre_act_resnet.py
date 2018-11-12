import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resnet


class PreActivationBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        self.bn1 = nn.BatchNorm3d(inplanes)
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class PreActivationBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        self.bn1 = nn.BatchNorm3d(inplanes)
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv2 = resnet.conv3x3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = resnet.ResNet(PreActivationBasicBlock, [2, 2, 2, 2],
                          resnet.get_inplanes(), **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = resnet.ResNet(PreActivationBasicBlock, [3, 4, 6, 3],
                          resnet.get_inplanes(), **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = resnet.ResNet(PreActivationBottleneck, [3, 4, 6, 3],
                          resnet.get_inplanes(), **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = resnet.ResNet(PreActivationBottleneck, [3, 4, 23, 3],
                          resnet.get_inplanes(), **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = resnet.ResNet(PreActivationBottleneck, [3, 8, 36, 3],
                          resnet.get_inplanes(), **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = resnet.ResNet(PreActivationBottleneck, [3, 24, 36, 3],
                          resnet.get_inplanes(), **kwargs)
    return model
