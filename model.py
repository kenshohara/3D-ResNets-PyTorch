import torch
from torch import nn

from models import resnet


def generate_model(opt):
    assert opt.model in ['resnet']

    if opt.model == 'resnet':
        assert opt.model_depth in [18, 34, 50, 101]

        if opt.model_depth == 18:
            model = resnet.resnet18(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut)
        elif opt.model_depth == 34:
            model = resnet.resnet34(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut)
        elif opt.model_depth == 50:
            model = resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut)
        elif opt.model_depth == 101:
            model = resnet.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

    return model
