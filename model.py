import torch
from torch import nn

from models import resnet, pre_act_resnet, densenet


def generate_model(opt):
    assert opt.model in ['resnet', 'preresnet', 'densenet']

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
    elif opt.model == 'preresnet':
        assert opt.model_depth in [18, 34, 50, 101]

        if opt.model_depth == 18:
            model = pre_act_resnet.resnet18(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut)
        elif opt.model_depth == 34:
            model = pre_act_resnet.resnet34(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut)
        elif opt.model_depth == 50:
            model = pre_act_resnet.resnet50(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut)
        elif opt.model_depth == 101:
            model = pre_act_resnet.resnet101(num_classes=opt.n_classes, shortcut_type=opt.resnet_shortcut)
    elif opt.model == 'densenet':
        assert opt.model_depth in [121, 169, 201, 161]

        if opt.model_depth == 121:
            model = densenet.densenet121(num_classes=opt.n_classes)
        elif opt.model_depth == 169:
            model = densenet.densenet169(num_classes=opt.n_classes)
        elif opt.model_depth == 201:
            model = densenet.densenet201(num_classes=opt.n_classes)
        elif opt.model_depth == 161:
            model = densenet.densenet161(num_classes=opt.n_classes)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

    return model
