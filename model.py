import torch
from torch import nn

from models import resnet


def generate_model(opt):
    assert opt.model in ['resnet']

    if opt.model == 'resnet':
        assert opt.model_depth in [18, 34, 50, 101]

        from models.resnet import get_fine_tuning_parameters
        
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

    if opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        assert opt.arch == pretrain['arch']

        model.load_state_dict(pretrain['state_dict'])

        model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
        if not opt.no_cuda:
            model.module.fc = model.module.fc.cuda()

        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
        return model, parameters

    return model, model.parameters()