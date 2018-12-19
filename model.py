import torch
from torch import nn

from models import resnet, pre_act_resnet, wide_resnet, resnext, densenet


def get_module_name(name):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1

    return name[i]


def get_fine_tuning_parameters(model, ft_begin_module):
    if not ft_begin_module:
        return model.parameters()

    parameters = []
    add_flag = False
    for k, v in model.named_parameters():
        if ft_begin_module == get_module_name(k):
            add_flag = True

        if add_flag:
            parameters.append({'params': v})

    return parameters


def generate_model(opt):
    assert opt.model in [
        'resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet'
    ]

    if opt.model == 'resnet':
        model = resnet.generate_model(
            model_depth=opt.model_depth,
            n_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool)
    elif opt.model == 'wideresnet':
        model = wide_resnet.generate_model(
            model_depth=opt.model_depth,
            k=opt.wide_resnet_k,
            n_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool)
    elif opt.model == 'resnext':
        model = resnext.generate_model(
            model_depth=opt.model_depth,
            cardinality=opt.resnext_cardinality,
            n_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool)
    elif opt.model == 'preresnet':
        model = pre_act_resnet.generate_model(
            model_depth=opt.model_depth,
            n_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool)
    elif opt.model == 'densenet':
        model = densenet.generate_model(
            model_depth=opt.model_depth,
            n_classes=opt.n_classes,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool)

    if not opt.no_cuda:
        model = nn.DataParallel(model, device_ids=None).cuda()

    if not opt.pretrain_path:
        return model, model.parameters()

    print('loading pretrained model {}'.format(opt.pretrain_path))
    pretrain = torch.load(opt.pretrain_path)
    assert opt.arch == pretrain['arch']

    model.load_state_dict(pretrain['state_dict'])

    if opt.no_cuda:
        tmp_model = model
    else:
        tmp_model = model.module
    if opt.model == 'densenet':
        tmp_model.classifier = nn.Linear(tmp_model.classifier.in_features,
                                         opt.n_finetune_classes).to(opt.device)
    else:
        tmp_model.fc = nn.Linear(tmp_model.fc.in_features,
                                 opt.n_finetune_classes).to(opt.device)

    parameters = get_fine_tuning_parameters(model, opt.ft_begin_module)

    return model, parameters