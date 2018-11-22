import torch
from torch import nn

from models import resnet, pre_act_resnet, wide_resnet, resnext, densenet


def get_fine_tuning_parameters(model, ft_begin_module):
    if not ft_begin_module:
        return model.parameters()

    get_module_name = (
        lambda x: x.split('.')[1] if x.split('.')[0] == 'features' else x.split('.')[0]
    )

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

    if opt.sample_duration >= 32:
        conv1_t_stride = 2
    else:
        conv1_t_stride = 1

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = resnet.resnet10(
                n_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                conv1_t_stride=conv1_t_stride,
                conv1_t_size=opt.conv1_t_size)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                n_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                conv1_t_stride=conv1_t_stride,
                conv1_t_size=opt.conv1_t_size)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                n_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                conv1_t_stride=conv1_t_stride,
                conv1_t_size=opt.conv1_t_size)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                n_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                conv1_t_stride=conv1_t_stride,
                conv1_t_size=opt.conv1_t_size)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                n_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                conv1_t_stride=conv1_t_stride,
                conv1_t_size=opt.conv1_t_size)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                n_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                conv1_t_stride=conv1_t_stride,
                conv1_t_size=opt.conv1_t_size)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                n_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                conv1_t_stride=conv1_t_stride,
                conv1_t_size=opt.conv1_t_size)
    elif opt.model == 'wideresnet':
        assert opt.model_depth in [50]

        if opt.model_depth == 50:
            model = wide_resnet.resnet50(
                n_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                k=opt.wide_resnet_k,
                conv1_t_stride=conv1_t_stride)
    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]

        if opt.model_depth == 50:
            model = resnext.resnext50(
                n_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                conv1_t_stride=conv1_t_stride)
        elif opt.model_depth == 101:
            model = resnext.resnext101(
                n_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                conv1_t_stride=conv1_t_stride)
        elif opt.model_depth == 152:
            model = resnext.resnext152(
                n_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                conv1_t_stride=conv1_t_stride)
    elif opt.model == 'preresnet':
        assert opt.model_depth in [18, 34, 50, 101, 152, 200]

        if opt.model_depth == 18:
            model = pre_act_resnet.resnet18(
                n_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                conv1_t_stride=conv1_t_stride)
        elif opt.model_depth == 34:
            model = pre_act_resnet.resnet34(
                n_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                conv1_t_stride=conv1_t_stride)
        elif opt.model_depth == 50:
            model = pre_act_resnet.resnet50(
                n_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                conv1_t_stride=conv1_t_stride)
        elif opt.model_depth == 101:
            model = pre_act_resnet.resnet101(
                n_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                conv1_t_stride=conv1_t_stride)
        elif opt.model_depth == 152:
            model = pre_act_resnet.resnet152(
                n_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                conv1_t_stride=conv1_t_stride)
        elif opt.model_depth == 200:
            model = pre_act_resnet.resnet200(
                n_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                conv1_t_stride=conv1_t_stride)
    elif opt.model == 'densenet':
        assert opt.model_depth in [121, 169, 201, 264]

        if opt.model_depth == 121:
            model = densenet.densenet121(
                n_classes=opt.n_classes, conv1_t_stride=conv1_t_stride)
        elif opt.model_depth == 169:
            model = densenet.densenet169(
                n_classes=opt.n_classes, conv1_t_stride=conv1_t_stride)
        elif opt.model_depth == 201:
            model = densenet.densenet201(
                n_classes=opt.n_classes, conv1_t_stride=conv1_t_stride)
        elif opt.model_depth == 264:
            model = densenet.densenet264(
                n_classes=opt.n_classes, conv1_t_stride=conv1_t_stride)

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