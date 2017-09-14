import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop,
                                MultiScaleCornerCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import TemporalRandomCrop, TemporalBeginCrop
from kinetics import Kinetics
from activitynet import ActivityNet
from utils import Logger
from train import train_epoch
from validation import val_epoch

if __name__=="__main__":
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.model_path = os.path.join(opt.root_path, opt.model_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.premodel_path:
            opt.premodel_path = os.path.join(opt.root_path, opt.premodel_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model = generate_model(opt)
    print(model)

    if not opt.no_train:
        spatial_transform = Compose([MultiScaleCornerCrop(opt.scales, opt.sample_size),
                                     RandomHorizontalFlip(),
                                     ToTensor(opt.norm_value),
                                     Normalize(opt.mean, [1, 1, 1])])
        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        if opt.dataset == 'kinetics':
            training_data = Kinetics(opt.video_path, opt.annotation_path, 'training',
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform)
        else:
            training_data = ActivityNet(opt.video_path, opt.annotation_path, 'training',
                                        spatial_transform=spatial_transform,
                                        temporal_transform=temporal_transform)
        train_loader = torch.utils.data.DataLoader(training_data, batch_size=opt.batch_size,
                                                   shuffle=True, num_workers=opt.n_threads, pin_memory=True)
        train_logger = Logger(os.path.join(opt.result_path, 'train.log'),
                              ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(os.path.join(opt.result_path, 'train_batch.log'),
                                    ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

        criterion = nn.CrossEntropyLoss()
        if not opt.no_cuda:
            criterion = criterion.cuda()

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.momentum
        optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate,
                              momentum=opt.momentum, dampening=dampening,
                              weight_decay=opt.weight_decay, nesterov=opt.nesterov)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    if not opt.no_val:
        spatial_transform = Compose([Scale(opt.sample_size),
                                     CenterCrop(opt.sample_size),
                                     ToTensor(opt.norm_value),
                                     Normalize(opt.mean, [1, 1, 1])])
        temporal_transform = TemporalBeginCrop(opt.sample_duration)
        if opt.dataset == 'kinetics':
            validation_data = Kinetics(opt.video_path, opt.annotation_path, 'validation', opt.n_val_samples,
                                       spatial_transform, temporal_transform)
        else:
            validation_data = ActivityNet(opt.video_path, opt.annotation_path, 'validation', opt.n_val_samples,
                                          spatial_transform, temporal_transform)
        val_loader = torch.utils.data.DataLoader(validation_data, batch_size=opt.batch_size,
                                                 shuffle=False, num_workers=opt.n_threads, pin_memory=True)
        val_logger = Logger(os.path.join(opt.result_path, 'val.log'),
                            ['epoch', 'loss', 'acc'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs):
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer,
                        opt, train_logger, train_batch_logger)
        if not opt.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, opt, val_logger)

        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)
