from pathlib import Path
import json
import random

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from torch.backends import cudnn
import torchvision

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Resize, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    RandomResizedCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalEvenCrop
from target_transforms import ClassLabel, VideoID
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, worker_init_fn
from train import train_epoch
from validation import val_epoch
import test


def get_opts():
    opt = parse_opts()

    if opt.root_path:
        opt.video_path = opt.root_path / opt.video_path
        opt.annotation_path = opt.root_path / opt.annotation_path
        opt.result_path = opt.root_path / opt.result_path
        if opt.resume_path:
            opt.resume_path = opt.root_path / opt.resume_path
        if opt.pretrain_path:
            opt.pretrain_path = opt.root_path / opt.pretrain_path

    if opt.pretrain_path:
        opt.n_finetune_classes = opt.n_classes
        opt.n_classes = opt.n_pretrain_classes

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    with open(opt.result_path / 'opts.json', 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    return opt


def resume(opt, model):
    print('loading checkpoint {}'.format(opt.resume_path))
    checkpoint = torch.load(opt.resume_path)
    assert opt.arch == checkpoint['arch']

    opt.begin_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    if not opt.no_train:
        optimizer.load_state_dict(checkpoint['optimizer'])


def get_norm_method(opt):
    if opt.no_mean_norm and not opt.std_norm:
        return Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        return Normalize(opt.mean, [1, 1, 1])
    else:
        return Normalize(opt.mean, opt.std)


def get_train_utils(opt):
    assert opt.train_crop in ['random', 'corner']
    if opt.train_crop == 'random':
        crop_method = RandomResizedCrop(
            opt.sample_size, (opt.train_crop_min_scale, 1.0),
            (opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio))
    elif opt.train_crop == 'corner':
        scales = [1.0]
        scale_step = 1 / (2**(1 / 4))
        for _ in range(1, 5):
            scales.append(scales[-1] * scale_step)
        crop_method = MultiScaleCornerCrop(opt.sample_size, scales)
    spatial_transform = Compose(
        [crop_method,
         RandomHorizontalFlip(),
         ToTensor(),
         get_norm_method(opt)])
    temporal_transform = TemporalRandomCrop(opt.sample_duration)
    target_transform = ClassLabel()
    training_data = get_training_set(opt, spatial_transform, temporal_transform,
                                     target_transform)
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_threads,
        pin_memory=True,
        worker_init_fn=worker_init_fn)
    train_logger = Logger(opt.result_path / 'train.log',
                          ['epoch', 'loss', 'acc', 'lr'])
    train_batch_logger = Logger(opt.result_path / 'train_batch.log',
                                ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
    optimizer = SGD(
        parameters,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=opt.lr_patience)

    return train_loader, train_logger, train_batch_logger, optimizer, scheduler


def get_val_utils(opt):
    spatial_transform = Compose([
        Resize(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor(),
        get_norm_method(opt)
    ])
    temporal_transform = TemporalEvenCrop(opt.sample_duration,
                                          opt.n_val_samples)
    target_transform = ClassLabel()
    validation_data = get_validation_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
        worker_init_fn=worker_init_fn)
    val_logger = Logger(opt.result_path / 'val.log', ['epoch', 'loss', 'acc'])

    return val_loader, val_logger


def get_test_utils(opt):
    spatial_transform = Compose([
        Resize(int(opt.sample_size / opt.scale_in_test)),
        CornerCrop(opt.sample_size, opt.crop_position_in_test),
        ToTensor(),
        get_norm_method(opt)
    ])
    temporal_transform = LoopPadding(opt.sample_duration)
    target_transform = VideoID()

    test_data = get_test_set(opt, spatial_transform, temporal_transform,
                             target_transform)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
        worker_init_fn=worker_init_fn)

    return test_loader, test_data.class_names


if __name__ == '__main__':
    opt = get_opts()

    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True
    if opt.accimage:
        torchvision.set_image_backend('accimage')
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)
    criterion = CrossEntropyLoss().to(opt.device)

    if not opt.no_train:
        (train_loader, train_logger, train_batch_logger, optimizer,
         scheduler) = get_train_utils(opt)
    if not opt.no_val:
        val_loader, val_logger = get_val_utils(opt)

    if opt.resume_path:
        resume(opt, model)

    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
        if not opt.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)

        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)

    if opt.test:
        test_loader, test_class_names = get_test_utils(opt)
        test.test(test_loader, model, opt, test_class_names)
