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
from mean import get_mean_std
from spatial_transforms import (
    Compose, Normalize, Resize, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    RandomResizedCrop, RandomHorizontalFlip, ToTensor, ScaleValue, ColorJitter)
from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalEvenCrop, SlidingWindow)
from target_transforms import ClassLabel, VideoID, Segment
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, worker_init_fn, get_lr
from train import train_epoch
from validation import val_epoch
import test


def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)


def get_opt():
    opt = parse_opts()

    if opt.root_path is not None:
        opt.video_path = opt.root_path / opt.video_path
        opt.annotation_path = opt.root_path / opt.annotation_path
        opt.result_path = opt.root_path / opt.result_path
        if opt.resume_path is not None:
            opt.resume_path = opt.root_path / opt.resume_path
        if opt.pretrain_path is not None:
            opt.pretrain_path = opt.root_path / opt.pretrain_path

    if opt.pretrain_path is not None:
        opt.n_finetune_classes = opt.n_classes
        opt.n_classes = opt.n_pretrain_classes

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1
    opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
    print(opt)
    with open(opt.result_path / 'opts.json', 'w') as opt_file:
        json.dump(vars(opt), opt_file, default=json_serial)

    return opt


def resume(resume_path,
           arch,
           begin_epoch,
           model,
           optimizer=None,
           scheduler=None):
    print('loading checkpoint {}'.format(resume_path))
    checkpoint = torch.load(resume_path)
    assert arch == checkpoint['arch']

    begin_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None and 'optimizer' in checkpoint.keys():
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint.keys():
        scheduler.load_state_dict(checkpoint['scheduler'])

    return begin_epoch, model, optimizer, scheduler


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


def get_train_utils(opt, model_parameters):
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
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    spatial_transform = Compose([
        crop_method,
        RandomHorizontalFlip(),
        ToTensor(),
        ScaleValue(opt.value_scale), normalize
    ])
    temporal_transform = TemporalRandomCrop(opt.sample_duration)
    target_transform = ClassLabel()

    training_data, collate_fn = get_training_set(
        opt.video_path, opt.annotation_path, opt.dataset, spatial_transform,
        temporal_transform, target_transform)
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_threads,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn)
    train_logger = Logger(opt.result_path / 'train.log',
                          ['epoch', 'loss', 'acc', 'lr'])
    train_batch_logger = Logger(opt.result_path / 'train_batch.log',
                                ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
    optimizer = SGD(
        model_parameters,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov)

    assert opt.lr_scheduler in ['plateau', 'multistep']
    if opt.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.plateau_patience)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             opt.multistep_milestones)

    return train_loader, train_logger, train_batch_logger, optimizer, scheduler


def get_val_utils(opt):
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    spatial_transform = Compose([
        Resize(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor(),
        ScaleValue(opt.value_scale), normalize
    ])
    temporal_transform = TemporalEvenCrop(opt.sample_duration,
                                          opt.n_val_samples)
    target_transform = ClassLabel()
    validation_data, collate_fn = get_validation_set(
        opt.video_path, opt.annotation_path, opt.dataset, spatial_transform,
        temporal_transform, target_transform)
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=(opt.batch_size // opt.n_val_samples),
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn)
    val_logger = Logger(opt.result_path / 'val.log', ['epoch', 'loss', 'acc'])

    return val_loader, val_logger


def get_test_utils(opt):
    assert opt.test_crop in ['center', 'nocrop']

    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)

    spatial_transform = [
        Resize(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor(),
        ScaleValue(opt.value_scale), normalize
    ]
    if opt.test_crop == 'center':
        temporal_transform = LoopPadding(opt.sample_duration)
    else:
        del spatial_transform[1]  #remove CenterCrop
        temporal_transform = SlidingWindow(opt.sample_duration, opt.test_stride)
    spatial_transform = Compose(spatial_transform)
    target_transform = TargetCompose([VideoID(), Segment()])

    test_data, collate_fn = get_test_set(
        opt.video_path, opt.annotation_path, opt.dataset, opt.test_subset,
        spatial_transform, temporal_transform, target_transform)

    if opt.test_crop == 'center':
        test_data.temporal_sliding_window(opt.sample_duration, opt.test_stride)
        batch_size = opt.batch_size
    else:
        batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn)

    return test_loader, test_data.class_names


def save_checkpoint(save_file_path, epoch, arch, model, optimizer, scheduler):
    save_states = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)


if __name__ == '__main__':
    opt = get_opt()

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
         scheduler) = get_train_utils(opt, parameters)
    if not opt.no_val:
        val_loader, val_logger = get_val_utils(opt)

    if opt.resume_path is not None:
        if not opt.no_train:
            opt.begin_epoch, model, optimizer, scheduler = resume(
                opt.resume_path, opt.arch, opt.begin_epoch, model, optimizer,
                scheduler)
        else:
            opt.begin_epoch, model, _, _ = resume(opt.resume_path, opt.arch,
                                                  opt.begin_epoch, model)

    prev_val_loss = None
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            if opt.lr_scheduler == 'multistep':
                scheduler.step()
            elif (opt.lr_scheduler == 'plateau' and not opt.no_val and
                  prev_val_loss is not None):
                scheduler.step(prev_val_loss)

            current_lr = get_lr(optimizer)
            train_epoch(i, train_loader, model, criterion, optimizer,
                        opt.device, current_lr, train_logger,
                        train_batch_logger)

            if i % opt.checkpoint == 0:
                save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                scheduler)

        if not opt.no_val:
            prev_val_loss = val_epoch(i, val_loader, model, criterion,
                                      opt.device, val_logger)

    if opt.test:
        test_loader, test_class_names = get_test_utils(opt)
        test_result_path = opt.result_path / '{}.json'.format(opt.test_subset)

        test.test(test_loader, model, test_result_path, test_class_names,
                  opt.test_no_average)
