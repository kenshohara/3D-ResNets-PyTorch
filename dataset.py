from datasets.videodataset import VideoDataset
from datasets.activitynet import ActivityNet


def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit']

    if opt.dataset == 'activitynet':
        training_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            'training',
            False,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    else:
        training_data = VideoDataset(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit']

    if opt.dataset == 'activitynet':
        validation_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            'validation',
            False,
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    else:
        validation_data = VideoDataset(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit']
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'activitynet':
        test_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            subset,
            True,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    else:
        test_data = VideoDataset(
            opt.video_path,
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    return test_data
