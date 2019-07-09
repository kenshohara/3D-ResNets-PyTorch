from datasets.videodataset import VideoDataset
from datasets.videodataset_multiclips import (VideoDatasetMultiClips,
                                              collate_fn)
from datasets.activitynet import ActivityNet
from datasets.loader import VideoLoader, VideoLoaderHDF5


def get_training_set(video_path,
                     annotation_path,
                     dataset_name,
                     file_type,
                     spatial_transform=None,
                     temporal_transform=None,
                     target_transform=None):
    assert dataset_name in [
        'kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit'
    ]
    assert file_type in ['jpg', 'hdf5']

    if dataset_name == 'activitynet':
        training_data = ActivityNet(video_path,
                                    annotation_path,
                                    'training',
                                    spatial_transform=spatial_transform,
                                    temporal_transform=temporal_transform,
                                    target_transform=target_transform)
    else:
        if file_type == 'jpg':
            loader = VideoLoader(lambda x: f'image_{x:05d}.jpg')
            video_path_formatter = (
                lambda root_path, label, video_id: root_path / label / video_id)
        else:
            loader = VideoLoaderHDF5()
            video_path_formatter = (lambda root_path, label, video_id: root_path
                                    / label / f'{video_id}.hdf5')

        training_data = VideoDataset(video_path,
                                     annotation_path,
                                     'training',
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     target_transform=target_transform,
                                     video_loader=loader,
                                     video_path_formatter=video_path_formatter)

    return training_data


def get_validation_set(video_path,
                       annotation_path,
                       dataset_name,
                       file_type,
                       spatial_transform=None,
                       temporal_transform=None,
                       target_transform=None):
    assert dataset_name in [
        'kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit'
    ]
    assert file_type in ['jpg', 'hdf5']

    if dataset_name == 'activitynet':
        validation_data = ActivityNet(video_path, annotation_path, 'validation',
                                      spatial_transform, temporal_transform,
                                      target_transform)
    else:
        if file_type == 'jpg':
            loader = VideoLoader(lambda x: f'image_{x:05d}.jpg')
            video_path_formatter = (
                lambda root_path, label, video_id: root_path / label / video_id)
        else:
            loader = VideoLoaderHDF5()
            video_path_formatter = (lambda root_path, label, video_id: root_path
                                    / label / f'{video_id}.hdf5')

        validation_data = VideoDataset(
            video_path,
            annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter)

    return validation_data


def get_test_set(video_path,
                 annotation_path,
                 dataset_name,
                 file_type,
                 test_subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None):
    assert dataset_name in [
        'kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit'
    ]
    assert file_type in ['jpg', 'hdf5']
    assert test_subset in ['train', 'val', 'test']

    if test_subset == 'train':
        subset = 'training'
    elif test_subset == 'val':
        subset = 'validation'
    elif test_subset == 'test':
        subset = 'testing'
    if dataset_name == 'activitynet':
        test_data = ActivityNet(video_path,
                                annotation_path,
                                subset,
                                spatial_transform,
                                temporal_transform,
                                target_transform,
                                is_untrimmed_setting=True)
    else:
        if file_type == 'jpg':
            loader = VideoLoader(lambda x: f'image_{x:05d}.jpg')
            video_path_formatter = (
                lambda root_path, label, video_id: root_path / label / video_id)
        else:
            loader = VideoLoaderHDF5()
            video_path_formatter = (lambda root_path, label, video_id: root_path
                                    / label / f'{video_id}.hdf5')

        test_data = VideoDatasetMultiClips(
            video_path,
            annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    return test_data, collate_fn