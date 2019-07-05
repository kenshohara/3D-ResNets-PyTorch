from datasets.activitynet import ActivityNet


def get_training_set(video_path, annotation_path, dataset_name, file_type,
                     spatial_transform, temporal_transform, target_transform):
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
            from datasets.videodataset import VideoDataset
        else:
            from datasets.videodataset_hdf5 import VideoDatasetHDF5 as VideoDataset

        training_data = VideoDataset(video_path,
                                     annotation_path,
                                     'training',
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     target_transform=target_transform)

    return training_data


def get_validation_set(video_path, annotation_path, dataset_name, file_type,
                       spatial_transform, temporal_transform, target_transform):
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
            from datasets.videodataset import VideoDataset
        else:
            from datasets.videodataset_hdf5 import VideoDatasetHDF5 as VideoDataset

        validation_data = VideoDataset(video_path,
                                       annotation_path,
                                       'validation',
                                       spatial_transform=spatial_transform,
                                       temporal_transform=temporal_transform,
                                       target_transform=target_transform)

    return validation_data


def get_test_set(video_path, annotation_path, dataset_name, file_type,
                 test_subset, spatial_transform, temporal_transform,
                 target_transform):
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
            from datasets.videodataset import VideoDataset
        else:
            from datasets.videodataset_hdf5 import VideoDatasetHDF5 as VideoDataset

        test_data = VideoDataset(video_path,
                                 annotation_path,
                                 subset,
                                 spatial_transform=spatial_transform,
                                 temporal_transform=temporal_transform,
                                 target_transform=target_transform)

    return test_data
