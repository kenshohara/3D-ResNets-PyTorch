import math

import torch
import torch.utils.data as data

from .utils import (get_default_video_loader, get_n_frames,
                    load_annotation_data)
from .videodataset import VideoDataset


def get_class_labels(data):
    class_names = []
    for node1 in data['taxonomy']:
        is_leaf = True
        for node2 in data['taxonomy']:
            if node2['parentId'] == node1['nodeId']:
                is_leaf = False
                break
        if is_leaf:
            class_names.append(node1['nodeName'])

    class_labels_map = {}

    for i, class_name in enumerate(class_names):
        class_labels_map[class_name] = i

    return class_labels_map


def get_video_ids_annotations_and_fps(data, subset):
    video_ids = []
    annotations = []
    fps_values = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            video_ids.append(key)
            annotations.append(value['annotations'])
            fps_values.append(value['fps'])

    return video_ids, annotations, fps_values


def make_dataset(root_path, annotation_path, subset):
    data = load_annotation_data(annotation_path)
    video_ids, annotations, fps_values = get_video_ids_annotations_and_fps(
        data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_ids)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_ids)))

        video_path = root_path / 'v_{}'.format(video_ids[i])
        if not video_path.exists():
            continue

        fps = fps_values[i]

        for annotation in annotations[i]:
            t_begin = math.floor(annotation['segment'][0] * fps) + 1
            t_end = math.floor(annotation['segment'][1] * fps) + 1
            n_video_frames = get_n_frames(video_path)
            t_end = min(t_end, n_video_frames)
            frame_indices = list(range(t_begin, t_end))

            sample = {
                'video': video_path,
                'segment': (frame_indices[0], frame_indices[-1] + 1),
                'frame_indices': frame_indices,
                'fps': fps,
                'video_id': video_ids[i]
            }
            if annotations is not None:
                sample['label'] = class_to_idx[annotation['label']]
            else:
                sample['label'] = -1

            if len(sample['frame_indices']) < 8:
                continue
            dataset.append(sample)

    return dataset, idx_to_class


def make_untrimmed_dataset(root_path, annotation_path, subset):
    data = load_annotation_data(annotation_path)
    video_ids, _ = get_video_ids_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_ids)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_ids)))

        video_path = root_path / 'v_{}'.format(video_ids[i])
        if not video_path.exists():
            continue

        fps_file_path = video_path / 'fps'
        fps = load_value_file(fps_file_path)

        t_begin = 1
        t_end = get_n_frames(video_path) + 1
        frame_indices = list(range(t_begin, t_end))

        sample = {
            'video': video_path,
            'segment': (frame_indices[0], frame_indices[-1] + 1),
            'frame_indices': frame_indices,
            'fps': fps,
            'video_id': video_ids[i]
        }
        dataset.append(sample)

    return dataset, idx_to_class


class ActivityNet(VideoDataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 is_untrimmed_setting=False,
                 get_loader=get_default_video_loader):
        if is_untrimmed_setting:
            self.data, self.class_names = make_untrimmed_dataset(
                root_path, annotation_path, subset)
        else:
            self.data, self.class_names = make_dataset(root_path,
                                                       annotation_path, subset)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()