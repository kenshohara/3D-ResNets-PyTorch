import math
import json

import torch
import torch.utils.data as data

from .loader import VideoLoader
from .videodataset import VideoDataset


def get_n_frames(video_path):
    return len([
        x for x in video_path.iterdir()
        if 'image' in x.name and x.name[0] != '.'
    ])


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


class ActivityNet(VideoDataset):

    def __init__(
            self,
            root_path,
            annotation_path,
            subset,
            spatial_transform=None,
            temporal_transform=None,
            target_transform=None,
            video_loader=None,
            video_path_formatter=(
                lambda root_path, label, video_id: root_path / f'v_{video_id}'),
            image_name_formatter=lambda x: f'image_{x:05d}.jpg',
            is_untrimmed_setting=False):
        if is_untrimmed_setting:
            self.data, self.class_names = self.__make_untrimmed_dataset(
                root_path, annotation_path, subset, video_path_formatter)
        else:
            self.data, self.class_names = self.__make_dataset(
                root_path, annotation_path, subset, video_path_formatter)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

    def __make_dataset(self, root_path, annotation_path, subset,
                       video_path_formatter):
        with annotation_path.open('r') as f:
            data = json.load(f)
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

            video_path = video_path_formatter(root_path, label, video_ids[i])
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

    def __make_untrimmed_dataset(self, root_path, annotation_path, subset,
                                 video_path_formatter):
        with annotation_path.open('r') as f:
            data = json.load(f)
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

            video_path = video_path_formatter(root_path, label, video_ids[i])
            if not video_path.exists():
                continue

            fps = fps_values[i]

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