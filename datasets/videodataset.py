import copy

import torch
import torch.utils.data as data

from .utils import (get_default_video_loader, get_n_frames,
                    load_annotation_data)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_ids_and_annotations(data, subset):
    video_ids = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            if subset == 'testing':
                video_ids.append(key)
            else:
                video_ids.append(key)
                annotations.append(value['annotations'])

    if subset == 'testing':
        return video_ids, None
    else:
        return video_ids, annotations


def make_dataset(root_path, annotation_path, subset):
    data = load_annotation_data(annotation_path)
    video_ids, annotations = get_video_ids_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_ids)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_ids)))

        if annotations is not None:
            label = annotations[i]['label']
            label_id = class_to_idx[label]
        else:
            label_id = -1

        video_path = root_path / label / video_ids[i]
        if not video_path.exists():
            continue

        n_frames = get_n_frames(video_path)
        if n_frames == 0:
            continue

        begin_t = 1
        end_t = n_frames

        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'frame_indices': list(range(1, n_frames + 1)),
            'video_id': video_ids[i],
            'label': label_id
        }
        dataset.append(sample)

    return dataset, idx_to_class


class VideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(root_path, annotation_path,
                                                   subset)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)