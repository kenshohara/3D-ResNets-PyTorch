import copy

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate

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

        frame_indices = list(range(1, n_frames + 1))
        sample = {
            'video': video_path,
            'segment': (frame_indices[0], frame_indices[-1] + 1),
            'frame_indices': frame_indices,
            'video_id': video_ids[i],
            'label': label_id
        }
        dataset.append(sample)

    return dataset, idx_to_class


def multi_clips_collate_fn(batch):
    batch_clips, batch_targets = zip(*batch)
    batch_clips = [clip for multi_clips in batch_clips for clip in multi_clips]
    batch_targets = [
        target for multi_targets in batch_targets for target in multi_targets
    ]

    return default_collate(list(zip(batch_clips, batch_targets)))


def collate_fn(batch):
    batch_clips, batch_targets = zip(*batch)

    return default_collate(batch_clips), batch_targets


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

    def loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip

    def temporal_sliding_window(self, sample_duration, sample_stride):
        data = []
        for x in self.data:
            t_begin, t_end = x['segment']
            for t in range(t_begin, t_end, sample_stride):
                sample = copy.deepcopy(x)
                segment = (t, min(t + sample_duration, t_end))
                sample['segment'] = segment
                sample['frame_indices'] = list(range(segment[0], segment[1]))
                data.append(sample)
        self.data = data

    def __getitem__(self, index):
        path = self.data[index]['video']
        target = self.data[index]

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        if isinstance(frame_indices[0], list):
            clips = []
            targets = []
            for one_frame_indices in frame_indices:
                clips.append(self.loading(path, one_frame_indices))

                current_target = target
                current_target['segment'] = [
                    one_frame_indices[0], one_frame_indices[-1] + 1
                ]
                if self.target_transform is not None:
                    current_target = self.target_transform(current_target)
                targets.append(current_target)

            return clips, targets
        else:
            clip = self.loading(path, frame_indices)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return clip, target

    def __len__(self):
        return len(self.data)