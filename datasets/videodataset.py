import json
import copy
import functools

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
import torchvision
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with path.open('rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(str(path))
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader,
                 file_name_formatter):
    video = []
    for i in frame_indices:
        image_path = video_dir_path / file_name_formatter(i)
        if image_path.exists():
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    file_name_formatter = lambda x: 'image_{:05d}.jpg'.format(x)
    return functools.partial(video_loader,
                             image_loader=image_loader,
                             file_name_formatter=file_name_formatter)


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
            video_ids.append(key)
            annotations.append(value['annotations'])

    return video_ids, annotations


def make_dataset(root_path, annotation_path, subset, video_path_formatter):
    with annotation_path.open('r') as f:
        data = json.load(f)
    video_ids, annotations = get_video_ids_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    n_videos = len(video_ids)
    dataset = []
    for i in range(n_videos):
        if i % (n_videos // 5) == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_ids)))

        if 'label' in annotations[i]:
            label = annotations[i]['label']
            label_id = class_to_idx[label]
        else:
            label = 'test'
            label_id = -1

        video_path = video_path_formatter(root_path, label, video_ids[i])
        if not video_path.exists():
            continue

        segment = annotations[i]['segment']
        if segment[1] == 1:
            continue

        frame_indices = list(range(segment[0], segment[1]))
        sample = {
            'video': video_path,
            'segment': segment,
            'frame_indices': frame_indices,
            'video_id': video_ids[i],
            'label': label_id
        }
        dataset.append(sample)

    return dataset, idx_to_class


def collate_fn(batch):
    batch_clips, batch_targets = zip(*batch)

    if isinstance(batch_clips[0], list):
        batch_clips = [
            clip for multi_clips in batch_clips for clip in multi_clips
        ]
        batch_targets = [
            target for multi_targets in batch_targets
            for target in multi_targets
        ]

    if isinstance(batch_targets[0], int):
        return default_collate(batch_clips), default_collate(batch_targets)
    else:
        return default_collate(batch_clips), batch_targets


class VideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=get_default_video_loader(),
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id)):
        self.data, self.class_names = make_dataset(root_path, annotation_path,
                                                   subset, video_path_formatter)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = video_loader

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