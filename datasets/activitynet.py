import torch
import torch.utils.data as data
from PIL import Image
import os
import functools
import json
import copy
import math

from utils import load_value_file


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_names = []
    index = 0
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


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            if subset == 'testing':
                video_names.append('v_{}'.format(key))
            else:
                video_names.append('v_{}'.format(key))
                annotations.append(value['annotations'])

    return video_names, annotations


def modify_frame_indices(video_dir_path, frame_indices):
    modified_indices = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if not os.path.exists(image_path):
            return modified_indices
        modified_indices.append(i)
    return modified_indices


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            continue

        fps_file_path = os.path.join(video_path, 'fps')
        fps = load_value_file(fps_file_path)

        for annotation in annotations[i]:
            begin_t = math.ceil(annotation['segment'][0] * fps)
            end_t = math.ceil(annotation['segment'][1] * fps)
            if begin_t == 0:
                begin_t = 1
            n_frames = end_t - begin_t

            sample = {
                'video': video_path,
                'segment': [begin_t, end_t],
                'fps': fps,
                'video_id': video_names[i][2:]
            }
            if len(annotations) != 0:
                sample['label'] = class_to_idx[annotation['label']]
            else:
                sample['label'] = -1

            if n_samples_for_each_video == 1:
                frame_indices = list(range(1, n_frames + 1))
                frame_indices = modify_frame_indices(sample['video'],
                                                     frame_indices)
                if len(frame_indices) < 16:
                    continue
                sample['frame_indices'] = frame_indices
                dataset.append(sample)
            else:
                if n_samples_for_each_video > 1:
                    step = max(1,
                               math.ceil((n_frames - 1 - sample_duration) /
                                         (n_samples_for_each_video - 1)))
                else:
                    step = sample_duration
                for j in range(1, n_frames, step):
                    sample_j = copy.deepcopy(sample)
                    frame_indices = list(range(j, j + sample_duration))
                    frame_indices = modify_frame_indices(
                        sample_j['video'], frame_indices)
                    if len(frame_indices) < 16:
                        continue
                    sample_j['frame_indices'] = frame_indices
                    dataset.append(sample_j)

    return dataset, idx_to_class


class ActivityNet(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
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
