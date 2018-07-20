import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import random

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
            print(frame_indices)
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration, sample_stride):
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

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1],
            'frame_indices': []
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        clip_length = sample_duration * sample_stride
        # print('='*80)
        if n_samples_for_each_video <= 0:
            n_samples_for_cur_video = math.ceil(n_frames/clip_length) # sample times for current video
        else:
           n_samples_for_cur_video = n_samples_for_each_video
        
        if n_samples_for_cur_video == 1:
            step = n_frames
        else:
            step = max( 1, 
                        (n_frames-clip_length)/(n_samples_for_cur_video-1) )
        
        for i in range(n_samples_for_cur_video):
            sample_i = copy.deepcopy(sample)
            if step < clip_length:
                random_offset = random.randint(0, sample_stride-1)
            else:
                random_offset = random.randint(0, math.floor(step-clip_length+sample_stride-1))

            start_frame = math.floor(i*step+random_offset)
            for j in range(sample_duration):
                sample_i['frame_indices'].append(start_frame % n_frames + 1)
                start_frame += sample_stride
            dataset.append(sample_i)
            # print('sample:', video_path, n_frames, sample_i['frame_indices'][0], len(sample_i['frame_indices']) )
    print('total samples:', len(dataset))
    return dataset, idx_to_class


class UCF101(data.Dataset):
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
                 sample_duration=8,
                 sample_stride=8,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration, sample_stride)

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

def unittest():
    root_path = '/home/dxx/UCF101/'
    video_path = os.path.join(root_path, 'UCF-101_frm')
    annotation_path=os.path.join(root_path, 'ucfTrainTestlist', 'ucf101_01.json')
    subset='training'
    n_samples_for_each_video = 2
    sample_duration=8
    sample_stride=8
    dataset = UCF101( root_path=video_path, 
                      annotation_path=annotation_path,
                      subset=subset,
                      n_samples_for_each_video=n_samples_for_each_video,
                      sample_duration=sample_duration,
                      sample_stride=sample_stride
                    )

    def _check(a, b):
        assert a == b, \
            '{} vs {}'.format(a, b)
    
    videos = 9537 if subset=='training' else 3783
    _check(len(dataset), videos*n_samples_for_each_video)

    d = {}
    for sample in dataset.data:
        _check(len(sample['frame_indices']), sample_duration)
        
        try:
            fisrt_frm = sample['frame_indices'][0]
            last_frm = sample['frame_indices'][-1]
            _check( (fisrt_frm+sample_duration*(sample_stride-1))%sample['n_frames'], \
                    last_frm )
        except AssertionError:
            raise AssertionError("{} vs {}".format(fisrt_frm, last_frm))
        
        video_id = sample['video']
        if not video_id in d:
            d[video_id] = 0
        d[video_id] += 1

    _check(len(d), videos)
    
    for v_id in d:
        try:
            _check(d[v_id], n_samples_for_each_video)
        except AssertionError:
            raise AssertionError('{}:{} vs {}'.format(v_id, d[v_id], n_samples_for_each_video))

   # TODO: check if image exists 

if __name__ == '__main__':
    unittest()
