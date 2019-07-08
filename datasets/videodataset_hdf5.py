import io

import h5py
from PIL import Image

from .videodataset import VideoDataset


def video_loader(video_file_path, frame_indices):
    with h5py.File(video_file_path, 'r') as f:
        video_data = f['video']

        video = []
        for i in frame_indices:
            if i < len(video_data):
                video.append(Image.open(io.BytesIO(video_data[i])))
            else:
                return video

    return video


class VideoDatasetHDF5(VideoDataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None):
        super().__init__(
            root_path,
            annotation_path,
            subset,
            spatial_transform,
            temporal_transform,
            target_transform,
            video_loader=video_loader,
            video_path_formatter=(lambda root_path, label, video_id: root_path /
                                  label / f'{video_id}.hdf5'))