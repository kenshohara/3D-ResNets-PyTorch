import h5py


def get_n_frames(video_path):
    return len([
        x for x in video_path.iterdir()
        if 'image' in x.name and x.name[0] != '.'
    ])


def get_n_frames_hdf5(video_path):
    with h5py.File(video_path, 'r') as f:
        video_data = f['video']
        return len(video_data)