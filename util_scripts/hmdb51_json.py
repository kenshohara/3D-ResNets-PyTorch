import argparse
import json
from pathlib import Path

import pandas as pd

from .utils import get_n_frames


def convert_csv_to_dict(csv_dir_path, split_index):
    database = {}
    for file_path in csv_dir_path.iterdir():
        filename = file_path.name
        if 'split{}'.format(split_index) not in filename:
            continue

        data = pd.read_csv(csv_dir_path / filename, delimiter=' ', header=None)
        keys = []
        subsets = []
        for i in range(data.shape[0]):
            row = data.iloc[i, :]
            if row[1] == 0:
                continue
            elif row[1] == 1:
                subset = 'training'
            elif row[1] == 2:
                subset = 'validation'

            keys.append(row[0].split('.')[0])
            subsets.append(subset)

        for i in range(len(keys)):
            key = keys[i]
            database[key] = {}
            database[key]['subset'] = subsets[i]
            label = '_'.join(filename.split('_')[:-2])
            database[key]['annotations'] = {'label': label}

    return database


def get_labels(csv_dir_path):
    labels = []
    for file_path in csv_dir_path.iterdir():
        labels.append('_'.join(file_path.name.split('_')[:-2]))
    return sorted(list(set(labels)))


def convert_hmdb51_csv_to_json(csv_dir_path, split_index, video_dir_path,
                               dst_json_path):
    labels = get_labels(csv_dir_path)
    database = convert_csv_to_dict(csv_dir_path, split_index)

    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(database)

    for k, v in dst_data['database'].items():
        if v['annotations'] is not None:
            label = v['annotations']['label']
        else:
            label = 'test'

        video_path = video_dir_path / label / k
        n_frames = get_n_frames(video_path)
        v['annotations']['segment'] = (1, n_frames + 1)

    with dst_json_path.open('w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_path',
                        default=None,
                        type=Path,
                        help='Directory path of HMDB51 annotation files.')
    parser.add_argument('video_path',
                        default=None,
                        type=Path,
                        help=('Path of video directory (jpg).'
                              'Using to get n_frames of each video.'))
    parser.add_argument('dst_dir_path',
                        default=None,
                        type=Path,
                        help='Directory path of dst json file.')

    args = parser.parse_args()

    for split_index in range(1, 4):
        dst_json_path = args.dst_dir_path / 'hmdb51_{}.json'.format(split_index)
        convert_hmdb51_csv_to_json(args.dir_path, split_index, args.video_path,
                                   dst_json_path)
