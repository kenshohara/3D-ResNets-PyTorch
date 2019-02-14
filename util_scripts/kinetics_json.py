import argparse
import json
from pathlib import Path

import pandas as pd

from utils import get_n_frames


def convert_csv_to_dict(csv_path, subset):
    data = pd.read_csv(csv_path)
    keys = []
    key_labels = []
    for i in range(data.shape[0]):
        row = data.iloc[i, :]
        basename = '%s_%s_%s' % (row['youtube_id'], '%06d' % row['time_start'],
                                 '%06d' % row['time_end'])
        keys.append(basename)
        if subset != 'testing':
            key_labels.append(row['label'])

    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        if subset != 'testing':
            label = key_labels[i]
            database[key]['annotations'] = {'label': label}
        else:
            database[key]['annotations'] = {}

    return database


def load_labels(train_csv_path):
    data = pd.read_csv(train_csv_path)
    return data['label'].unique().tolist()


def convert_kinetics_csv_to_json(train_csv_path, val_csv_path, test_csv_path,
                                 video_dir_path, dst_json_path):
    labels = load_labels(train_csv_path)
    train_database = convert_csv_to_dict(train_csv_path, 'training')
    val_database = convert_csv_to_dict(val_csv_path, 'validation')
    if test_csv_path.exists():
        test_database = convert_csv_to_dict(test_csv_path, 'testing')

    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)
    if test_csv_path.exists():
        dst_data['database'].update(test_database)

    for k, v in dst_data['database'].items():
        if 'label' in v['annotations']:
            label = v['annotations']['label']
        else:
            label = 'test'

        video_path = video_dir_path / label / k
        if video_path.exists():
            n_frames = get_n_frames(video_path)
            v['annotations']['segment'] = (1, n_frames + 1)

    with dst_json_path.open('w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir_path',
        default=None,
        type=Path,
        help=('Directory path including '
              'kinetics_train.csv, kinetics_val.csv, '
              '(kinetics_test.csv (optional))'))
    parser.add_argument(
        'n_classes',
        default=400,
        type=int,
        help='400 or 600 (Kinetics-400 or Kinetics-600)')
    parser.add_argument(
        'video_path',
        default=None,
        type=Path,
        help=('Path of video directory (jpg).'
              'Using to get n_frames of each video.'))
    parser.add_argument(
        'dst_path', default=None, type=Path, help='Path of dst json file.')

    args = parser.parse_args()

    train_csv_path = (
        args.dir_path / 'kinetics-{}_train.csv'.format(args.n_classes))
    val_csv_path = (
        args.dir_path / 'kinetics-{}_val.csv'.format(args.n_classes))
    test_csv_path = (
        args.dir_path / 'kinetics-{}_test.csv'.format(args.n_classes))

    convert_kinetics_csv_to_json(train_csv_path, val_csv_path, test_csv_path,
                                 args.video_path, args.dst_path)
