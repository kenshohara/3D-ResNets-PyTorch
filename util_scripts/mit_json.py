import argparse
import json
from pathlib import Path

import pandas as pd

from .utils import get_n_frames


def convert_csv_to_dict(csv_path, subset):
    data = pd.read_csv(csv_path, header=None)
    keys = []
    key_labels = []
    if subset == 'testing':
        for i in range(data.shape[0]):
            basename = data.iloc[i, 0].split('/')
            assert len(basename) == 1
            basename = Path(basename[0]).stem

            keys.append(basename)
    else:
        for i in range(data.shape[0]):
            basename = data.iloc[i, 0].split('/')
            assert len(basename) == 2
            basename = Path(basename[1]).stem

            keys.append(basename)
            key_labels.append(data.iloc[i, 1])

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
    data = pd.read_csv(train_csv_path, header=None)
    return data.iloc[:, 0].tolist()


def convert_mit_csv_to_json(class_file_path, train_csv_path, val_csv_path,
                            test_csv_path, video_dir_path, dst_json_path):
    labels = load_labels(class_file_path)
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
        help=('Directory path including moments_categories.txt, '
              'trainingSet.csv, validationSet.csv, '
              '(testingSet.csv (optional))'))
    parser.add_argument('video_path',
                        default=None,
                        type=Path,
                        help=('Path of video directory (jpg).'
                              'Using to get n_frames of each video.'))
    parser.add_argument('dst_path',
                        default=None,
                        type=Path,
                        help='Path of dst json file.')

    args = parser.parse_args()

    class_file_path = args.dir_path / 'moments_categories.txt'
    train_csv_path = args.dir_path / 'trainingSet.csv'
    val_csv_path = args.dir_path / 'validationSet.csv'
    test_csv_path = args.dir_path / 'testingSet.csv'

    convert_mit_csv_to_json(class_file_path, train_csv_path, val_csv_path,
                            test_csv_path, args.video_path, args.dst_path)
