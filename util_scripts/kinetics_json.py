import pandas as pd

import sys
import json
from pathlib import Path

from .utils import get_n_frames


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
    test_database = convert_csv_to_dict(test_csv_path, 'testing')

    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)
    dst_data['database'].update(test_database)

    for k, v in dst_data['database'].items():
        if v['annotations'] is not None:
            label = v['annotations']['label']
        else:
            label = 'test'

        video_path = video_dir_path / label / k
        n_frames = get_n_frames(video_path)
        v['annotations']['segment'] = (1, n_frames)

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == "__main__":
    train_csv_path = Path(sys.argv[1])
    val_csv_path = Path(sys.argv[2])
    test_csv_path = Path(sys.argv[3])
    video_dir_path = Path(sys.argv[4])
    dst_json_path = Path(sys.argv[5])

    convert_kinetics_csv_to_json(train_csv_path, val_csv_path, test_csv_path,
                                 video_dir_path, dst_json_path)
