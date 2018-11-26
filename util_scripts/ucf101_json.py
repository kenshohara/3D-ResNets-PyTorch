from pathlib import Path
import sys
import json

import pandas as pd

from utils import get_n_frames


def convert_csv_to_dict(csv_path, subset):
    data = pd.read_csv(csv_path, delimiter=' ', header=None)
    keys = []
    key_labels = []
    for i in range(data.shape[0]):
        row = data.iloc[i, :]
        slash_rows = data.iloc[i, 0].split('/')
        class_name = slash_rows[0]
        basename = slash_rows[1].split('.')[0]

        keys.append(basename)
        key_labels.append(class_name)

    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        label = key_labels[i]
        database[key]['annotations'] = {'label': label}

    return database


def load_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, delimiter=' ', header=None)
    labels = []
    for i in range(data.shape[0]):
        labels.append(data.iloc[i, 1])
    return labels


def convert_ucf101_csv_to_json(label_csv_path, train_csv_path, val_csv_path,
                               video_dir_path, dst_json_path):
    labels = load_labels(label_csv_path)
    train_database = convert_csv_to_dict(train_csv_path, 'training')
    val_database = convert_csv_to_dict(val_csv_path, 'validation')

    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)

    for k, v in dst_data['database'].items():
        if v['annotations'] is not None:
            label = v['annotations']['label']
        else:
            label = 'test'

        video_path = video_dir_path / label / k
        n_frames = get_n_frames(video_path)
        v['annotations']['segment'] = (1, n_frames + 1)

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    csv_dir_path = Path(sys.argv[1])
    video_dir_path = Path(sys.argv[2])

    for split_index in range(1, 4):
        label_csv_path = csv_dir_path / 'classInd.txt'
        train_csv_path = csv_dir_path / 'trainlist0{}.txt'.format(split_index)
        val_csv_path = csv_dir_path / 'testlist0{}.txt'.format(split_index)
        dst_json_path = csv_dir_path / 'ucf101_0{}.json'.format(split_index)

        convert_ucf101_csv_to_json(label_csv_path, train_csv_path, val_csv_path,
                                   video_dir_path, dst_json_path)
