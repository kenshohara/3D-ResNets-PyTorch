import argparse
from collections import OrderedDict

import torch

parser = argparse.ArgumentParser()
parser.add_argument('file_path', type=str)
parser.add_argument('--dst_file_path', default=None, type=str)
args = parser.parse_args()

if args.dst_file_path is None:
    args.dst_file_path = args.file_path

x = torch.load(args.file_path)
state_dict = x['state_dict']
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    new_k = '.'.join(k.split('.')[1:])
    new_state_dict[new_k] = v

x['state_dict'] = new_state_dict

torch.save(x, args.dst_file_path)