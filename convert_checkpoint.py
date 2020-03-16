import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
parser.add_argument('-a','--arch', default='resnet-18')

args = parser.parse_args()

cp = torch.load(args.input)

new_cp = {
    'arch': args.arch,
    'state_dict': cp['model'],
    'epoch': cp['epoch']
}

torch.save(new_cp, args.output)
