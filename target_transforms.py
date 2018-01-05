import random
import math


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, target):
        dst = []
        for t in self.transforms:
            dst.append(t(target))
        return dst


class ClassLabel(object):

    def __call__(self, target):
        return target['label']


class VideoID(object):

    def __call__(self, target):
        return target['video_id']
