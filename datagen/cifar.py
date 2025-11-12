import os
import torch
import torchvision
import random
import pickle

from PIL import Image
from datasets.cifar10 import CIFAR10
from platforms.platform import get_platform

from pathlib import Path

import numpy as np
import matplotlib as plt

from .teacher_introspection import TeacherIntrospection
from .datagens_generators import DatagensGenerator



class CIFAR10:

    @classmethod
    def cifar10(cls, augment=None, d_in=None):
        # augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
        train_set = CIFAR10(train=True, root=os.path.join(get_platform().dataset_root, 'cifar10'), download=True)
        transforms = [torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        X = []
        for im in train_set.data:
            for t in transforms:
                im = t(im)
            X.append(im)
        X = torch.stack(X, dim=0)
        return X

    @classmethod
    def cifar10_conv(cls, augment=None, d_in=None):
        X = cifar10(augment, d_in)
        return X

