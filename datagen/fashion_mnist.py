

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


class FashionMNIST:

    @classmethod
    def fashion_mnist_test(cls, augment=None, d_in=None):
        train_set = torchvision.datasets.FashionMNIST(
            train=False, root=os.path.join(get_platform().dataset_root, 'fashion_mnist'), download=True)
        transforms = [torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])]
        X = []
        for im in train_set.data.numpy():
            im = Image.fromarray(im, mode='L')
            for t in transforms:
                im = t(im)
            X.append(im)
        X = torch.concat(X)
        return X

    @classmethod
    def fashion_mnist(cls, augment=None, d_in=None):
        train_set = torchvision.datasets.FashionMNIST(
            train=True, root=os.path.join(get_platform().dataset_root, 'fashion_mnist'), download=True)
        transforms = [torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])]
        X = []
        for im in train_set.data.numpy():
            im = Image.fromarray(im, mode='L')
            for t in transforms:
                im = t(im)
            X.append(im)
        X = torch.concat(X)
        return X

