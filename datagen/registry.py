"""
 # Created on 13.09.23
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: A registry for all data generation policies
 #
"""

import torch
import numpy as np

from foundations.hparams import DatasetHparams
from platforms.platform import get_platform

from datagen.mnist import MNIST
from datagen.mnist_composition_sampling import MNISTCompositionSampling
from datagen.mnist_random_noise import MNISTRandomNoise
from datagen.mnist_random_noise_overlay import MNISTRandomNoiseOverlay
from datagen.visualization_2D_generators import Visualization2DGenerators
from datagen.data_points_test import DataPointsTest
from datagen.cifar import CIFAR10
from datagen.fashion_mnist import FashionMNIST
from datagen.teacher_introspection import TeacherIntrospectionCaller
from datagen.hyperplanes_teacher_aware_data_generation import HyperplanesTeacherAwareGeneration

from foundations.precision import Precision

registered_datagen_classes = [
    MNIST,
    CIFAR10,
    FashionMNIST,
    MNISTRandomNoise,
    MNISTRandomNoiseOverlay,
    MNISTCompositionSampling,
    DataPointsTest,
    Visualization2DGenerators,
    TeacherIntrospectionCaller,
    HyperplanesTeacherAwareGeneration,
]

def method_searcher(cls_, method_name):
    if not hasattr(cls_, method_name):
        return None

    method = getattr(cls_, method_name)

    if method.__self__ is not cls_: # Verify that it is a classmethod
        return None

    return method.__func__

def get(dataset_hparams: DatasetHparams, model, use_augmentation, output_location = "", train_hparams = None):
    # Get the data generator policies and attach the resulting inputs together
    print("Generating dataset with teacher...")
    X = []
    if type(dataset_hparams.d_in) is not int:
        raise ValueError('Need to define input dimension --d_in: {} '.format(dataset_hparams.d_in))
    for policy in dataset_hparams.datagen.split("+"):

        method = None
        for datagen_class in registered_datagen_classes:
            query_value = method_searcher(datagen_class, policy)

            if query_value is not None:
                method = query_value
                class_ = datagen_class 
                break

        if method is None:
            raise ValueError('Could not find provided datagen: {}'.format(policy))

        if "teacher_aware" in policy:
            X.append(method(class_, use_augmentation, dataset_hparams.d_in, model))
        elif policy.startswith("viz_"):
            X.append(method(class_, model = model, output_location = output_location))
        else:
            X.append(method(class_, use_augmentation, dataset_hparams.d_in))
    with torch.no_grad():
        X = torch.concat(X, dim=0)

        if train_hparams is not None:
            precision = Precision.get_precision_from_string(train_hparams.precision)
            if X.dtype != precision:
                X = X.to(dtype=precision)

        # TODO: Temp fix
        if "mnist" in dataset_hparams.datagen:
            X = X.unsqueeze(1) # teacher generates data without MNIST b&w channel dimension of 1

        y = model(X)

    return X, y
