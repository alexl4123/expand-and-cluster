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


class MNISTRandomNoise:

    glob_debug_show_images = False

    @classmethod
    def mnist_random_noise_60k_min_0_max_1_b_0_1(cls, augment=None, d_in=None):

        return DatagensGenerator.random_noise_single_multiplication(sample_size=60000,min=0, max=1, smin=0, smax=1)

    @classmethod
    def mnist_random_noise_60k_min_0_max_1(cls, augment=None, d_in=None):

        return DatagensGenerator.random_noise_single_multiplication(sample_size=60000,min=0, max=1)

    @classmethod
    def mnist_random_noise_60k_min_m1_max_0(cls, augment=None, d_in=None):

        return DatagensGenerator.random_noise_single_multiplication(sample_size=60000,min=-1, max=0)

    @classmethod
    def mnist_random_noise_60k_min_0p5_max_1p5(cls, augment=None, d_in=None):

        return DatagensGenerator.random_noise_single_multiplication(sample_size=60000,min=0.5, max=1.5)

    @classmethod
    def mnist_random_noise_60k_min_m1p5_max_m0p5(cls, augment=None, d_in=None):

        return DatagensGenerator.random_noise_single_multiplication(sample_size=60000,min=-1.5, max=0.5)

    @classmethod
    def mnist_random_noise_120k_min_0p5_max_1p5(cls, augment=None, d_in=None):

        return DatagensGenerator.random_noise_single_multiplication(sample_size=120000,min=0.5, max=1.5)

    @classmethod
    def mnist_random_noise_120k_min_m1p5_max_m0p5(cls, augment=None, d_in=None):

        return DatagensGenerator.random_noise_single_multiplication(sample_size=120000,min=-1.5, max=0.5)

    @classmethod
    def mnist_random_noise_60k_min_m50_max_50_smin_m100_smax_100(cls, augment=None, d_in=None):

        return DatagensGenerator.random_noise_single_multiplication(sample_size=60000,min=-50, max=50, smin=-100, smax=100)

    @classmethod
    def mnist_random_noise_240k_min_m50_max_50_smin_m100_smax_100(cls, augment=None, d_in=None):

        return DatagensGenerator.random_noise_single_multiplication(sample_size=240000,min=-50, max=50, smin=-100, smax=100)

    @classmethod
    def mnist_random_noise_130k_min_0_max_1(cls, augment=None, d_in=None):

        return DatagensGenerator.random_noise_single_multiplication(sample_size=130000,min=0, max=1)

    @classmethod
    def mnist_random_noise_130k_min_m1_max_0(cls, augment=None, d_in=None):

        return DatagensGenerator.random_noise_single_multiplication(sample_size=130000,min=-1, max=0)

    @classmethod
    def mnist_random_noise_360k_min_m1_max_1(cls, augment=None, d_in=None):

        return DatagensGenerator.random_noise_single_multiplication(sample_size=360000,min=-1, max=1)

    @classmethod
    def mnist_random_noise_180k_min_0_max_1(cls, augment=None, d_in=None):

        return DatagensGenerator.random_noise_single_multiplication(sample_size=180000,min=0, max=1)

    @classmethod
    def mnist_random_noise_180k_min_m1_max_0(cls, augment=None, d_in=None):

        return DatagensGenerator.random_noise_single_multiplication(sample_size=180000,min=-1, max=0)

