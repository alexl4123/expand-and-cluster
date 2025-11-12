
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



class MNISTRandomNoiseOverlay(object):

    glob_debug_show_images = False

    @classmethod
    def mnist_random_noise_overlay_0_0p5_60k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=0, upper_uniform_perturbation=0.5, data_size=60000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m0p5_0_60k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-0.5, upper_uniform_perturbation=0, data_size=60000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_0p5_1_60k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=0.5, upper_uniform_perturbation=1, data_size=60000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m1_m0p5_60k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-1, upper_uniform_perturbation=-0.5, data_size=60000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_0_1_60k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=0, upper_uniform_perturbation=1, data_size=60000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m1_0_60k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-1, upper_uniform_perturbation=0, data_size=60000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_0_1_120k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=0, upper_uniform_perturbation=1, data_size=120000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m1_0_120k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-1, upper_uniform_perturbation=0, data_size=120000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_0_2_60k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=0, upper_uniform_perturbation=2, data_size=60000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m2_0_60k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-2, upper_uniform_perturbation=0, data_size=60000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m1_1_60k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-1, upper_uniform_perturbation=1, data_size=60000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m1_1_120k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-1, upper_uniform_perturbation=1, data_size=120000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m1_1_360k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-1, upper_uniform_perturbation=1, data_size=120000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_0_1_60k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=0, upper_uniform_perturbation=1, data_size=60000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m1_0_60k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-1, upper_uniform_perturbation=0, data_size=60000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m0p05_0p05_30k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-0.05, upper_uniform_perturbation=0.05, data_size=30000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m0p05_0p05_60k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-0.05, upper_uniform_perturbation=0.05, data_size=60000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m0p05_0p05_120k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-0.05, upper_uniform_perturbation=0.05, data_size=120000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m0p05_0p05_240k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-0.05, upper_uniform_perturbation=0.05, data_size=240000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m0p25_0p25_30k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-0.25, upper_uniform_perturbation=0.25, data_size=30000, debug_show_images=debug_show_images)


    @classmethod
    def mnist_random_noise_overlay_m0p25_0p25_60k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-0.25, upper_uniform_perturbation=0.25, data_size=60000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m0p25_0p25_120k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-0.25, upper_uniform_perturbation=0.25, data_size=120000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m0p25_0p25_240k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-0.25, upper_uniform_perturbation=0.25, data_size=240000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_5k_random_noise_overlay_m0p5_0_177p5k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_5k_random_noise_overlay(lower_uniform_perturbation=-0.5, upper_uniform_perturbation=0, data_size=177500, debug_show_images=debug_show_images)

    @classmethod
    def mnist_5k_random_noise_overlay_0_0p5_177p5k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_5k_random_noise_overlay(lower_uniform_perturbation=0, upper_uniform_perturbation=0.5, data_size=177500, debug_show_images=debug_show_images)

    @classmethod
    def mnist_5k_random_noise_overlay_m1_0_177p5k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_5k_random_noise_overlay(lower_uniform_perturbation=-1, upper_uniform_perturbation=0, data_size=177500, debug_show_images=debug_show_images)

    @classmethod
    def mnist_5k_random_noise_overlay_0_1_177p5k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_5k_random_noise_overlay(lower_uniform_perturbation=0, upper_uniform_perturbation=1, data_size=177500, debug_show_images=debug_show_images)

    @classmethod
    def mnist_5k_random_noise_overlay_m2_0_177p5k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_5k_random_noise_overlay(lower_uniform_perturbation=-2, upper_uniform_perturbation=0, data_size=177500, debug_show_images=debug_show_images)

    @classmethod
    def mnist_5k_random_noise_overlay_0_2_177p5k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_5k_random_noise_overlay(lower_uniform_perturbation=0, upper_uniform_perturbation=2, data_size=177500, debug_show_images=debug_show_images)

    @classmethod
    def mnist_5k_random_noise_overlay_m1_1_355k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_5k_random_noise_overlay(lower_uniform_perturbation=-1, upper_uniform_perturbation=1, data_size=360000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_5k_random_noise_overlay_m2_2_355k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_5k_random_noise_overlay(lower_uniform_perturbation=-2, upper_uniform_perturbation=2, data_size=360000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_0_1_150k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=0, upper_uniform_perturbation=1, data_size=150000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m1_0_150k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-1, upper_uniform_perturbation=0, data_size=150000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_0_2_150k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=0, upper_uniform_perturbation=2, data_size=150000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m2_0_150k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-2, upper_uniform_perturbation=0, data_size=150000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_0_0p5_150k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=0, upper_uniform_perturbation=0.5, data_size=150000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m0p5_0_150k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-0.5, upper_uniform_perturbation=0, data_size=150000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m1_1_300k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-1, upper_uniform_perturbation=1, data_size=300000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_0_1_180k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=0, upper_uniform_perturbation=1, data_size=180000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m1_0_180k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-1, upper_uniform_perturbation=0, data_size=180000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_5k_random_noise_overlay_m1_0_87p5k(cls, augment=None, d_in = None):

        return DatagensGenerator.mnist_5k_random_noise_overlay_m1_0_87p5k()

    @classmethod
    def mnist_5k_random_noise_overlay_0_1_87p5k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_5k_random_noise_overlay_0_1_87p5k()

    @classmethod
    def mnist_random_noise_overlay_0_1_100(cls, augment=None, d_in = None):
        debug_show_images = True

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=0, upper_uniform_perturbation=1, data_size=100, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_0p5_offsets_60k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay_test(lower_uniform_perturbation=0, upper_uniform_perturbation=0.5, data_size=60000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m0p5_offsets_60k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay_test(lower_uniform_perturbation=0, upper_uniform_perturbation=-0.5, data_size=60000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_horizontal_flip_reduced_mnist_5k(cls, augment=None, d_in=None):
        return DatagensGenerator.mnist_horizontal_flip_reduced_mnist_5k()

    @classmethod
    def mnist_random_noise_overlay_0_10_60k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=0, upper_uniform_perturbation=10, data_size=60000, debug_show_images=debug_show_images)

    @classmethod
    def mnist_random_noise_overlay_m10_0_60k(cls, augment=None, d_in = None):
        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist_random_noise_overlay(lower_uniform_perturbation=-10, upper_uniform_perturbation=0, data_size=60000, debug_show_images=debug_show_images)



