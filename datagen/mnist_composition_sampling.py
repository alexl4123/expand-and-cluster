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


class MNISTCompositionSampling:

    mnist_normalization = torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])
    glob_debug_show_images = False

    @classmethod
    def mnist_composition_sampling_7p5k_30d30_x2_y1(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [torchvision.transforms.RandomRotation((-30,30)), cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=2, images_on_y_axis=1, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 7500)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_composition_sampling_60k_0d360_x2_y1(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [torchvision.transforms.RandomRotation((0,360)), cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=2, images_on_y_axis=1, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 60000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X


    @classmethod
    def mnist_composition_sampling_0p1k_0d360_x3_y3(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [torchvision.transforms.RandomRotation((0,360)), cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 100)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_composition_sampling_3p5k_0d360_x3_y3(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [torchvision.transforms.RandomRotation((0,360)), cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 3500)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_composition_sampling_7p5k_0d360_x3_y3(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [torchvision.transforms.RandomRotation((0,360)), cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 7500)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_composition_sampling_15k_0d360_x3_y3(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [torchvision.transforms.RandomRotation((0,360)), cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 15000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X



    @classmethod
    def mnist_composition_sampling_30k_0d360_x3_y3(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [torchvision.transforms.RandomRotation((0,360)), cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 30000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X


    @classmethod
    def mnist_composition_sampling_60k_0d360_x3_y3(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [torchvision.transforms.RandomRotation((0,360)), cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 60000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_composition_sampling_120k_0d360_x3_y3(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [torchvision.transforms.RandomRotation((0,360)), cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 120000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_composition_sampling_180k_0d360_x3_y3(cls, augment=None, d_in=None):
        """
        One of the most often tested variants
        """
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [torchvision.transforms.RandomRotation((0,360)), cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 180000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_composition_sampling_240k_0d360_x3_y3(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [torchvision.transforms.RandomRotation((0,360)), cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 240000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_composition_sampling_360k_0d360_x3_y3(cls, augment=None, d_in=None):
        """
        One of the most tested variants
        """
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [torchvision.transforms.RandomRotation((0,360)), cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 360000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_composition_sampling_480k_0d360_x3_y3(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [torchvision.transforms.RandomRotation((0,360)), cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 480000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_composition_sampling_1k_0d0_x3_y3(cls, augment=None, d_in=None):
        debug_show_images = True

        data = DatagensGenerator.get_pure_mnist()

        transforms = [cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 3500)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X



    ##################################################################################################################################
    @classmethod
    def composition_sampling_random_noise_overlay_600k_600k_0_1_m1_0(cls, augment=None, d_in=None):

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=600000, data_size=600000, lower_uniform_perturbation=0, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_300k_300k_0_1_m1_0(cls, augment=None, d_in=None):

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=300000, data_size=300000, lower_uniform_perturbation=0, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_240k_240k_0_1_m1_0(cls, augment=None, d_in=None):

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=240000, data_size=240000, lower_uniform_perturbation=0, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_180k_180k_0_1_m1_0(cls, augment=None, d_in=None):

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=180000, data_size=180000, lower_uniform_perturbation=0, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0, debug_show_images=cls.glob_debug_show_images)

        return return_vector


    @classmethod
    def composition_sampling_random_noise_overlay_30k_30k_0_1_m1_0(cls, augment=None, d_in=None):
        """
        -> This is one of the best methods!
        """

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=30000, data_size=30000, lower_uniform_perturbation=0, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_60k_60k_0_1_m1_0(cls, augment=None, d_in=None):
        """
        -> This is one of the best methods!
        """

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=60000, data_size=60000, lower_uniform_perturbation=0, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_120k_120k_0_1_m1_0(cls, augment=None, d_in=None):

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=120000, data_size=120000, lower_uniform_perturbation=0, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_120k_120k_0_0p5_m0p5_0(cls, augment=None, d_in=None):

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=120000, data_size=120000, lower_uniform_perturbation=0, upper_uniform_perturbation=0.5, lower_uniform_perturbation_2=-0.5, upper_uniform_perturbation_2=0, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_120k_120k_0_2_m2_0(cls, augment=None, d_in=None):

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=120000, data_size=120000, lower_uniform_perturbation=0, upper_uniform_perturbation=2, lower_uniform_perturbation_2=-2, upper_uniform_perturbation_2=0, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_1k_1k_0_1_m1_0_reduced_mnist_1k(cls, augment=None, d_in=None):

        pure_mnist_reduced_data =DatagensGenerator.get_pure_mnist()

        pure_mnist_reduced_data.data = pure_mnist_reduced_data.data[:1000, :,: ]

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=1000, data_size=1000, lower_uniform_perturbation=0, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_10k_10k_0_1_m1_0_reduced_mnist_1k(cls, augment=None, d_in=None):

        pure_mnist_reduced_data =DatagensGenerator.get_pure_mnist()

        pure_mnist_reduced_data.data = pure_mnist_reduced_data.data[:1000, :,: ]

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=10000, data_size=10000, lower_uniform_perturbation=0, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_30k_30k_0_1_m1_0_reduced_mnist_1k(cls, augment=None, d_in=None):

        pure_mnist_reduced_data =DatagensGenerator.get_pure_mnist()

        pure_mnist_reduced_data.data = pure_mnist_reduced_data.data[:1000, :,: ]

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=30000, data_size=30000, lower_uniform_perturbation=0, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_60k_60k_0_1_m1_0_reduced_mnist_1k(cls, augment=None, d_in=None):

        pure_mnist_reduced_data =DatagensGenerator.get_pure_mnist()

        pure_mnist_reduced_data.data = pure_mnist_reduced_data.data[:1000, :,: ]

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=60000, data_size=60000, lower_uniform_perturbation=0, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_120k_120k_m1_1_m1_1(cls, augment=None, d_in=None):

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=120000, data_size=120000, lower_uniform_perturbation=-1, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=1, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_60k_60k_m1_1_m1_1(cls, augment=None, d_in=None):

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=60000, data_size=60000, lower_uniform_perturbation=-1, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=1, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_60k_60k_0_0p5_m0p5_0(cls, augment=None, d_in=None):

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=60000, data_size=60000, lower_uniform_perturbation=0, upper_uniform_perturbation=0.5, lower_uniform_perturbation_2=-0.5, upper_uniform_perturbation_2=0, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_60k_60k_0p25_0p75_m0p75_m0p25(cls, augment=None, d_in=None):

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=60000, data_size=60000, lower_uniform_perturbation=0.25, upper_uniform_perturbation=0.75, lower_uniform_perturbation_2=-0.75, upper_uniform_perturbation_2=-0.25, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_60k_240k_0_1__m1_0__1_2__m2_m1_add_add(cls, augment=None, d_in=None):

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay_two_mode(composition_data=60000, data_size=60000,
                                                                lower_uniform_perturbation=0, upper_uniform_perturbation=1,
                                                                lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0,
                                                                lower_uniform_perturbation_3=1, upper_uniform_perturbation_3=2,
                                                                lower_uniform_perturbation_4=-2, upper_uniform_perturbation_4=-1,
                                                                mode_1="+", mode_2="+",
                                                                debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_60k_120k_0_1__m1_0__1_2__m2_m1_add_add(cls, augment=None, d_in=None):

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay_two_mode(composition_data=60000, data_size=30000,
                                                                lower_uniform_perturbation=0, upper_uniform_perturbation=1,
                                                                lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0,
                                                                lower_uniform_perturbation_3=1, upper_uniform_perturbation_3=2,
                                                                lower_uniform_perturbation_4=-2, upper_uniform_perturbation_4=-1,
                                                                mode_1="+", mode_2="+",
                                                                debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_120k_240k_0_1__m1_0__1_2__m2_m1_add_add(cls, augment=None, d_in=None):

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay_two_mode(composition_data=120000, data_size=60000,
                                                                lower_uniform_perturbation=0, upper_uniform_perturbation=1,
                                                                lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0,
                                                                lower_uniform_perturbation_3=1, upper_uniform_perturbation_3=2,
                                                                lower_uniform_perturbation_4=-2, upper_uniform_perturbation_4=-1,
                                                                mode_1="+", mode_2="+",
                                                                debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_120k_240k_0_1__m1_0__1_2__m2_m1_add_mult(cls, augment=None, d_in=None):

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay_two_mode(composition_data=120000, data_size=60000,
                                                                lower_uniform_perturbation=0, upper_uniform_perturbation=1,
                                                                lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0,
                                                                lower_uniform_perturbation_3=1, upper_uniform_perturbation_3=2,
                                                                lower_uniform_perturbation_4=-2, upper_uniform_perturbation_4=-1,
                                                                mode_1="+", mode_2="*",
                                                                debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_10k_10k_0_1_m1_0_reduced_mnist_5k(cls, augment=None, d_in=None):

        pure_mnist_reduced_data =DatagensGenerator.get_pure_mnist()

        pure_mnist_reduced_data.data = pure_mnist_reduced_data.data[:10000, :,: ]

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=10000, data_size=10000, lower_uniform_perturbation=0, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0, debug_show_images=cls.glob_debug_show_images,
                                                                                data=pure_mnist_reduced_data)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_30k_30k_0_1_m1_0_reduced_mnist_5k(cls, augment=None, d_in=None):

        pure_mnist_reduced_data =DatagensGenerator.get_pure_mnist()

        pure_mnist_reduced_data.data = pure_mnist_reduced_data.data[:5000, :,: ]

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=30000, data_size=30000, lower_uniform_perturbation=0, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0, debug_show_images=cls.glob_debug_show_images,
                                                                                data=pure_mnist_reduced_data)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_120k_120k_0_1_m1_0_reduced_mnist_5k(cls, augment=None, d_in=None):

        pure_mnist_reduced_data =DatagensGenerator.get_pure_mnist()

        pure_mnist_reduced_data.data = pure_mnist_reduced_data.data[:5000, :,: ]

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=120000, data_size=120000, lower_uniform_perturbation=0, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0, debug_show_images=cls.glob_debug_show_images,
                                                                                data=pure_mnist_reduced_data)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_120k_120k_m1_1_m1_1_reduced_mnist_5k(cls, augment=None, d_in=None):

        pure_mnist_reduced_data =DatagensGenerator.get_pure_mnist()

        pure_mnist_reduced_data.data = pure_mnist_reduced_data.data[:5000, :,: ]

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=120000, data_size=120000, lower_uniform_perturbation=-1, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=1, debug_show_images=cls.glob_debug_show_images,
                                                                                data=pure_mnist_reduced_data)

        return return_vector

    @classmethod
    def mnist_composition_sampling_180k_0d360_x3_y3_reduced_mnist_5k(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()
        data.data = data.data[:5000, :,: ]

        transforms = [torchvision.transforms.RandomRotation((0,360)), cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 180000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_composition_sampling_180k_0d360_x2_y1(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [torchvision.transforms.RandomRotation((0,360)), cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=2, images_on_y_axis=1, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 180000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_composition_sampling_360k_0d360_x2_y1(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [torchvision.transforms.RandomRotation((0,360)), cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=2, images_on_y_axis=1, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 360000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_composition_sampling_180k_0d360_x4_y4(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [torchvision.transforms.RandomRotation((0,360)), cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=4, images_on_y_axis=4, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 180000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_composition_sampling_360k_0d360_x4_y4(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [torchvision.transforms.RandomRotation((0,360)), cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=4, images_on_y_axis=4, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 360000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_composition_sampling_180k_0d0_x3_y3(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 180000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_composition_sampling_360k_0d0_x3_y3(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 360000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def composition_sampling_random_noise_overlay_60k_60k_0_0p5_m0p5_0_test(cls, augment=None, d_in=None):

        return_vector = DatagensGenerator.composition_sampling_random_noise_overlay_test(composition_data=60000, data_size=60000, lower_uniform_perturbation=0, upper_uniform_perturbation=0.5, lower_uniform_perturbation_2=0, upper_uniform_perturbation_2=-0.5, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_60k_60k_0_2_m2_0(cls, augment=None, d_in=None):

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=60000, data_size=60000, lower_uniform_perturbation=0, upper_uniform_perturbation=2, lower_uniform_perturbation_2=-2, upper_uniform_perturbation_2=0, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_only_overlay_360k_m1_1(cls, augment=None, d_in=None):

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay_only_overlay(sample_base_size=60000, data_size=360000, lower_uniform_perturbation=-1, upper_uniform_perturbation=1, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def mnist_composition_sampling_360k_0d360_x3_y3_reduced_mnist_5k(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()
        data.data = data.data[:5000, :,: ]

        transforms = [torchvision.transforms.RandomRotation((0,360)), cls.mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 360000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def composition_sampling_random_noise_overlay_only_overlay_180k_0_1(cls, augment=None, d_in=None):

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay_only_overlay(sample_base_size=60000, data_size=180000, lower_uniform_perturbation=0, upper_uniform_perturbation=1, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_only_overlay_180k_m1_0(cls, augment=None, d_in=None):

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay_only_overlay(sample_base_size=60000, data_size=180000, lower_uniform_perturbation=-1, upper_uniform_perturbation=0, debug_show_images=cls.glob_debug_show_images)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_100_100_0_1_m1_0(cls, augment=None, d_in=None):

        debug_show_imgs = True

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=100, data_size=100, lower_uniform_perturbation=0, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0, debug_show_images=debug_show_imgs)

        if debug_show_imgs == True:
            quit(0)

        return return_vector

    @classmethod
    def composition_sampling_random_noise_overlay_5k_5k_0_1_m1_0_reduced_mnist_5k(cls, augment=None, d_in=None):
        return DatagensGenerator.composition_sampling_random_noise_overlay_5k_5k_0_1_m1_0_reduced_mnist_5k()

    @classmethod
    def mnist_composition_sampling_180k_0d0_x3_y3_reduced_mnist_5k(cls, augment=None, d_in=None):
        return DatagensGenerator.mnist_composition_sampling_180k_0d0_x3_y3_reduced_mnist_5k()

    @classmethod
    def composition_sampling_random_noise_overlay_60k_60k_0_1_m1_0_reduced_mnist_5k(cls, augment=None, d_in=None):
        return DatagensGenerator.composition_sampling_random_noise_overlay_60k_60k_0_1_m1_0_reduced_mnist_5k()

