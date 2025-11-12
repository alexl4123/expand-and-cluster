
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


class MNIST:

    mnist_normalized_min_value = -0.42 # If Data available: torch.min(...)
    mnist_normalization = torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])
    glob_debug_show_images = False

    @classmethod
    def mnist(cls, augment=None, d_in=None):

        debug_show_images = cls.glob_debug_show_images

        return DatagensGenerator.mnist()

    @classmethod
    def mnist_test(cls, augment=None, d_in=None):
        return DatagensGenerator.mnist(train=False)

    @classmethod
    def mnist_reduced_1k(cls, augment=None, d_in=None):
        data = DatagensGenerator.mnist_reduced_1k()
        return data

    @classmethod
    def mnist_reduced_5k(cls, augment=None, d_in=None):
        data = DatagensGenerator.mnist_reduced_5k()
        return data

    @classmethod
    def mnist_reduced_0p5k(cls, augment=None, d_in=None):
        data = DatagensGenerator.mnist()
        return data[:500,:,:]

    @classmethod
    def mnist_reduced_0p1k(cls, augment=None, d_in=None):
        data = DatagensGenerator.mnist()
        return data[:100,:,:]

    @classmethod
    def mnist_reduced_10k(cls, augment=None, d_in=None):
        data = DatagensGenerator.mnist()
        return data[:10000,:,:]

    @classmethod
    def mnist_reduced_30k(cls, augment=None, d_in=None):
        data = DatagensGenerator.mnist()
        return data[:30000,:,:]

    @classmethod
    def mnist_conv(cls, augment=None, d_in=None):
        X = mnist(augment, d_in)
        return X.unsqueeze(1)

    @classmethod
    def mnist_vertical_flip(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images
        
        train_set = DatagensGenerator.get_pure_mnist()
        additional_transforms = [cls.mnist_normalization]

        X = DatagensGenerator._vertical_flip(train_set, debug_show_images = debug_show_images, additional_transforms=additional_transforms)

        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_horizontal_flip(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images
        
        train_set = DatagensGenerator.get_pure_mnist()
        additional_transforms = [cls.mnist_normalization]

        X = DatagensGenerator._horizontal_flip(train_set, debug_show_images = debug_show_images, additional_transforms=additional_transforms)

        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def _invert(cls, train_data, debug_show_images = False, additional_transforms = []):
        transforms = [torchvision.transforms.functional.invert] + additional_transforms
        
        X = DatagensGenerator._apply_standard_augmentations(train_data, transforms, debug_show_images=debug_show_images)

        return X


    @classmethod
    def mnist_invert(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        train_data = DatagensGenerator.get_pure_mnist()
        additional_transforms = [cls.mnist_normalization]

        X = cls._invert(train_data, debug_show_images=debug_show_images, additional_transforms=additional_transforms)

        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_random_rotation(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        train_data = DatagensGenerator.get_pure_mnist()

        additional_transforms = [cls.mnist_normalization]

        X = DatagensGenerator._random_rotation(train_data, debug_show_images=debug_show_images, additional_transforms=additional_transforms)


        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    class DiscreteRandomDraw:

        def __init__(self, elements, probabilities):
            self._elements = elements
            self._probabilities = probabilities

        def draw(self, number_of_draws = 1):
            return np.random.choice(self._elements, number_of_draws, self._probabilities)

    @classmethod
    def _get_mnist_discrete_probability_distribution(cls, data = None):

        distribution_file_name = "probability_distribution_mnist.pickle"
        if Path(distribution_file_name).is_file():
            with open(distribution_file_name, 'rb') as inp:
                distribution = pickle.load(inp)

        else:
            t_data = []

            toTensorTransform = torchvision.transforms.ToTensor()
            for im in data.data.numpy():
                t_data.append(toTensorTransform(im))

            data_shape = data.data.shape
            dataset_size = data_shape[0]
            datapoint_shape = data_shape[1:]

            samples = {}

            for y in range(datapoint_shape[0]):
                for x in range(datapoint_shape[1]):
                    samples[(x,y)] = {}

            index = 0
            for im in t_data:
                if index % 100 == 0:
                    print(index)
                index += 1
                for y in range(datapoint_shape[0]):
                    for x in range(datapoint_shape[1]):
                        value = float(im[0][y][x])

                        if value not in samples[(x,y)]:
                            samples[(x,y)][value] = 1
                        else:
                            samples[(x,y)][value] += 1

            distribution = {}
            for y in range(datapoint_shape[0]):
                for x in range(datapoint_shape[1]):
                    my_sample = samples[(x,y)]

                    elements = []
                    probabilities = []

                    for key in my_sample.keys():
                        elements.append(key)
                        probabilities.append(my_sample[key] / dataset_size)

                    distribution[(x,y)] = DiscreteRandomDraw(elements, probabilities)

            with open(distribution_file_name, 'wb') as handle:
                pickle.dump(distribution, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return distribution

    @classmethod
    def mnist_distribution_sampling(cls, augment=None, d_in=None, number_of_samples = 60000, output="standard"):
        """
        Generating a MNIST distribution and sampling according to this distribution.
        """
        #debug_show_images = False
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()

        transforms = [torchvision.transforms.ToPILImage(), torchvision.transforms.ToTensor(), cls.mnist_normalization]

        data_shape = data.data.shape
        datapoint_shape = data_shape[1:]

        distribution = _get_mnist_discrete_probability_distribution(data = data)

        #X = _generate_normalized_samples_from_distribution(distribution, data_shape, transforms, number_of_samples=60000, debug_show_images = debug_show_images)
        X = []
        #number_of_samples = 60000


        for index in range(number_of_samples):
            X.append([])
            for y in range(datapoint_shape[0]):
                X[index].append([])


        for y in range(datapoint_shape[0]):
            for x in range(datapoint_shape[1]):
                samples = distribution[(x,y)].draw(number_of_samples)

                for index in range(len(samples)):
                    X[index][y].append(samples[index])

        new_X = []
        new_index = 0

        for im in X:
            if new_index % 100 == 0:
                print(new_index)

            img = torch.Tensor(im)

            for t in transforms:
                img = t(img)

            new_X.append(img)

            new_index += 1

        X = torch.concat(new_X)
        current_mean = float(torch.mean(X))

        new_X_2 = []
        for im in new_X:

            im = im.apply_(lambda x: (x if x - current_mean >= cls.mnist_normalized_min_value else cls.mnist_normalized_min_value))
            #im = normalization(im)
            new_X_2.append(im)

            if debug_show_images is True:
                plt.pyplot.imshow(im[0,:,:], cmap='gray')
                plt.pyplot.show()

        if output == "standard":
            X = torch.concat(new_X_2)
        else:
            X = new_X_2

        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    ####################################################################################################################

    @classmethod
    def mnist_extreme_value_sampling_240k_lm50_u50(cls, augment=None, d_in=None):

        return cls._mnist_extreme_value_sampling(number_samples=240000, lower_uniform_perturbation=-50, upper_uniform_perturbation=50)

    @classmethod
    def mnist_extreme_value_sampling_120k_lm50_u50(cls, augment=None, d_in=None):

        return _mnist_extreme_value_sampling(number_samples=120000, lower_uniform_perturbation=-50, upper_uniform_perturbation=50)

    @classmethod
    def mnist_extreme_value_sampling_60k_lm50_u50(cls, augment=None, d_in=None):

        return _mnist_extreme_value_sampling(number_samples=60000, lower_uniform_perturbation=-50, upper_uniform_perturbation=50)

    @classmethod
    def mnist_extreme_value_sampling_gaussian_60k_u0_sd16p5(cls, augment=None, d_in=None):

        return _mnist_extreme_value_sampling(number_samples=60000, gaussian=True, mu=0, sd=16.5)

    @classmethod
    def mnist_extreme_value_sampling_gaussian_240k_u0_sd16p5(cls, augment=None, d_in=None):

        return _mnist_extreme_value_sampling(number_samples=240000, gaussian=True, mu=0, sd=16.5)

    @classmethod
    def _mnist_extreme_value_sampling(cls, augment=None, d_in=None, number_samples = 60000, lower_uniform_perturbation=1, upper_uniform_perturbation = 1, debug_show_images = glob_debug_show_images,
                                    gaussian = False, mu=0, sd = 1):
        debug_show_images = cls.glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()
        np_data = data.data.numpy()

        transforms = [torchvision.transforms.ToTensor(), cls.mnist_normalization]
        X = []

        if debug_show_images is True:
            number_of_examples = 3
            count = 0

        for _ in range(number_samples):

            sample_image_index = random.randrange(0,len(data))

            image = np.copy(np_data[sample_image_index])

            for t in transforms:
                image = t(image)

            if gaussian is False:
                overlay = np.random.uniform(low=lower_uniform_perturbation, high=upper_uniform_perturbation)
            else:
                overlay = np.random.normal(loc=mu, scale=sd)

            image[0,:,:] = image[0,:,:] * overlay

            X.append(image)

            if debug_show_images is True:
                plt.pyplot.imshow(image[0,:,:], cmap='gray')
                plt.pyplot.show()

                if count < number_of_examples:
                    count += 1
                else:
                    break

        X = torch.concat(X)

        return X

    @classmethod
    def mnist_5k_invert(cls, augment=None, d_in=None):
        debug_show_images = cls.glob_debug_show_images

        train_data = DatagensGenerator.get_pure_mnist()
        train_data.data = train_data.data[:5000, :,: ]
    
        additional_transforms = [cls.mnist_normalization]

        X = _invert(train_data, debug_show_images=debug_show_images, additional_transforms=additional_transforms)

        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X
