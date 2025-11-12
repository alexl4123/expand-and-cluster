# TODO -> Refactor for all generators.

import os
import torchvision
import numpy as np
import multiprocessing
import torch
import random

from functools import partial
from PIL import Image
from platforms.platform import get_platform
from matplotlib import pyplot as plt

mnist_normalized_min_value = -0.42 # If Data available: torch.min(...)
mnist_normalized_max_value = 2.821 # If Data available: torch.max(...)
mnist_normalization = torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])
glob_debug_show_images = False

class DatagensGenerator:


    @classmethod
    def get_pure_mnist(cls, train=True):
        train_set = torchvision.datasets.MNIST(
            train=train, root=os.path.join(get_platform().dataset_root, 'mnist'), download=True)

        return train_set


    @classmethod
    def mnist(cls, train=True):
        debug_show_images = False
        number_of_examples = 10

        train_set = cls.get_pure_mnist(train=train)

        transforms = [torchvision.transforms.ToTensor(),
                    mnist_normalization]
        
        count = 0
        X = []
        for im in train_set.data.numpy():
            im = Image.fromarray(im, mode='L')
            for t in transforms:
                im = t(im)
            X.append(im)

            if debug_show_images is True:
                plt.imshow(im[0,:,:], cmap='gray')
                plt.savefig(f"imgs_tmp/mnist_{count}.png")

                if count < number_of_examples:
                    count += 1
                else:
                    quit(0)



        X = torch.concat(X)
        return X
    
    @classmethod
    def mnist_reduced_1k(cls):
        data = cls.mnist()

        return data[:1000,:,:]
     
    @classmethod
    def mnist_reduced_5k(cls):
        data = cls.mnist()

        return data[:5000,:,:]

    @classmethod
    def mnist_reduced_15k(cls):
        data = cls.mnist()

        return data[:15000,:,:]
    
    @classmethod
    def mnist_image_composition(cls, augment=None, d_in=None, number_of_samples = None):
        debug_show_images = glob_debug_show_images

        data = cls.get_pure_mnist()

        transforms = [mnist_normalization]

        X = cls._image_composition(data, transforms,
                                images_on_x_axis=2, images_on_y_axis=1, 
                                debug_show_images=debug_show_images,
                                number_of_samples=number_of_samples)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X



    @classmethod
    def _image_composition(cls, data, transforms, images_on_x_axis = 1, images_on_y_axis = 1,  debug_show_images = False, number_of_samples = None):
        """
        images_on_x_axis -> Defines how many images on the x-axis should be put
        images_on_y_axis -> Defines how many images on the y-axis should be put

        Dynamically creates an augmented image that combines randomly sampled other images.

        """
        to_tensor_transform = torchvision.transforms.ToTensor()
        X = []

        if debug_show_images is True:
            number_of_examples = 100


        np_data = data.data.numpy()

        if number_of_samples == None:
            number_of_samples = len(data)


        x_size = (np_data[0].shape)[1]
        y_size = (np_data[0].shape)[0]

        y_sizes = cls.calculate_y_sizes_on_axis(images_on_y_axis, images_on_x_axis, y_size)
        x_sizes = cls.calculate_x_sizes_on_axis(images_on_x_axis, images_on_y_axis, x_size)

        for index in range(number_of_samples):

            if debug_show_images is True:
                base_imgs = []

            new_img = None
            for y_index in range(images_on_y_axis):
                new_x_img = None
                for x_index in range(images_on_x_axis):
                    img_index = random.randrange(0,len(data))
                    img = np_data[img_index]
                    img = Image.fromarray(img, mode='L')
                    img = to_tensor_transform(img)

                    if debug_show_images is True:
                        base_imgs.append(img)

                    img_x_sizes = x_sizes[y_index][x_index]
                    img_y_sizes = y_sizes[y_index][x_index]
                    
                    tmp_img = img[:, img_y_sizes[0]: img_y_sizes[0] + len(img_y_sizes),img_x_sizes[0]:img_x_sizes[0] + len(img_x_sizes)]
                    if new_x_img is None:
                        new_x_img = tmp_img
                    else:
                        new_x_img = torch.cat((new_x_img, tmp_img), 2)

                if new_img is None:
                    new_img = new_x_img
                else:
                    new_img = torch.cat((new_img, new_x_img), 1)

            for t in transforms:
                new_img = t(new_img)

            X.append(new_img)

            if debug_show_images == True:
                plt.imshow(new_img[0,:,:], cmap='gray')
                plt.savefig(f"imgs_tmp/composition_sampling_{index}.png")
                #plt.show()

                """
                for img in base_imgs:
                    plt.imshow(img[0,:,:], cmap='gray')
                    plt.show()

                plt.imshow(new_img[0,:,:], cmap='gray')
                plt.show()
                """


                if index >= number_of_examples:
                    break

        X = torch.concat(X)

        return X


    @classmethod
    def calculate_y_sizes_on_axis(cls, images_on_y_axis, images_on_x_axis, y_size):
        """
        y-major order, starting left-top
        i.e., y_sizes[0][1] -> row/y=0, column/x=1
        """

        if images_on_y_axis == 1: 
            y_sizes = [[list(range(y_size)) for _ in range(images_on_x_axis)]]
        else:
            tmp_y_size = int(y_size / images_on_y_axis)
            missing_size = y_size - (tmp_y_size * images_on_y_axis)
            y_sizes = []

            single_y_vector = []
            value = 0
            for _ in range(1,images_on_y_axis):
                single_y_vector.append(list(range(value, value + tmp_y_size)))
                value += tmp_y_size
            single_y_vector += [list(range(value, value + tmp_y_size + missing_size))]

            for index in range(images_on_y_axis):
                y_sizes.append([single_y_vector[index] for _ in range(images_on_x_axis)])

        return y_sizes

    @classmethod
    def calculate_x_sizes_on_axis(cls, images_on_x_axis, images_on_y_axis, x_size):
        """
        y-major order, starting left-top
        i.e., y_sizes[0][1] -> row/y=0, column/x=1
        """

        if images_on_x_axis == 1: 
            x_sizes = [list(range(0,x_size)) for _ in range(images_on_y_axis)]
        else:

            tmp_x_size = int(x_size / images_on_x_axis)
            missing_size = x_size - (tmp_x_size * images_on_x_axis)

            single_x_vector = []
            value = 0
            for _ in range(1,images_on_x_axis):
                single_x_vector.append(list(range(value, value + tmp_x_size)))
                value += tmp_x_size

            single_x_vector += [list(range(value, value + tmp_x_size + missing_size))]

            x_sizes = [single_x_vector for _ in range(images_on_y_axis)]

        return x_sizes


    @classmethod
    def random_noise_single_multiplication(cls, augment=None, d_in=None, sample_size=60000, min = 0, max = 1, smin=1, smax=1, mode="*"):
        debug_show_images = glob_debug_show_images

        data = cls.get_pure_mnist()

        transforms = [torchvision.transforms.ToPILImage(), torchvision.transforms.ToTensor(), mnist_normalization]

        data_shape = data.data.shape
        size = data_shape[0]
        datapoint_shape = data_shape[1:]
        
        number_noise_samples = sample_size
        X = []

        for index in range(number_noise_samples):

            im = np.random.uniform(low=min, high=max, size = tuple(datapoint_shape))
            scalar = np.random.uniform(low=smin, high=smax, size = 1)

            im = torch.Tensor(im)

            for t in transforms:
                im = t(im)

            if mode == "*":
                def multiplication(x1, x2):
                    return x1 * x2
                operation = multiplication
            else: #Mode (should be) is "+"
                def addition(x1, x2):
                    return x1 + x2
                operation = addition

            im[0,:,:] = operation(im[0,:,:], scalar)

            X.append(im)

            if debug_show_images is True:
                plt.pyplot.imshow(im[0,:,:], cmap='gray')
                plt.pyplot.show()


                if index > 5:
                    break

        X = torch.concat(X)

        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X


    @classmethod
    def mnist_random_noise_single_multiplication_overlay(cls, augment=None, d_in=None, sample_size=60000, smin=1, smax=1, np_data = None, transforms = None):
        debug_show_images = glob_debug_show_images

        if transforms is None:
            transforms = [torchvision.transforms.ToPILImage(), torchvision.transforms.ToTensor(), mnist_normalization]


        if np_data is None:
            data = cls.get_pure_mnist()
            np_data = data.data.numpy()
        
        X = []

        for index in range(sample_size):
            sample_image_index = random.randrange(0,len(np_data))

            image = np.copy(np_data[sample_image_index])

            scalar = np.random.uniform(low=smin, high=smax, size = 1)

            image = torch.Tensor(image)

            for t in transforms:
                image = t(image)

            if len(image.shape) == 3:
                image[0,:,:] = image[0,:,:] * scalar
            elif len(image.shape) == 2:
                image = image * scalar

            X.append(image)

            if debug_show_images is True:
                plt.pyplot.imshow(image[0,:,:], cmap='gray')
                plt.pyplot.show()


                if index > 5:
                    break

        X = torch.concat(X)

        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X
    
    @classmethod
    def mnist_random_noise_overlay(cls, lower_uniform_perturbation=-0.25,upper_uniform_perturbation=0.25,data_size=60000, debug_show_images = glob_debug_show_images):

        data = DatagensGenerator.get_pure_mnist()

        np_data = data.data.numpy()

        transforms = [torchvision.transforms.ToTensor(),
                    mnist_normalization]

        return cls._random_noise_overlay(np_data, transforms, lower_uniform_perturbation=lower_uniform_perturbation, upper_uniform_perturbation=upper_uniform_perturbation, data_size=data_size, debug_show_images=debug_show_images)
       
    @classmethod
    def composition_sampling_random_noise_overlay_two_mode(cls, composition_data = 60000, data_size=60000,
                                                        lower_uniform_perturbation=0, upper_uniform_perturbation=1,
                                                        lower_uniform_perturbation_2=-1,upper_uniform_perturbation_2=0,
                                                        lower_uniform_perturbation_3=1, upper_uniform_perturbation_3=2,
                                                        lower_uniform_perturbation_4=-2, upper_uniform_perturbation_4=-1,
                                                        mode_1 = "+",
                                                        mode_2 = "+",
                                                        debug_show_images = glob_debug_show_images):

        data = DatagensGenerator.get_pure_mnist()

        transforms = [torchvision.transforms.RandomRotation((0,0)), mnist_normalization]

        X = []

        img_comp_data = cls._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = composition_data)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        transforms = [torchvision.transforms.ToTensor()]

        rand_noise_data_1 = cls._random_noise_overlay(img_comp_data.data.numpy(), transforms, lower_uniform_perturbation=lower_uniform_perturbation, upper_uniform_perturbation=upper_uniform_perturbation, data_size=data_size, debug_show_images=debug_show_images, mode=mode_1)
        rand_noise_data_2 = cls._random_noise_overlay(img_comp_data.data.numpy(), transforms, lower_uniform_perturbation=lower_uniform_perturbation_2, upper_uniform_perturbation=upper_uniform_perturbation_2, data_size=data_size, debug_show_images=debug_show_images, mode=mode_1)


        rand_noise_data_3 = cls._random_noise_overlay(img_comp_data.data.numpy(), transforms, lower_uniform_perturbation=lower_uniform_perturbation_3, upper_uniform_perturbation=upper_uniform_perturbation_3, data_size=data_size, debug_show_images=debug_show_images, mode=mode_2)
        rand_noise_data_4 = cls._random_noise_overlay(img_comp_data.data.numpy(), transforms, lower_uniform_perturbation=lower_uniform_perturbation_4, upper_uniform_perturbation=upper_uniform_perturbation_4, data_size=data_size, debug_show_images=debug_show_images, mode=mode_2)

        X = [img_comp_data, rand_noise_data_1, rand_noise_data_2, rand_noise_data_3, rand_noise_data_4]
        X = torch.concat(X, dim=0)

        return X

    @classmethod
    def composition_sampling_random_noise_overlay(cls, composition_data = 60000, data_size=60000,lower_uniform_perturbation=-0.25, upper_uniform_perturbation=0.25, lower_uniform_perturbation_2=-0.25,upper_uniform_perturbation_2=0.25, debug_show_images = glob_debug_show_images, mode = "+", data = None):

        if data is None:
            data = cls.get_pure_mnist()

        transforms = [mnist_normalization]

        X = []

        img_comp_data = cls._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = composition_data)
        
        transforms = [torchvision.transforms.ToTensor()]

        rand_noise_data_1 = cls._random_noise_overlay(img_comp_data.data.numpy(), transforms, lower_uniform_perturbation=lower_uniform_perturbation, upper_uniform_perturbation=upper_uniform_perturbation, data_size=data_size, debug_show_images=debug_show_images, mode=mode)
        rand_noise_data_2 = cls._random_noise_overlay(img_comp_data.data.numpy(), transforms, lower_uniform_perturbation=lower_uniform_perturbation_2, upper_uniform_perturbation=upper_uniform_perturbation_2, data_size=data_size, debug_show_images=debug_show_images, mode=mode)

        X = [img_comp_data, rand_noise_data_1, rand_noise_data_2]
        X = torch.concat(X, dim=0)
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def _random_noise_overlay(cls, np_data, transforms, augment=None, d_in=None, lower_uniform_perturbation=-0.25,upper_uniform_perturbation=0.25,data_size=60000, debug_show_images = glob_debug_show_images, mode="+"):

        X = []

        if debug_show_images is True:
            number_of_examples = 100


        if data_size == None:
            data_size = len(np_data)

        for index in range(data_size):
            sample_image_index = random.randrange(0,len(np_data))

            image = np.copy(np_data[sample_image_index])
            # TODO COMMENT FOR PRODUCTION!!!
            #image = np.copy(np_data[index])

            for t in transforms:
                image = t(image)

            overlay = np.random.uniform(low=lower_uniform_perturbation, high=upper_uniform_perturbation, size = (28,28))

            if mode == "*":
                def multiplication(x1, x2):
                    return x1 * x2
                operation = multiplication
            else: #Mode (should be) is "+"
                def addition(x1, x2):
                    return x1 + x2
                operation = addition

            if len(image.shape) == 3:
                image[0,:,:] = operation(image[0,:,:], overlay)
            elif len(image.shape) == 2:
                image = operation(image, overlay)

            X.append(image)

            if debug_show_images is True:

                #plt.pyplot.imshow(np_data[sample_image_index], cmap='gray')
                #plt.pyplot.show()

                plt.imshow(image[0,:,:], cmap='gray')
                plt.savefig(f"imgs_tmp/overlay_{index}_{lower_uniform_perturbation}_{upper_uniform_perturbation}.png")
                #plt.pyplot.show()

                if index >= number_of_examples:
                    break

        X = torch.concat(X)

        return X


    @classmethod
    def mnist_7p5k_random_noise_overlay(cls, lower_uniform_perturbation=-0.25,upper_uniform_perturbation=0.25,data_size=60000, debug_show_images = glob_debug_show_images):

        data = DatagensGenerator.get_pure_mnist()
        data.data = data.data[:7500, :,: ]

        np_data = data.data.numpy()

        transforms = [torchvision.transforms.ToTensor(),
                    mnist_normalization]

        return cls._random_noise_overlay(np_data, transforms, lower_uniform_perturbation=lower_uniform_perturbation, upper_uniform_perturbation=upper_uniform_perturbation, data_size=data_size, debug_show_images=debug_show_images)


    @classmethod
    def mnist_5k_random_noise_overlay(cls, lower_uniform_perturbation=-0.25,upper_uniform_perturbation=0.25,data_size=60000, debug_show_images = glob_debug_show_images):

        data = DatagensGenerator.get_pure_mnist()
        data.data = data.data[:5000, :,: ]

        np_data = data.data.numpy()

        transforms = [torchvision.transforms.ToTensor(),
                    mnist_normalization]

        return cls._random_noise_overlay(np_data, transforms, lower_uniform_perturbation=lower_uniform_perturbation, upper_uniform_perturbation=upper_uniform_perturbation, data_size=data_size, debug_show_images=debug_show_images)

    @classmethod
    def mnist_15k_random_noise_overlay(cls, lower_uniform_perturbation=-0.25,upper_uniform_perturbation=0.25,data_size=60000, debug_show_images = glob_debug_show_images):

        data = DatagensGenerator.get_pure_mnist()
        data.data = data.data[:15000, :,: ]

        np_data = data.data.numpy()

        transforms = [torchvision.transforms.ToTensor(),
                    mnist_normalization]

        return cls._random_noise_overlay(np_data, transforms, lower_uniform_perturbation=lower_uniform_perturbation, upper_uniform_perturbation=upper_uniform_perturbation, data_size=data_size, debug_show_images=debug_show_images)


    @classmethod
    def composition_sampling_random_noise_overlay_only_overlay(cls, sample_base_size = 60000, data_size=60000,lower_uniform_perturbation=-0.25, upper_uniform_perturbation=0.25, debug_show_images = glob_debug_show_images, mode = "+", data = None):

        if data is None:
            data = cls.get_pure_mnist()

        transforms = [torchvision.transforms.RandomRotation((0,0)), mnist_normalization]

        X = []

        img_comp_data = cls._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=False,
                                number_of_samples = sample_base_size)
        
        transforms = [torchvision.transforms.ToTensor()]

        rand_noise_data_1 = cls._random_noise_overlay(img_comp_data.data.numpy(), transforms, lower_uniform_perturbation=lower_uniform_perturbation, upper_uniform_perturbation=upper_uniform_perturbation, data_size=data_size, debug_show_images=debug_show_images, mode=mode)

        # Difference is here compared to standard comp.-sampling
        X = [rand_noise_data_1]
        X = torch.concat(X, dim=0)
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()


        return X

    @classmethod
    def mnist_random_noise_overlay_test(cls, lower_uniform_perturbation=-0.25,upper_uniform_perturbation=0.25,data_size=60000, debug_show_images = glob_debug_show_images):

        data = DatagensGenerator.get_pure_mnist()

        np_data = data.data.numpy()

        transforms = [torchvision.transforms.ToTensor(),
                    mnist_normalization]

        return cls._random_noise_overlay_test(np_data, transforms, lower_uniform_perturbation=lower_uniform_perturbation, upper_uniform_perturbation=upper_uniform_perturbation, data_size=data_size, debug_show_images=debug_show_images)

    @classmethod
    def _random_noise_overlay_test(cls, np_data, transforms, augment=None, d_in=None, lower_uniform_perturbation=-0.25,upper_uniform_perturbation=0.25,data_size=60000, debug_show_images = glob_debug_show_images, mode="+"):

        X = []

        if debug_show_images is True:
            number_of_examples = 100


        if data_size == None:
            data_size = len(np_data)

        for index in range(data_size):
            sample_image_index = random.randrange(0,len(np_data))

            image = np.copy(np_data[sample_image_index])
            #image = np.copy(np_data[index])

            for t in transforms:
                image = t(image)

            #overlay = np.random.uniform(low=lower_uniform_perturbation, high=upper_uniform_perturbation, size = (28,28))

            if mode == "*":
                def multiplication(x1, x2):
                    return x1 * x2
                operation = multiplication
            else: #Mode (should be) is "+"
                def addition(x1, x2):
                    return x1 + x2
                operation = addition

            if len(image.shape) == 3:
                image[0,:,:] = operation(image[0,:,:], upper_uniform_perturbation)
            elif len(image.shape) == 2:
                image = operation(image, 0.5)

            X.append(image)

            if debug_show_images is True:

                #plt.pyplot.imshow(np_data[sample_image_index], cmap='gray')
                #plt.pyplot.show()

                plt.imshow(image[0,:,:], cmap='gray')
                plt.savefig(f"imgs_tmp/overlay_{index}_{lower_uniform_perturbation}_{upper_uniform_perturbation}.png")
                #plt.pyplot.show()

                if index >= number_of_examples:
                    break

        X = torch.concat(X)

        return X

    @classmethod
    def composition_sampling_random_noise_overlay_test(cls, composition_data = 60000, data_size=60000,lower_uniform_perturbation=-0.25, upper_uniform_perturbation=0.25, lower_uniform_perturbation_2=-0.25,upper_uniform_perturbation_2=0.25, debug_show_images = glob_debug_show_images, mode = "+", data = None):

        if data is None:
            data = cls.get_pure_mnist()

        transforms = [mnist_normalization]

        X = []

        img_comp_data = cls._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = composition_data)
        
        transforms = [torchvision.transforms.ToTensor()]

        rand_noise_data_1 = cls._random_noise_overlay_test(img_comp_data.data.numpy(), transforms, lower_uniform_perturbation=lower_uniform_perturbation, upper_uniform_perturbation=upper_uniform_perturbation, data_size=data_size, debug_show_images=debug_show_images, mode=mode)
        rand_noise_data_2 = cls._random_noise_overlay_test(img_comp_data.data.numpy(), transforms, lower_uniform_perturbation=lower_uniform_perturbation_2, upper_uniform_perturbation=upper_uniform_perturbation_2, data_size=data_size, debug_show_images=debug_show_images, mode=mode)

        X = [img_comp_data, rand_noise_data_1, rand_noise_data_2]
        X = torch.concat(X, dim=0)
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def _vertical_flip(cls, train_set, debug_show_images = False, additional_transforms = []):

        transforms = [torchvision.transforms.functional.vflip] + additional_transforms

        X = cls._apply_standard_augmentations(train_set, transforms, debug_show_images=debug_show_images)
        return X


    @classmethod    
    def _horizontal_flip(cls, train_set, debug_show_images = False, additional_transforms = []):

        transforms = [torchvision.transforms.functional.hflip] + additional_transforms

        X = cls._apply_standard_augmentations(train_set, transforms, debug_show_images=debug_show_images)
        return X

    @classmethod    
    def _random_rotation(cls, train_data, debug_show_images = False, additional_transforms = []):

        transforms = [torchvision.transforms.RandomRotation((0,360))] + additional_transforms
        
        X = cls._apply_standard_augmentations(train_data, transforms, debug_show_images=debug_show_images)
        return X


    @classmethod
    def _apply_standard_augmentations(cls, data, transforms, debug_show_images=False):
        
        to_tensor_transform = torchvision.transforms.ToTensor()
        X = []

        if debug_show_images is True:
            number_of_examples = 3
            count = 0

        for im in data.data.numpy():
            im = Image.fromarray(im, mode='L')

            im = to_tensor_transform(im)

            if debug_show_images is True:
                plt.pyplot.imshow(im[0,:,:], cmap='gray')
                plt.pyplot.show()

            for t in transforms:
                im = t(im)
            X.append(im)

            if debug_show_images is True:
                plt.pyplot.imshow(im[0,:,:], cmap='gray')
                plt.pyplot.show()

                if count < number_of_examples:
                    count += 1
                else:
                    break

        X = torch.concat(X)

        return X

    @classmethod
    def mnist_horizontal_flip_reduced_mnist_5k(cls, augment=None, d_in=None):
        debug_show_images = glob_debug_show_images
        
        train_set = cls.get_pure_mnist()
        train_set.data = train_set.data[:5000, :,: ]
        additional_transforms = [mnist_normalization]

        X = cls._horizontal_flip(train_set, debug_show_images = debug_show_images, additional_transforms=additional_transforms)

        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_vertical_flip_reduced_mnist_5k(cls, augment=None, d_in=None):
        debug_show_images = glob_debug_show_images
        
        train_set = cls.get_pure_mnist()
        train_set.data = train_set.data[:5000, :,: ]
        additional_transforms = [mnist_normalization]

        X = cls._vertical_flip(train_set, debug_show_images = debug_show_images, additional_transforms=additional_transforms)

        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_horizontal_flip_reduced_mnist_7p5k(cls, augment=None, d_in=None):
        debug_show_images = glob_debug_show_images
        
        train_set = cls.get_pure_mnist()
        train_set.data = train_set.data[:7500, :,: ]
        additional_transforms = [mnist_normalization]

        X = cls._horizontal_flip(train_set, debug_show_images = debug_show_images, additional_transforms=additional_transforms)

        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_vertical_flip_reduced_mnist_7p5k(cls, augment=None, d_in=None):
        debug_show_images = glob_debug_show_images
        
        train_set = cls.get_pure_mnist()
        train_set.data = train_set.data[:7500, :,: ]
        additional_transforms = [mnist_normalization]

        X = cls._vertical_flip(train_set, debug_show_images = debug_show_images, additional_transforms=additional_transforms)

        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X


    @classmethod
    def mnist_5k_random_rotation(cls, augment=None, d_in=None):
        debug_show_images = glob_debug_show_images

        train_data = cls.get_pure_mnist()
        train_data.data = train_data.data[:5000, :,: ]

        additional_transforms = [mnist_normalization]

        X = cls._random_rotation(train_data, debug_show_images=debug_show_images, additional_transforms=additional_transforms)


        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def composition_sampling_random_noise_overlay_5k_5k_0_1_m1_0_reduced_mnist_5k(cls, augment=None, d_in=None):

        pure_mnist_reduced_data =DatagensGenerator.get_pure_mnist()

        pure_mnist_reduced_data.data = pure_mnist_reduced_data.data[:5000, :,: ]

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=5000, data_size=5000, lower_uniform_perturbation=0, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0, debug_show_images=glob_debug_show_images,
                                                                                data=pure_mnist_reduced_data)

        return return_vector


    @classmethod
    def composition_sampling_random_noise_overlay_5k_5k_m1_1_m1_1_reduced_mnist_5k(cls, augment=None, d_in=None):

        pure_mnist_reduced_data =DatagensGenerator.get_pure_mnist()

        pure_mnist_reduced_data.data = pure_mnist_reduced_data.data[:5000, :,: ]

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=5000, data_size=5000, lower_uniform_perturbation=-1, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=1, debug_show_images=glob_debug_show_images,
                                                                                data=pure_mnist_reduced_data)

        return return_vector

    @classmethod
    def mnist_composition_sampling_180k_0d0_x3_y3_reduced_mnist_5k(augment=None, d_in=None):
        debug_show_images = glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()
        data.data = data.data[:5000, :,: ]

        transforms = [mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 180000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_composition_sampling_15k_0d0_x3_y3_reduced_mnist_5k(augment=None, d_in=None):
        debug_show_images = glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()
        data.data = data.data[:5000, :,: ]

        transforms = [mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 15000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_composition_sampling_45k_0d0_x3_y3_reduced_mnist_5k(augment=None, d_in=None):
        debug_show_images = glob_debug_show_images

        data = DatagensGenerator.get_pure_mnist()
        data.data = data.data[:5000, :,: ]

        transforms = [mnist_normalization]

        X = DatagensGenerator._image_composition(data, transforms,
                                images_on_x_axis=3, images_on_y_axis=3, 
                                debug_show_images=debug_show_images,
                                number_of_samples = 45000)
        
        if debug_show_images is True:
            print("DONE - QUIT")
            quit()

        return X

    @classmethod
    def mnist_5k_random_noise_overlay_m1_0_87p5k(cls, augment=None, d_in = None):
        debug_show_images = glob_debug_show_images

        return cls.mnist_5k_random_noise_overlay(lower_uniform_perturbation=-1, upper_uniform_perturbation=0, data_size=87500, debug_show_images=debug_show_images)

    @classmethod
    def mnist_5k_random_noise_overlay_0_1_87p5k(cls, augment=None, d_in = None):
        debug_show_images = glob_debug_show_images

        return cls.mnist_5k_random_noise_overlay(lower_uniform_perturbation=0, upper_uniform_perturbation=1, data_size=87500, debug_show_images=debug_show_images)

    @classmethod
    def composition_sampling_random_noise_overlay_60k_60k_0_1_m1_0_reduced_mnist_5k(augment=None, d_in=None):

        pure_mnist_reduced_data =DatagensGenerator.get_pure_mnist()

        pure_mnist_reduced_data.data = pure_mnist_reduced_data.data[:5000, :,: ]

        return_vector =DatagensGenerator.composition_sampling_random_noise_overlay(composition_data=60000, data_size=60000, lower_uniform_perturbation=0, upper_uniform_perturbation=1, lower_uniform_perturbation_2=-1, upper_uniform_perturbation_2=0, debug_show_images=glob_debug_show_images,
                                                                                data=pure_mnist_reduced_data)

        return return_vector

