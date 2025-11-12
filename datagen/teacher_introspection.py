"""
 - For generation of teacher introspection plots
 - I.e., plotting pre- and post-activations of certain datasets
"""
import os
import torchvision
import numpy as np
import multiprocessing
import torch

from functools import partial
from PIL import Image
from platforms.platform import get_platform
from matplotlib import pyplot as plt

from .datagens_generators import DatagensGenerator

def teacher_aware_weights_test_helper(neuron_index, first_layer_weights = None, first_layer_biases = None,  X = None):
    """
    Helper method used during (fist) multiprocessing step.
    """

    neuron = first_layer_weights[neuron_index,:]
    bias = first_layer_biases[neuron_index]
    curr_mnist_activations = []

    for data_point in X:
        curr_mnist_activations.append(float(np.dot(neuron, data_point[0,:]) + bias))

    return curr_mnist_activations

def teacher_aware_weights_test_helper_2(neuron_activations, classes_list = None, class_indices_list = None):
    """
    Helper (2) method used during (second) multiprocessing step.
    """
    #neuron_activations = w_mnist_activations[neuron_index]

    class_activations_lists = []

    for _class in classes_list:

        class_indices = class_indices_list[_class]

        class_activations = [neuron_activations[class_index] for class_index in class_indices]

        class_activations_lists.append(class_activations)

    return class_activations_lists

class TeacherIntrospectionCaller:

    @classmethod
    def teacher_aware_weights_test(cls, augment=None, d_in=None, model=None):

        introspection = TeacherIntrospection(False)

        values = introspection.teacher_aware_weights_test(augment=augment, d_in=d_in, model=model)

        return values


class TeacherIntrospection:


    def __init__(self, glob_debug_show_images):

        self.mnist_normalized_min_value = -0.42 # If Data available: Use torch.min(...)
        self.mnist_normalized_max_value = 2.821 # If Data available: Use torch.max(...)
        self.mnist_normalization = torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])
        self.glob_debug_show_images = glob_debug_show_images 

    def teacher_aware_weights_test(self, augment=None, d_in=None, model=None):
        """
        Activations for teacher, etc.
        """

        number_different_plots = 8
        number_parallel_threads = 10
        plot_activations_per_class_flag = False

        #data = DatagensGenerator.mnist_image_composition(number_of_samples=60000)
        #data = DatagensGenerator.random_noise_single_multiplication(sample_size=60000, min=0, max = 1, smin=1, smax=1, mode="+")
        #data = DatagensGenerator.random_noise_single_multiplication(sample_size=600, min=0, max = 1, smin=1, smax=1, mode="+")

        #data = DatagensGenerator.mnist_reduced_5k()
        #data = DatagensGenerator.mnist()

        #mnist+mnist_horizontal_flip+mnist_vertical_flip


        #datagens = [
        #    DatagensGenerator.mnist_reduced_5k(), 
        #    DatagensGenerator.mnist_5k_random_noise_overlay(lower_uniform_perturbation=0, upper_uniform_perturbation=1, data_size=5000, debug_show_images=False),
        #    DatagensGenerator.mnist_5k_random_noise_overlay(lower_uniform_perturbation=-1, upper_uniform_perturbation=0, data_size=5000, debug_show_images=False),
        #            ]

        #datagens = [
        #    DatagensGenerator.mnist_reduced_5k(), 
        #    DatagensGenerator.mnist_horizontal_flip_reduced_mnist_5k(),
        #    DatagensGenerator.mnist_vertical_flip_reduced_mnist_5k(),        
        #]


        #datagens = [
        #    DatagensGenerator.mnist_reduced_5k(), 
        #    DatagensGenerator.mnist_5k_random_rotation(),
        #    DatagensGenerator.mnist_5k_random_rotation()
        #]

        #datagens = [
        #    DatagensGenerator.mnist_reduced_5k(), 
        #    DatagensGenerator.mnist_5k_random_noise_overlay(lower_uniform_perturbation=-1, upper_uniform_perturbation=1, data_size=10000, debug_show_images=False),
        #            ]

        #data = DatagensGenerator.composition_sampling_random_noise_overlay_5k_5k_0_1_m1_0_reduced_mnist_5k()
        #data = DatagensGenerator.mnist_reduced_15k()
        #data = DatagensGenerator.composition_sampling_random_noise_overlay_5k_5k_m1_1_m1_1_reduced_mnist_5k()

        """
        datagens = [
                DatagensGenerator.mnist_reduced_5k(), 
                DatagensGenerator.mnist_5k_random_noise_overlay_0_1_87p5k(),
                DatagensGenerator.mnist_5k_random_noise_overlay_m1_0_87p5k()
        ]
        """

        #data = DatagensGenerator.mnist_reduced_5k()
        #data = DatagensGenerator.mnist()
        #data = DatagensGenerator.mnist_composition_sampling_15k_0d0_x3_y3_reduced_mnist_5k()
        #data = DatagensGenerator.mnist_composition_sampling_15k_0d0_x3_y3_reduced_mnist_5k()
        #data = DatagensGenerator.mnist_composition_sampling_45k_0d0_x3_y3_reduced_mnist_5k()

        #data = DatagensGenerator.composition_sampling_random_noise_overlay_60k_60k_0_1_m1_0_reduced_mnist_5k()

        #data = DatagensGenerator.mnist_composition_sampling_180k_0d0_x3_y3_reduced_mnist_5k()

        dataset_filename_list = [
            #([DatagensGenerator.mnist_reduced_15k()], "00_MNIST15k"),
            ([DatagensGenerator.mnist_reduced_5k()],"0009_MNIST_RAND_ROT"),
            #([DatagensGenerator.mnist_reduced_5k(), DatagensGenerator.mnist_5k_random_rotation(),DatagensGenerator.mnist_5k_random_rotation()],"09_MNIST_RAND_ROT"),
            #([DatagensGenerator.mnist_reduced_5k(), DatagensGenerator.mnist_horizontal_flip_reduced_mnist_5k(), DatagensGenerator.mnist_vertical_flip_reduced_mnist_5k()], "10_MNIST_HVFLIP"),
            #
            #([DatagensGenerator.mnist_7p5k_random_noise_overlay(lower_uniform_perturbation=0, upper_uniform_perturbation=1, data_size=7500, debug_show_images=False), DatagensGenerator.mnist_7p5k_random_noise_overlay(lower_uniform_perturbation=-1, upper_uniform_perturbation=0, data_size=7500, debug_show_images=False)], "03_ETA_0_1"),
            #([DatagensGenerator.mnist_15k_random_noise_overlay(lower_uniform_perturbation=-0.5, upper_uniform_perturbation=0.5, data_size=1500, debug_show_images=False)], "04_ETA_m0p5_0p5"),
            #([DatagensGenerator.mnist_15k_random_noise_overlay(lower_uniform_perturbation=-1, upper_uniform_perturbation=1, data_size=15000, debug_show_images=False)], "05_ETA_m1_1"),
            #
            #([DatagensGenerator.mnist_reduced_5k(), DatagensGenerator.mnist_5k_random_noise_overlay(lower_uniform_perturbation=0, upper_uniform_perturbation=1, data_size=5000, debug_show_images=False), DatagensGenerator.mnist_5k_random_noise_overlay(lower_uniform_perturbation=-1, upper_uniform_perturbation=0, data_size=5000, debug_show_images=False)], "06_MNIST_ETA_0_1"),
            #([DatagensGenerator.mnist_reduced_5k(), DatagensGenerator.mnist_5k_random_noise_overlay(lower_uniform_perturbation=-0.5, upper_uniform_perturbation=0.5, data_size=10000, debug_show_images=False)], "07_MNIST_ETA_m0p5_0p5"),
            #([DatagensGenerator.mnist_reduced_5k(), DatagensGenerator.mnist_5k_random_noise_overlay(lower_uniform_perturbation=-1, upper_uniform_perturbation=1, data_size=10000, debug_show_images=False)], "08_MNIST_ETA_m1_1"),
        ]
        #all_my_data = [([DatagensGenerator.mnist_reduced_15k()], "00_MNIST15k")]

        for datagens, file_name in dataset_filename_list:
            if len(datagens) > 1:
                with torch.no_grad():
                    data = torch.concat(datagens, dim=0)
            else:
                data = datagens[0]

            # Reshape data (tenso) into a list of images (each image only 1-dimension)
            data = (list(data.reshape(data.shape[0],1,-1).numpy()))

            self.compute_activations_and_plot_activations(model, data, number_different_plots,
                                                        number_parallel_threads, plot_activations_per_class_flag,
                                                        #csv_name="mnist_5k_rand_rot_activations")
                                                        csv_name=file_name)

            print("<<<<<>>>>")
            print("Teacher aware weights test has finished! - Quitting!")
            print("<<<<<>>>>")

        quit(0)

    def compute_activations_and_plot_activations(self, model, data, number_different_plots,
                                                 number_parallel_threads, plot_activations_per_class_flag,
                                                 csv_name = "activations"):

        first_layer_weights = (list(model.children())[0])[0].weight.data
        first_layer_biases = (list(model.children())[0])[0].bias.data

        w_squared, w_squared_activation = self.compute_w_squared(first_layer_weights, first_layer_biases)

        w_mnist_pre_activations = []
        
        pool = multiprocessing.Pool(number_parallel_threads)

        w_mnist_pre_activations = pool.map(partial(teacher_aware_weights_test_helper, first_layer_weights = first_layer_weights, first_layer_biases = first_layer_biases, X = data),
                                           list(range(first_layer_weights.shape[0])))

        w_mnist_pre_activations_np = np.array(w_mnist_pre_activations)

        if plot_activations_per_class_flag is True:
            self.plot_activations_per_class(w_mnist_pre_activations, number_parallel_threads=number_parallel_threads, number_different_plots=number_different_plots)

        np.savetxt(f"{csv_name}_pre.csv", w_mnist_pre_activations_np, delimiter=",", fmt="%.6f")

        pre_tensor = torch.tensor(w_mnist_pre_activations_np).cpu()
        post_tensor = model.act_fun(pre_tensor)
        w_mnist_post_activations_np = post_tensor.numpy()
        np.savetxt(f"{csv_name}_post.csv", w_mnist_post_activations_np, delimiter=",", fmt="%.6f")

        #bplot = plt.boxplot(w_mnist_activations)
        #plt.ylim(-15,15)
        #plt.show()

        """
        print("CIAO!")
        quit()

        bplot = plt.boxplot([w_squared, w_squared_activation])
        plt.show()

        print("BYE!")
        quit()
        return w_squared
        """


    def plot_activations_per_class(self, w_mnist_activations, number_parallel_threads = 1, number_different_plots = 1):
       
        labels = [] 

        train_set = DatagensGenerator.mnist()

        class_indices_list = []

        for _class in list((train_set.class_to_idx).values()):
            class_indices = [index for index in range(len(labels)) if labels[index] == _class]

            class_indices_list.append(class_indices)

        print("FINISHED ACTIVATIONS -> NOW ACTIVATIONS PER CLASS")


        pool = multiprocessing.Pool(number_parallel_threads)
        #activations_per_neuron_per_class = pool.map(partial(teacher_aware_weights_test_helper_2(w_mnist_activations=w_mnist_activations, train_set=train_set, class_indices_dict=class_indices_dict)), list(range(first_layer_weights.shape[0])))
        classes_list = list((train_set.class_to_idx).values())
        #activations_per_neuron_per_class = pool.map(partial(teacher_aware_weights_test_helper_2, classes_list = classes_list, class_indices_list=class_indices_list), list(range(first_layer_weights.shape[0])))
        activations_per_neuron_per_class = pool.map(partial(teacher_aware_weights_test_helper_2, classes_list = classes_list, class_indices_list=class_indices_list), w_mnist_activations)
        activations_per_class_dict = {}

        print("FINISHED ACTIVATIONS_PER_CLASS")

        for _class in list((train_set.class_to_idx).values()):
            activations_per_class = [activations_per_neuron[_class] for activations_per_neuron in activations_per_neuron_per_class]

            activations_per_class_dict[_class] = activations_per_class

        for key in activations_per_class_dict.keys():

            cur_activations = activations_per_class_dict[key]

            cur_activations_lists = [_list.tolist() for _list in np.array_split(cur_activations, number_different_plots)]

            counter = 0
            for cur_activation_list in cur_activations_lists:
                bplot = plt.boxplot(cur_activation_list)
                plt.title(f"Activation Boxplot for Class/Number: {key} - {counter}")
                # TODO: Adjust
                plt.show()
                #plt.savefig(f"/home/thinklex/Dropbox/2019_x_Studium/Erasmus/02_university/00_project/07_theory_stuff/activation_images/layer_1_activations_number_{key}_{counter}.svg", dpi=400)
                plt.clf()

                counter += 1

    @classmethod
    def compute_w_squared(cls, weights, biases):
        w_squared = []
        w_squared_activation = []
        for neuron_index in range(weights.shape[0]):

            neuron = weights[neuron_index,:]
            neuron_squared = np.dot(neuron, neuron)

            w_squared.append(neuron_squared)

            w_squared_activation.append(neuron_squared + biases[neuron_index])

        return w_squared, w_squared_activation
