"""
 # Created on 12.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: class for any teacher-generated dataset
 #
"""

import os
import torch
import pickle
import models
import numpy as np

import torchvision.transforms as T

import matplotlib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import models.registry

from datasets import base
from platforms.platform import get_platform
from foundations.hparams import DatasetHparams, ModelHparams
from datagen import registry


class Dataset(base.Dataset):
    """The teacher generated dataset."""

    def num_train_examples(self): return self._num_examples

    def num_test_examples(self): return self._num_examples

    def num_classes(self): return self._num_classes

    @staticmethod
    def get_train_set(use_augmentation):
        pass

    @staticmethod
    def get_test_set():
        pass

    def __init__(self, dataset_hparams: DatasetHparams, use_augmentation, output_location: str = "", model = None, train_hparams = None):
        
        self._output_location = output_location
        self._train_hparams = train_hparams

        if dataset_hparams.teacher_name is not None:
            examples, labels, teacher = self.extract_teacher_data(dataset_hparams, use_augmentation, model = model)
        elif dataset_hparams.teacher_file is not None:  # load just a dataset of input-output pairs
            data_file = os.path.join(get_platform(), "data", "train_" + dataset_hparams.teacher_name)
            dataset = np.load(data_file)
            examples = dataset.X
            labels = dataset.y
            teacher = None

        self._teacher = teacher
        self._num_examples = examples.shape[0]
        self._num_classes = labels.shape[1]
        super(Dataset, self).__init__(examples, labels)

    def extract_teacher_data(self, dataset_hparams, use_augmentation, model = None):
        if not hasattr(dataset_hparams, "teacher_seed"):
            raise ValueError('No --teacher_seed specified!')

        if model is None:
            teacher_folder = os.path.join(get_platform().root, "train_" + dataset_hparams.teacher_name,
                                        "seed_" + dataset_hparams.teacher_seed, "main")
            model = self.get_specified_model(teacher_folder)
            teacher = model
        else:
            teacher = model

        debug_flag = False
        if debug_flag is True:

            first_layer_weights = (list(model.children())[0])[0].weight.data
            second_layer_weights = list(model.children())[1].weight.data

            #maxpool_first_layer = self.three_three_max_pool(first_layer_weights, mode="max_pool")
            #avgpool_first_layer = self.three_three_max_pool(first_layer_weights, mode="average")
            #abs_avgpool_first_layer = self.three_three_max_pool(first_layer_weights, mode="abs_average")

            layers_weights = []
            layers_weights.append(first_layer_weights)
            #layers_weights.append(maxpool_first_layer)
            #layers_weights.append(avgpool_first_layer)
            #layers_weights.append(abs_avgpool_first_layer)
            #layers_weights.append(second_layer_weights)

            #self.pca_important_weight_vectors(first_layer_weights)
            #self.pca_important_weight_vectors(second_layer_weights)


            #self.visualize_layer_0_weights(first_layer_weights)           
            self.visualize_layer_magnitude(first_layer_weights)           
            self.visualize_layer_magnitude(second_layer_weights)           
            #self.visualize_layer_1_weights(second_layer_weights)


            """
            for layer_weights in layers_weights:
                #similarity_matrix = self.calculate_layer_weight_similarity(layer_weights)
                #self.visualize_similarity_matrix(similarity_matrix)

                #l2_norms = self.calculate_l2_norms(layer_weights)
                #self.plot_l2_norms(l2_norms)
            """

            quit(0)

        data = registry.get(dataset_hparams, model, use_augmentation, output_location = self._output_location, train_hparams = self._train_hparams)

        return data[0], data[1], teacher

    def visualize_layer_magnitude(self, layer_weights):

        # Convert to logarithmic scale; add a small value to avoid log(0)
        #log_l2_norms = torch.log(l2_norms + 1e-10).numpy()

        neuron_weights = np.linalg.norm(layer_weights, axis=1)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.bar(range(neuron_weights.shape[0]), list(neuron_weights), log=False)
        plt.xlabel('Weight Vector Index')
        plt.ylabel('Weight')
        plt.title(f'Norm of neurons')

        #plt.savefig(f"imgs/512_teacher_layer_1/output_neuron_{neuron_index}")
        plt.show()


    def visualize_layer_1_weights(self, layer_weights):

        # Convert to logarithmic scale; add a small value to avoid log(0)
        #log_l2_norms = torch.log(l2_norms + 1e-10).numpy()

        num_neurons = layer_weights.shape[0]

        for neuron_index in range(num_neurons):
            
            # Plotting
            plt.figure(figsize=(10, 6))
            plt.bar(range(layer_weights.shape[1]), list(layer_weights[neuron_index,:].numpy()), log=False)
            plt.xlabel('Weight Vector Index')
            plt.ylabel('Weight')
            plt.title(f'Weight of Neuron {neuron_index}')

            plt.savefig(f"imgs/512_teacher_layer_1/output_neuron_{neuron_index}")
            #plt.show()



    
    def visualize_layer_0_weights(self, layer_weights):

        matplotlib.use('Agg')

        quadratic_image_size = 27
        num_neurons = layer_weights.shape[0]
        for neuron_index in range(num_neurons):

            first_neuron = layer_weights[neuron_index,:]

            # Initialize an empty tensor for storing cosine similarities
            #num_vectors = first_neuron.shape[0]
            #num_vectors = quadratic_image_size
            #similarity_matrix = torch.zeros((num_vectors, num_vectors))

            similarity_matrix = (first_neuron.numpy()).reshape(28,28)
            
            # Calculate cosine similarity between each pair of weight vectors
            #for x in range(0, quadratic_image_size):
            #    for y in range(0, quadratic_image_size):
            #        position_in_array = y * quadratic_image_size + x
            #        similarity_matrix[x][y] = first_neuron[position_in_array]


            fig, ax = plt.subplots()

            color_list_size = [
                "#000000",
                "#2a4858",
                "#00898a",
                "#64c987",
                "#ffffff",
                "#fafa6e",
                "#ffc21e",
                "#ff8000",
                "#ff0000"
            ]
            
            color_list_time = color_list_size + ["#ffffff"]
            
            cmap1 = matplotlib.colors.ListedColormap(color_list_size)
            cmap2 = matplotlib.colors.ListedColormap(color_list_time)

            
            bounds= [-1,-0.5,-0.2, -0.1,-0.05,0.05,0.1,0.2,0.5,1]
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap1.N)
            """
            plt.scatter(data_X, data_Y, linewidth=1,edgecolor="black",s=180,c=data_Z,
                        marker=MarkerStyle("o", fillstyle="right"),
                    cmap = cmap1,
                    norm = norm)
            """

            cax = ax.matshow(similarity_matrix, cmap=cmap1, norm=norm)  # 'viridis', 'plasma', 'inferno', etc.
            
            # Add color bar on the right
            fig.colorbar(cax)

            # Adding titles and labels
            plt.title(f"Visualize Neuron {neuron_index}")
            plt.xlabel('Weight Vector Index')
            plt.ylabel('Weight Vector Index')

            # Add grid lines to separate cells for better visibility
            ax.set_xticks(np.arange(similarity_matrix.shape[1]+1)-.5, minor=True)
            ax.set_yticks(np.arange(similarity_matrix.shape[0]+1)-.5, minor=True)
            ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
            ax.tick_params(which="minor", size=0)

            plt.savefig(f"imgs/512_teacher_layer_0/neuron_{neuron_index}")
            #plt.show()



    
    def pca_important_weight_vectors(self, layer_weights):

        pca = PCA().fit(layer_weights.numpy())

        plt.rcParams["figure.figsize"] = (12,6)

        fig, ax = plt.subplots()
        xi = np.arange(1, pca.explained_variance_ratio_.shape[0] + 1, step=1)
        y = np.cumsum(pca.explained_variance_ratio_)

        plt.ylim(0.0,1.1)
        plt.plot(xi, y, marker='o', linestyle='--', color='b')

        plt.xlabel('Number of Components')
        plt.xticks(np.arange(0, 11, step=1)) #change from 0-based array index to 1-based human-readable label
        plt.ylabel('Cumulative variance (%)')
        plt.title('The number of components needed to explain variance')

        plt.axhline(y=0.95, color='r', linestyle='-')
        plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

        ax.grid(axis='x')
        plt.show()



    
    def three_three_max_pool(self, layer_weights, mode="max_pool"):
        """
        Three by three maxpool on weights
        """

        max_pooled_layer_weights = []

        for neuron_index in range(layer_weights.shape[0]):

            max_pooled_weights_neuron = []

            for x in range(0,27,1):
                for y in range(0,27,1):

                    values = []
                    
                    for conv_x in range(-1,2,1):
                        for conv_y in range(-1,2,1):
                            tmp_x = x + conv_x
                            tmp_y = y + conv_y
                            if tmp_x < 0 or tmp_x >= 27 or tmp_y < 0 or tmp_y >= 27:
                                continue

                            position_in_array = tmp_y * 27 + tmp_x

                            values.append(float(layer_weights[neuron_index, position_in_array]))

                    if mode == "max_pool":
                        max_pooled_weights_neuron.append(max(values))
                    elif mode == "average":
                        max_pooled_weights_neuron.append((sum(values)/len(values)))
                    elif mode == "abs_average":
                        max_pooled_weights_neuron.append((sum([abs(value) for value in values])/len(values)))

            max_pooled_layer_weights.append(max_pooled_weights_neuron) 
                    

        return_tensor = torch.Tensor(max_pooled_layer_weights)
        return return_tensor




        
    def calculate_layer_weight_similarity(self, layer_weights):
        # Flatten each weight vector if not already 1D and ensure float type for cosine similarity
        if len(layer_weights.shape) > 2:
            # For convolutional layers, we reshape the weights
            num_filters = layer_weights.shape[0]
            layer_weights_flat = layer_weights.view(num_filters, -1).float()
        else:
            # For fully connected layers, the weights are already 2D (but we ensure float type)
            layer_weights_flat = layer_weights.float()
        
        # Initialize an empty tensor for storing cosine similarities
        num_vectors = layer_weights_flat.shape[0]
        similarity_matrix = torch.zeros((num_vectors, num_vectors))
        
        # Calculate cosine similarity between each pair of weight vectors
        for i in range(num_vectors):
            for j in range(i, num_vectors):  # Start from i to avoid recalculating symmetric entries
                similarity = torch.nn.functional.cosine_similarity(
                    layer_weights_flat[i].unsqueeze(0),
                    layer_weights_flat[j].unsqueeze(0)
                )
                similarity_matrix[i][j] = similarity
                if i != j:
                    if float(similarity) >= 0.9:
                        print(f">> SIMILARITY (>= 0.9)>>(i={i}, j={j})={similarity}")
                    similarity_matrix[j][i] = similarity  # Fill in the symmetric entry
        
        return similarity_matrix
    
    def visualize_similarity_matrix(self, similarity_matrix, title='Similarity Matrix'):
        fig, ax = plt.subplots()
        cax = ax.matshow(similarity_matrix, cmap='viridis')  # 'viridis', 'plasma', 'inferno', etc.
        
        # Add color bar on the right
        fig.colorbar(cax)

        # Adding titles and labels
        plt.title(title)
        plt.xlabel('Weight Vector Index')
        plt.ylabel('Weight Vector Index')

        # Add grid lines to separate cells for better visibility
        ax.set_xticks(np.arange(similarity_matrix.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(similarity_matrix.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        plt.show()

    def calculate_l2_norms(self, weight_matrix):
        # Calculate the L2 norm for each weight vector (along the first dimension, i.e., for each row)
        l2_norms = torch.sqrt((weight_matrix ** 2).sum(dim=1))
        return l2_norms

    def plot_l2_norms(self, l2_norms):
        # Convert to logarithmic scale; add a small value to avoid log(0)
        log_l2_norms = torch.log(l2_norms + 1e-10).numpy()
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(log_l2_norms)), log_l2_norms, log=True)
        plt.xlabel('Weight Vector Index')
        plt.ylabel('Logarithmic L2 Norm')
        plt.title('Logarithmic L2 Norms of Weight Vectors')
        plt.show()


    @staticmethod
    def get_specified_model(directory):
        """ Returns the final saved model from the directory """
        all_models = [d for d in os.listdir(directory) if d.startswith("model_")]
        eps = [int(model_name.split("_")[1][2:]) for model_name in all_models]
        id_max_ep = np.argmax(eps)
        model_state_dict = torch.load(os.path.join(directory, all_models[id_max_ep]),
                                      map_location=get_platform().torch_device)
        loaded_model_hparams = torch.load(os.path.join(directory, "hparams_dict"))["model_hparams"]
        loaded_model_hparams = ModelHparams.create_from_args(loaded_model_hparams)
        model = models.registry.get(loaded_model_hparams)
        model.load_state_dict(model_state_dict)
        return model

    def __getitem__(self, index):
        example, label = self._examples[index], self._labels[index]
        return example, label

    # IMPORTANT: the base class implements this as labels.size (since labels are scalar digits), while here the
    # labels are 10-dimensional vectors of floats
    def __len__(self):
        return self.num_train_examples()


DataLoader = base.DataLoader
