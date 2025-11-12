"""
 # Created on 10.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: fully connected MNIST net
 #
"""

import torch
import torch.nn as nn

import os
import numpy as np

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global

from foundations.precision import Precision


class Model(base.Model):
    '''Fully connected from data'''

    def __init__(self, plan, initializer, act_fun,
                 all_weights, all_biases,
                 outputs=10, input_dimensions = 2,
                 precision=None):
        super(Model, self).__init__()

        self.act_fun = act_fun

        layers = []
        current_size = input_dimensions
        for size in plan:
            layers.append(nn.Linear(current_size, size))
            current_size = size

        self.fc_layers = nn.ModuleList(layers)
        self.fc = nn.Linear(current_size, outputs)

        self.precision = Precision.get_precision_from_string(precision)


        with torch.no_grad():
            for layer_weights_index in range(len(all_weights)):
                layer_weights = all_weights[layer_weights_index]
                layer_bias = all_biases[layer_weights_index]

                weight_parameters = nn.Parameter(torch.tensor(layer_weights, dtype=self.precision))
                weight_biases = nn.Parameter(torch.tensor(layer_bias, dtype=self.precision))

                if layer_weights_index < len(all_weights) - 1:
                    self.fc_layers[layer_weights_index].weight = weight_parameters
                    self.fc_layers[layer_weights_index].bias = weight_biases
                else:
                    self.fc.weight = weight_parameters
                    self.fc.bias = weight_biases

        self.criterion = nn.CrossEntropyLoss()

        #self.apply(initializer)

    def forward(self, x):

        if x.dtype != self.precision:
            print("DTYPE CONVERSION")
            x = x.to(self.precision)



        x = x.view(x.size(0), -1)  # Flatten.
        for layer in self.fc_layers:
            x = self.act_fun(layer(x))

        return self.fc(x)

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return model_name.startswith('fully_connected_specified')

    @staticmethod
    def get_model_from_name(model_name, initializer, act_fun, model_hparams, outputs=None):
        """The name of a model is mnist_lenet_N1[_N2...].

        N1, N2, etc. are the number of neurons in each fully-connected layer excluding the
        output layer (10 neurons by default). A LeNet with 300 neurons in the first hidden layer,
        100 neurons in the second hidden layer, and 10 output neurons is 'mnist_lenet_300_100'.
        """

        experiment_path = os.path.join("exps","shallow_visualizations_test","g")
        model_specification = model_hparams.model_specification

        date, time = model_specification.split("_")

        outputs = outputs or 1

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        layer_sizes = model_name.split('_')[3:]
        plan = [int(n) for n in layer_sizes]

        all_weights = []
        all_biases = []
        for layer_index in range(len(layer_sizes) + 1):

            student_layer_file_name = f"{date}_{time}_layer_{layer_index}.csv"
            student_weights_path = os.path.join(experiment_path, date, student_layer_file_name)

            if os.path.isfile(student_weights_path):
                hidden_layer = np.genfromtxt(student_weights_path, delimiter=',', skip_header=0)
            else:
                break

            if len(hidden_layer.shape) > 1:
                biases = list(hidden_layer[:, -1])
                hidden_layer_weights = hidden_layer[:, :-1]
                weights = []
                for row_index in range(hidden_layer_weights.shape[0]):
                    weights.append(list(hidden_layer_weights[row_index, :]))

            else:
                biases = [hidden_layer[-1]]
                weights = [list(hidden_layer[:-1])]
                
            all_weights.append(weights) 
            all_biases.append(biases)

        return Model(plan, initializer, act_fun,
                     all_weights, all_biases,
                     outputs = outputs, input_dimensions=2)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():

        #raise NotImplementedError("Default Hparams not implemented for fully_connected_specified")

        model_hparams = hparams.ModelHparams(
            model_name='fully_connected_specified_1',
            model_init='kaiming_normal',
            batchnorm_init='uniform'
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='tbd',
            batch_size=128
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='adam',
            lr=0.1,
            training_steps='40ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore='fc.weight',
        )

        extraction_hparams = hparams.ExtractionHparams(
            gamma=None,
            beta=None
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams, extraction_hparams)
