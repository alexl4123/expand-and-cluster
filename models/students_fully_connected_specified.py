"""
 # Created on 11.09.23
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Students of the MNIST lenet (fully connected) network types
 #
"""
import typing
import os

import numpy as np
import torch
import torch.nn as nn

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global
from models.activation_functions import identity

from foundations.precision import Precision


class Model(base.Model):
    '''Training many LeNet fully-connected models in parallel for MNIST'''

    class InitialParallelFCModule(nn.Module):
        """A module for N linear layers run in a unique tensor (first layer only)."""

        def __init__(self, d_in, d_out, N, act_fun):
            super(Model.InitialParallelFCModule, self).__init__()
            self.fc = nn.Parameter(torch.zeros(d_in, d_out, N))
            self.b = nn.Parameter(torch.zeros(d_out, N))
            self.act_fun = act_fun

        def forward(self, x):
            """ i:input size -- b:sample batch size -- h:hidden layer dim -- o:output_size -- n:number of nets """
            return self.act_fun(torch.einsum('bi,ihn->bhn', x, self.fc) +
                   self.b.expand([x.shape[0]] + list(self.b.shape)))

    class ParallelFCModule(nn.Module):
        """A module for N linear layers run in a unique tensor."""

        def __init__(self, d_in, d_out, N, act_fun):
            super(Model.ParallelFCModule, self).__init__()
            self.fc = nn.Parameter(torch.zeros(d_in, d_out, N))
            self.b = nn.Parameter(torch.zeros(d_out, N))
            self.act_fun = act_fun

        def forward(self, x):
            """ i:input size -- b:sample batch size -- h:hidden layer dim -- o:output_size -- n:number of nets """
            return self.act_fun(torch.einsum('bin,ihn->bhn', x, self.fc) +
                   self.b.expand([x.shape[0]] + list(self.b.shape)))

    def __init__(self, plan, initializer, act_fun, outputs=10,
                 students_weights = None, students_biases = None,
                 input_dimensions = 2,
                 precision = None):
        super(Model, self).__init__()


        self.act_fun = act_fun
        self.plan = plan
        self.N = plan[0]
        self.initializer = initializer
        self.outputs = outputs


        self.precision = Precision.get_precision_from_string(precision)

        layers = []
        current_size = input_dimensions
        for i, size in enumerate(self.plan[1:]):
            if i == 0:  # The first layer has a different dimensionality (input tensor is not 3D)
                layers.append(self.InitialParallelFCModule(current_size, size, self.N, self.act_fun))
            else:
                layers.append(self.ParallelFCModule(current_size, size, self.N, self.act_fun))
            current_size = size
        layers.append(self.ParallelFCModule(current_size, outputs, self.N, identity()))

        self.fc_layers = nn.ModuleList(layers)

        if students_weights is not None and students_biases is not None:
            with torch.no_grad():
                for student_index in range(len(students_weights)):
                    for layer_index in range(len(students_weights[student_index])):
                        to_set_weights = students_weights[student_index][layer_index]
                        to_set_biases = students_biases[student_index][layer_index]

                        weights_parameter = nn.Parameter(torch.tensor(to_set_weights, dtype=self.precision))
                        bias_parameter = nn.Parameter(torch.tensor(to_set_biases, dtype=self.precision))
                        self.fc_layers[layer_index].fc[:,:,student_index] = weights_parameter.T
                        self.fc_layers[layer_index].b[:,student_index] = bias_parameter

        self.criterion = self.loss_fn

        if self.initializer is not None:
            self.apply(self.initializer)

    def forward(self, x):

        if x.dtype != self.precision:
            print("DTYPE CONVERSION")
            x = x.to(self.precision)

        x = x.view(x.size(0), -1)  # Flatten.
        for layer in self.fc_layers:
            x = layer(x)
        return x

    @property
    def output_layer_names(self):
        out_name = list(self.named_modules())[-1][0]
        return [f'{out_name}.fc', f'{out_name}.b']

    @staticmethod
    def is_valid_model_name(model_name):
        return model_name.startswith('students_fully_connected_specified(')

    @staticmethod
    def get_model_from_name(model_name, initializer, act_fun, model_hparams, outputs=None, dataset_hparams = None):
        """The name of a model is students_mnist_lenet_(N)_W1[_W2...].

        W1, W2, etc. are the number of neurons in each fully-connected layer excluding the
        output layer (10 neurons by default). The number of nets run in parallel is set by the parameter N. To run 5
        LeNets with 300 neurons in the first hidden layer, 100 neurons in the second hidden layer, and 10 output
        neurons -> 'mnist_lenet_students(5)_300_100'. If N is omitted (i.e. mnist_lenet_students()_W1[_W2...] then N
        is set to 1).
        """

        if dataset_hparams is None:
            experiment_path = os.path.join("exps","shallow_visualizations_test","ec_g")
            model_specification = model_hparams.model_specification

            date, time = model_specification.split("_")

            outputs = outputs or 1

            if not Model.is_valid_model_name(model_name):
                raise ValueError('Invalid model name: {}'.format(model_name))
            # TODO: change '(' and ')' into some other characters. They are escape chars for the command line : (
            N = int(model_name[model_name.find("(")+1:model_name.find(")")])
            if N == "":
                N = 1
            plan = [N]

            layer_sizes = model_name.split('_')[4:]

            plan.extend([int(n) for n in layer_sizes])


            students_weights = []
            students_biases = []

            no_more_specified_students = False
            for student_index in range(N):
                student_weights = []
                student_biases = []
                for layer_index in range(len(layer_sizes) + 1):

                    student_layer_file_name = f"{date}_{time}_student_{student_index}_layer_{layer_index}.csv"
                    student_weights_path = os.path.join(experiment_path, date, student_layer_file_name)

                    if os.path.isfile(student_weights_path):
                        hidden_layer = np.genfromtxt(student_weights_path, delimiter=',', skip_header=0)
                    else:
                        no_more_specified_students = True
                        break

                    if len(hidden_layer.shape) > 1:
                        bias = list(hidden_layer[:, -1])
                        hidden_layer_weights = hidden_layer[:, :-1]
                        weights = []
                        for row_index in range(hidden_layer_weights.shape[0]):
                            weights.append(list(hidden_layer_weights[row_index, :]))
                    else:
                        bias = [hidden_layer[-1]]
                        weights = [list(hidden_layer[:-1])]

                    student_biases.append(bias) 
                    student_weights.append(weights)

                if no_more_specified_students == True:
                    break

                students_weights.append(student_weights)
                students_biases.append(student_biases)
            model = Model(plan, initializer, act_fun, outputs, students_weights = students_weights, students_biases = students_biases)

        else:

            student_specification = dataset_hparams.student_specification

            if not Model.is_valid_model_name(student_specification):
                raise ValueError('Invalid model name: {}'.format(student_specification))
            # TODO: change '(' and ')' into some other characters. They are escape chars for the command line : (
            N = int(student_specification[student_specification.find("(")+1:model_name.find(")")])
            if N == "":
                N = 1
            plan = [N]

            layer_sizes = model_name.split('_')[4:]
            plan.extend([int(n) for n in layer_sizes])

            model = Model(plan, initializer, act_fun, outputs, students_weights = None, students_biases = None)

        return model

    @staticmethod
    def loss_fn(y_hat, y):
        """y_hat is the network's output, y is the label"""
        individual_losses = Model.individual_losses(y_hat, y)
        overall_loss = individual_losses.sum()
        return overall_loss

    @staticmethod
    def individual_losses(y_hat, y):

        device = y_hat.device

        return_tensor = torch.zeros(y_hat.shape[-1])
        return_tensor = return_tensor.to(device)

        for student_index in range(y_hat.shape[-1]):
            return_tensor[student_index] = ((y_hat[:,:,student_index] - y).square().mean())
           
        return return_tensor

    @property
    def loss_criterion(self):
        return self.criterion


    @property
    def prunable_layer_names(self) -> typing.List[str]:
        """A list of the names of Tensors of this model that are valid for pruning.
        By default, only the weights of convolutional and linear layers are prunable, not biases.
        """

        return [name + '.fc' for name, module in self.named_modules() if
                isinstance(module, self.InitialParallelFCModule) or
                isinstance(module, self.ParallelFCModule)]


    @staticmethod
    def default_hparams():

        model_hparams = hparams.ModelHparams(
            model_name='students_mnist_lenet(10)_300_100',
            model_init='kaiming_normal',
            batchnorm_init='uniform'
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='mnist',
            batch_size=512
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='adam',
            lr=0.001,
            training_steps='25000ep',
            lr_scheduler="plateau",
            delta=1/np.sqrt(10),
            patience=1000,
            cooldown=2000
        )

        last_hidden_idx = len(model_hparams.model_name.split('_')[1:])

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore=f'fc_layers.{last_hidden_idx}.fc'  # By default do not prune output layer
        )

        extraction_hparams = hparams.ExtractionHparams(
            gamma=0.5,
            beta=6,
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams, extraction_hparams)
