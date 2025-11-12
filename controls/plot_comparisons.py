"""
 # Created on 08.12.2023
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: plotting comparisons between OTO trained student and teacher
 #
"""
import os
import torch
import pathlib
import numpy as np

import datasets.mnist, datasets.teacher_dataset, datasets.registry
import platforms.registry
import platforms.platform

from controls.control_utils import teacher_comparison
from foundations import paths
from foundations.hparams import DatasetHparams
from models import mnist_lenet, initializers, activation_functions
from utils.utils import find_final_model_step

folder = pathlib.Path().resolve().joinpath('controls', 'oto')

platform = platforms.registry.get('local')(num_workers=0)
platforms.platform._PLATFORM = platform

cluster_mask = datasets.mnist.Dataset.get_boruta_mask()
cluster_mask = np.concatenate([cluster_mask, [1.0]])

dataset_hparams = DatasetHparams(dataset_name='teacher', batch_size=512, teacher_name='f32850c8f7',
                                 teacher_seed='1', samples=60000, datagen='mnist', d_in=784)

teacher_folder = os.path.join(platform.root, "train_" + dataset_hparams.teacher_name,
                              "seed_" + dataset_hparams.teacher_seed, "main")
teacher = datasets.teacher_dataset.Dataset.get_specified_model(teacher_folder)

student = mnist_lenet.Model([40], initializers.kaiming_normal, activation_functions.relu())
train_loader = datasets.registry.get(dataset_hparams, train=True)

maybe_train_end_step = find_final_model_step(folder, train_loader.iterations_per_epoch)
if maybe_train_end_step is not None:  # load saved model before returning
    state_dict = platform.load_model(paths.model(folder, maybe_train_end_step),
                                     map_location=platform.torch_device)
    student.load_state_dict(state_dict)

symmetry = "even_linear_positive_scaling"

teacher_comparison(teacher, student, symmetry, cluster_mask, folder)
