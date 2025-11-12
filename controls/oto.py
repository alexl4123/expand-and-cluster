"""
 # Created on 05.12.2023
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Running Only-Train-Once (OTO) experiment on overparameterized student networks
 #
"""
from context import *

import pathlib
import torch
import argparse
import numpy as np

from only_train_once import OTO

import platforms.registry
import platforms.platform
import datasets.registry
import datasets.teacher_dataset
import datasets.mnist
from controls.control_utils import train_oto, teacher_comparison
from foundations.hparams import DatasetHparams, TrainingHparams
from foundations.step import Step
from models import mnist_lenet, activation_functions, initializers
from training import optimizers, checkpointing
from training.plotting import plot_metrics
from training.standard_callbacks import run_at_step, run_every_100_epochs, save_model, \
    save_logger, create_eval_reg_callback
from utils.utils import set_seeds

# receive input lambda from argparse
parser = argparse.ArgumentParser()
parser.add_argument("--l", "-l", help="lambda", default=1e-4)
parser.add_argument("--t", "-t", help="teacher seed", default='1')
args = parser.parse_args()

lbda = float(args.l)
teacher_seed = args.t

set_seeds(1)

platform = platforms.registry.get('local')(num_workers=0)
platforms.platform._PLATFORM = platform
output_location = pathlib.Path().resolve().joinpath('controls', 'oto_sweep', f"teacher_seed_{teacher_seed}",
                                                    f"lambda_{lbda}")
try: platform.makedirs(output_location)
except: pass

model = mnist_lenet.Model([512], initializers.kaiming_normal, activation_functions.tanh())
model.criterion = torch.nn.MSELoss()

dummy_input = torch.zeros(1, 28, 28)
oto = OTO(model=model, dummy_input=dummy_input)

dataset_hparams = DatasetHparams(dataset_name='teacher', batch_size=512, teacher_name='12645e254b',
                                 teacher_seed=teacher_seed, samples=60000, datagen='mnist', d_in=784)
training_hparams = TrainingHparams(optimizer_name='adam', lr=0.001, training_steps='25000ep', delta=0.31622776601683794,
                                   lr_scheduler="plateau", patience=100, cooldown=2000)
train_loader = datasets.registry.get(dataset_hparams, train=True)

optimizer = oto.dhspg(lr=1e-3, epsilon=0.95, lmbda=lbda, variant='adam', target_group_sparsity=0.99,
                      start_pruning_steps=5000 * len(train_loader),
                      weight_decay=0.0)

lr_schedule = optimizers.get_lr_schedule(training_hparams, optimizer, train_loader.iterations_per_epoch)

start_step = Step.zero(train_loader.iterations_per_epoch)
end_step = Step.from_str(training_hparams.training_steps, train_loader.iterations_per_epoch)

train_eval_callback = create_eval_reg_callback('train', train_loader, verbose=True)
callbacks = [run_at_step(start_step, save_model),
             run_at_step(start_step, train_eval_callback),
             run_at_step(end_step, save_model),
             run_at_step(end_step, save_logger),
             run_every_100_epochs(checkpointing.save_checkpoint_callback),
             run_every_100_epochs(train_eval_callback)]

model.to(device=platform.torch_device)
train_oto(training_hparams, model, train_loader, output_location, callbacks, start_step, end_step, optimizer, platform,
          lr_schedule)

# PLOTTING

plot_metrics(folder_path=output_location, metric_name='train_loss', logscale=True)

teacher_folder = os.path.join(platform.root, "train_" + dataset_hparams.teacher_name,
                              "seed_" + dataset_hparams.teacher_seed, "main")
teacher = datasets.teacher_dataset.Dataset.get_specified_model(teacher_folder)
cluster_mask = datasets.mnist.Dataset.get_boruta_mask()
cluster_mask = np.concatenate([cluster_mask, [1.0]])

symmetry = "odd_constant"
teacher_comparison(teacher, model, symmetry, cluster_mask, output_location)

print("Fine")
