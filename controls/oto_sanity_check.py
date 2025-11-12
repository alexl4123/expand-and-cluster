"""
 # Created on 05.12.2023
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Running Only-Train-Once (OTO) experiment on overparameterized student networks. Sanity check to see
 # if OTO with pruning turned off learns as the original EC experiments.
 #
"""
import pathlib

import torch
from only_train_once import OTO

from context import *

import platforms.registry
import platforms.platform
import datasets.registry
from controls.control_utils import train_oto
from foundations.hparams import DatasetHparams, TrainingHparams
from foundations.step import Step
from models import mnist_lenet, activation_functions, initializers
from training import optimizers, checkpointing
from training.plotting import plot_metrics
from training.standard_callbacks import run_at_step, run_every_100_epochs, save_model, \
    save_logger, create_eval_reg_callback
from utils.utils import set_seeds

set_seeds(1)

platform = platforms.registry.get('local')(num_workers=0)
platforms.platform._PLATFORM = platform
print(f"training on {platform.torch_device}")
output_location = pathlib.Path().resolve().joinpath('controls', 'oto_sanity_check')
try: platform.makedirs(output_location)
except: pass

model = mnist_lenet.Model([40], initializers.kaiming_normal, activation_functions.relu())
model.criterion = torch.nn.MSELoss()

dummy_input = torch.zeros(1, 28, 28)
oto = OTO(model=model, dummy_input=dummy_input)

dataset_hparams = DatasetHparams(dataset_name='teacher', batch_size=512, teacher_name='f32850c8f7',
                                 teacher_seed='1', samples=60000, datagen='mnist', d_in=784)
training_hparams = TrainingHparams(optimizer_name='adam', lr=0.001, training_steps='25000ep', delta=0.31622776601683794,
                                   lr_scheduler="plateau", patience=100, cooldown=2000)
train_loader = datasets.registry.get(dataset_hparams, train=True)

optimizer = oto.dhspg(lr=1e-3, lmbda=0.0, lmbda_amplify=0.0, hat_lmbda_coeff=0.0, epsilon=0.0, weight_decay=0.0,
                      weight_decay_type='l2_norm', first_momentum=None, second_momentum=None, variant='adam',
                      target_group_sparsity=0.0, tolerance_group_sparsity=0.0, partition_step=None,
                      start_pruning_steps=1e50, half_space_project_steps=None, warm_up_steps=0, dampening=None,
                      group_divisible=1, fixed_zero_groups=True)

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

plot_metrics(folder_path=output_location, metric_name='train_loss', logscale=True)


print("Fine")
