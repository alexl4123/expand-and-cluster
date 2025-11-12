import os
import time
import torch
import logging

import numpy as np

from datasets.base import DataLoader
from foundations import hparams
from foundations.step import Step
from platforms.platform import get_platform
from training import checkpointing
from training.wandb_init import log_metric_wandb, sync_wandb, get_wandb_prefix

from training.standard_callbacks import create_ec_eval_callback, run_every_10_epochs, \
    run_every_1000_steps, run_at_step, save_model, save_logger, sync_wandb_callback, run_every_100_epochs

def trajectory_callback():

    result = [] 

    # result.append(run_every_100_epochs(add_to_trajectory)) # breaks on GPU


    return result


# Standard callbacks.
def add_to_trajectory(output_location, step, model, optimizer, logger):

    #print(output_location)
    
    # TODO: support conv networks
    if not hasattr(model, 'fc_layers'):
        return 
    
    for student_index in range(model.fc_layers[0].fc.shape[2]): # Iterate through students
        for neuron_index in range(model.fc_layers[0].fc.shape[1]):
            student_weights = model.fc_layers[0].fc[:,neuron_index,student_index]
            student_biases = model.fc_layers[0].b[neuron_index,student_index]

            params = list(student_weights.detach().numpy())
            params.append(float(student_biases.detach().numpy()))

            student_csv_path = os.path.join(os.path.normpath(output_location), f"student_{student_index}_neuron_{neuron_index}.csv")
            if os.path.isfile(student_csv_path):
                student_data = np.genfromtxt(student_csv_path, delimiter=',', skip_header=0)
                new_row = np.array(params)
                updated_data = np.vstack([student_data, new_row])
            else:
                updated_data = np.array(params)

            np.savetxt(student_csv_path, updated_data, delimiter=',')

    for neuron_index in range(model.fc_layers[1].fc.shape[0]):
        student_weights = model.fc_layers[1].fc[neuron_index,:,student_index]

        params = list(student_weights.detach().numpy())

        student_csv_path = os.path.join(os.path.normpath(output_location), f"student_{student_index}_neuron_{neuron_index}_layer_1.csv")
        if os.path.isfile(student_csv_path):
            student_data = np.genfromtxt(student_csv_path, delimiter=',', skip_header=0)
            if len(student_data.shape) == 0:
                student_data = [student_data]
            new_row = np.array(params)
            updated_data = np.concatenate([student_data, new_row])
        else:
            updated_data = np.array(params)

        np.savetxt(student_csv_path, updated_data, delimiter=',')





