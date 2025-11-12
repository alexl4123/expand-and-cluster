"""
 # Created on 08.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Runner for Expand-and-Cluster experiment
 #
"""
import argparse
import os
import logging

import torch

from dataclasses import dataclass

from cli import shared_args
from foundations.runner import Runner
import models.registry
from platforms.platform import get_platform
from extraction import expand_and_cluster
from extraction.desc import ExtractionDesc
from training.plotting import plot_metrics
from training.wandb_init import wandb_init
from extraction.random_teacher_generation import RandomTeacherGeneration
from foundations.precision import Precision

sep = '='*140
sep2 = '-'*140

@dataclass
class ExtractionRunner(Runner):
    global_seed: int
    desc: ExtractionDesc
    verbose: bool = True
    evaluate_every_epoch: bool = True

    @staticmethod
    def description():
        return "Runs Expand-and-Cluster on the selected model."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        shared_args.JobArgs.add_args(parser)
        ExtractionDesc.add_args(parser, shared_args.maybe_get_default_hparams())

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'TrainingRunner':
        return ExtractionRunner(args.global_seed, ExtractionDesc.create_from_args(args),
                                not args.quiet, not args.evaluate_only_at_end)

    def display_output_location(self):
        print(self.desc.run_path(self.global_seed))


    def get_output_location(self):
        return self.desc.run_path(self.global_seed)

    def run(self):

        precision = Precision.get_precision_from_string(self.desc.training_hparams.precision)
        #torch.set_default_tensor_type(precision)
        torch.set_default_dtype(precision)

        self.desc.save(self.desc.run_path(self.global_seed))
        # Initialize default logger
        logger = logging.getLogger('')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.get_output_location(),"print_logger.log"))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s ->>> %(message)s')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        handler_sh = logging.StreamHandler()
        handler_sh.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(handler_sh)

        if self.verbose and get_platform().is_primary_process:
            logger.info(f"running on: {get_platform().device_str}")
            logger.info(sep + f'\nExtracting model\n' + sep2)
            logger.info(self.desc.display)
            logger.info(f'Output Location: {self.desc.run_path(self.global_seed)}' + '\n' + sep + '\n')
            logger.info(sep2 + f'\nTraining student networks\n' + sep2)

        wandb_successful_login = wandb_init([self.desc.model_hparams,
                                             self.desc.dataset_hparams,
                                             self.desc.training_hparams,
                                             self.desc.extraction_hparams],
                                            run_type='ec',
                                            summary_infos={'local_hashname': self.desc.hashname.split('_')[-1]},
                                            wandb_name = self.desc.dataset_hparams.wandb_name)

        # if self.desc.extraction_hparams.conv_level is bigger than 0:
        # then loads reconstructed weights of previous lvl and train next layers
        if (hasattr(self.desc.extraction_hparams, "conv_level") and self.desc.extraction_hparams.conv_level == 0) or\
            not hasattr(self.desc.extraction_hparams, "conv_level"):
            self.desc.extraction_hparams.conv_level = None

        if self.desc.extraction_hparams.conv_level is not None:
            model = load_conv_bottom_model(self.desc)
            original_steps = self.desc.training_hparams.training_steps
            self.desc.training_hparams.training_steps = '1000ep'
            print(f'Output Location: {self.desc.run_path(self.global_seed)}' + '\n' + sep + '\n')
            students, losses = expand_and_cluster.train_students(
                       model, self.desc.run_path(self.global_seed),
                       self.desc.dataset_hparams, self.desc.training_hparams,
                       evaluate_every_epoch=self.evaluate_every_epoch)
            # remove freeze on the first layer and finetune again
            print("Unfreezing first layer(s)")
            self.desc.training_hparams.training_steps = original_steps
            print(f'Output Location: {self.desc.run_path(self.global_seed)}' + '\n' + sep + '\n')
            for param in model.parameters():
                param.requires_grad = True
            students, losses = expand_and_cluster.train_students(
                       model, self.desc.run_path(self.global_seed),
                       self.desc.dataset_hparams, self.desc.training_hparams,
                       evaluate_every_epoch=self.evaluate_every_epoch)

        # classic training
        else:
            # Setup for detecting number datapoints threshold (rank-nullity continuation)
            teacher_model = None

            if "random_generation" not in str(self.desc.dataset_hparams.teacher_name):
                model = models.registry.get(self.desc.model_hparams)
            else:

                d_out = 1


                teacher_model = RandomTeacherGeneration.random_teacher_generation(
                    self.desc.dataset_hparams.d_in,
                    self.desc.dataset_hparams.teacher_specification,
                    self.desc.model_hparams.act_fun,
                    precision=self.desc.training_hparams.precision,
                    )


                # Student Model
                model = RandomTeacherGeneration.random_teacher_generation(
                    self.desc.dataset_hparams.d_in,
                    self.desc.dataset_hparams.student_specification,
                    self.desc.model_hparams.act_fun,
                    precision=self.desc.training_hparams.precision,
                )

                #model = models.registry.get(self.desc.model_hparams, outputs=d_out, dataset_hparams=self.desc.dataset_hparams)
                
            students, losses = expand_and_cluster.train_students(
                       model, self.desc.run_path(self.global_seed),
                       self.desc.dataset_hparams, self.desc.training_hparams,
                       evaluate_every_epoch=self.evaluate_every_epoch,
                       model_hparams=self.desc.model_hparams,
                       teacher_model = teacher_model)

        plot_metrics(folder_path=self.desc.run_path(self.global_seed), metric_name='train_individual_losses',
                     logscale=True)

        if self.verbose and get_platform().is_primary_process:
            logger.info(sep + f'\nReconstruction\n' + sep2)

        logger.info(f'Reconstruction Location: {self.desc.extraction_path(self.global_seed)}' + '\n' + sep2 + '\n')

        layer = None  # Only needed in convnets

        # if model is a convnet, merge models of different seeds
        if hasattr(students, 'ConvNet'):
            print("Merge convnet seeds")
            students, losses = merge_convnet_seeds(students, self.desc.run_path(self.global_seed),
                                                   self.desc)
            layer = 0 if self.desc.extraction_hparams.conv_level is None else self.desc.extraction_hparams.conv_level

        if "random_generation" not in str(self.desc.dataset_hparams.teacher_name):
            expand_and_cluster.reconstruct(students, losses, self.desc.extraction_path(self.global_seed),
                                        self.desc.extraction_hparams, self.desc.dataset_hparams,
                                        self.desc.training_hparams, self.desc.model_hparams,
                                        verbose=self.verbose, layer=layer)
            

def merge_convnet_seeds(students, path, desc):
    import os
    import torch
    from utils.utils import find_final_model_step
    from foundations import paths
    from training.metric_logger import MetricLogger

    it_per_epoch = desc.dataset_hparams.samples // desc.dataset_hparams.batch_size + 1
    exp_path = os.path.dirname(os.path.dirname(path))
    merged_model = models.registry.get(desc.model_hparams)
    losses = []

    i = 0
    for seed_path in os.listdir(exp_path):
        if not seed_path.startswith('seed'):
            continue
        seed = int(seed_path.split('_')[-1])
        model_path = os.path.join(exp_path, seed_path, "main")
        last_step = find_final_model_step(model_path, it_per_epoch)
        if last_step is None:
            print(f"Skipping seed {seed} because no final model was found.")
            continue
        logger = MetricLogger.create_from_file(model_path)
        current_loss = logger.get_data('train_individual_losses')[-1][1][0]
        if current_loss > 1e-3:
            print(f"Skipping seed {seed} because the final loss was too high: {current_loss}")
            continue

        losses.append(current_loss)
        model_dict = get_platform().load_model(paths.model(model_path, last_step),
                                               map_location=get_platform().torch_device)
        if i == 0:
            for param_name, param in merged_model.named_parameters():
                param.data = model_dict[param_name].unsqueeze(-1)
        else:
            for param_name, param in merged_model.named_parameters():
                param.data = torch.cat((param.data, model_dict[param_name].unsqueeze(-1)), dim=-1)
        i += 1
    return merged_model, losses


def load_conv_bottom_model(desc):
    import os
    import torch
    import numpy as np
    from foundations import paths
    from models import registry as models_registry
    from utils.utils import find_final_model_step

    for l in range(desc.extraction_hparams.conv_level):
        # assumes saved weights are in the same folder as the current run (seed 1)
        run_path = desc.run_path(1)
        w_path = os.path.join(desc.extraction_path(1), f"w_rec_L{l+1}.npy")
        w = torch.tensor(np.load(w_path), device=get_platform().torch_device)

        # overwrite the number of channels in the model name based on the saved reconstructed weights
        # THIS WILL CHANGE THE HASHNAME OF TRAINING! BE CAREFUL
        model_name = desc.model_hparams.model_name.split('_')
        if model_name[3+l] == 'M':
            s = 3+l+1
        else:
            s = 3+l
        model_name[s] = f"C{w.shape[-1]}"
        model_name = '_'.join(model_name)
        model_hparams = desc.model_hparams
        model_hparams.model_name = model_name
        print(f"Model: {model_name}")
        model = models_registry.get(desc.model_hparams)

        # alternative approach: load the best previous model and reinitialise the weights of the layers >= 2*l
        # load the model
        it_per_epoch = desc.dataset_hparams.samples // desc.dataset_hparams.batch_size + 1
        best_seed, loss = find_best_seed(it_per_epoch, os.path.dirname(os.path.dirname(run_path)))
        print(f"Best seed ({best_seed}) loss: {loss}")
        model_path = os.path.normpath(run_path).split(os.path.sep)
        model_path[-2] = f"seed_{best_seed}"
        model_path = os.path.sep + os.path.join(*[s for s in model_path])
        last_step = find_final_model_step(model_path, it_per_epoch)
        model_dict = get_platform().load_model(paths.model(model_path, last_step),
                                               map_location=get_platform().torch_device)

        # load the model weights up to layer 2*l
        for i, key in enumerate(model_dict.keys()):
            if i >= 2*l:
                break
            list(model.parameters())[i].data = model_dict[key]
            list(model.parameters())[i].requires_grad = False

        w, b = w[:-1], w[-1]
        # load the last reconstructed weights into the model
        list(model.parameters())[2*l].data = w.T.reshape(w.shape[-1], -1, 3, 3)
        list(model.parameters())[2*l+1].data = b
        list(model.parameters())[2*l].requires_grad = False
        list(model.parameters())[2*l+1].requires_grad = False

    return model


def find_best_seed(it_per_epoch, path):
    import os
    import numpy as np
    from utils.utils import find_final_model_step
    from training.metric_logger import MetricLogger

    seed_loss = []
    for seed_path in os.listdir(path):
        if not seed_path.startswith('seed'):
            continue
        seed = int(seed_path.split('_')[-1])
        model_path = os.path.join(path, seed_path, "main")
        last_step = find_final_model_step(model_path, it_per_epoch)
        if last_step is None:
            print(f"Skipping seed {seed} because no final model was found.")
            continue
        logger = MetricLogger.create_from_file(model_path)
        current_loss = logger.get_data('train_individual_losses')[-1][1][0]
        seed_loss.append([seed, current_loss])
    seed_loss = np.array(seed_loss)
    best_seed = seed_loss[np.argmin(seed_loss[:, 1])][0]
    return int(best_seed), seed_loss[np.argmin(seed_loss[:, 1])][1]
