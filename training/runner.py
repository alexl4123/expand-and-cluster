"""
 # Created on 08.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Runner for training experiment
 #
"""

import argparse
import logging
import os

from dataclasses import dataclass

from cli import shared_args
from foundations.runner import Runner
import models.registry
from platforms.platform import get_platform
from training import train
from training.desc import TrainingDesc
from training.plotting import plot_metrics
from training.wandb_init import wandb_init
from foundations.step import Step


@dataclass
class TrainingRunner(Runner):
    global_seed: int
    desc: TrainingDesc
    verbose: bool = True
    evaluate_every_epoch: bool = True

    @staticmethod
    def description():
        return "Train a model."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        shared_args.JobArgs.add_args(parser)
        TrainingDesc.add_args(parser, shared_args.maybe_get_default_hparams())

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'TrainingRunner':
        return TrainingRunner(args.global_seed, TrainingDesc.create_from_args(args),
                              not args.quiet, not args.evaluate_only_at_end)

    def display_output_location(self):
        print(self.desc.run_path(self.global_seed))
        
    def get_output_location(self):
        return self.desc.run_path(self.global_seed)

    def run(self):
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
            logger.info('='*82 + f'\nTraining a Model (Seed {self.global_seed})\n' + '-'*82)
            logger.info(self.desc.display)
            logger.info(f'Output Location: {self.desc.run_path(self.global_seed)}' + '\n' + '='*82 + '\n')

        wandb_successful_login = wandb_init([self.desc.model_hparams,
                                             self.desc.dataset_hparams,
                                             self.desc.training_hparams],
                                            run_type='train',
                                            summary_infos={'local_hashname': self.desc.hashname.split('_')[-1]},
                                            wandb_name = self.desc.dataset_hparams.wandb_name)
        model = models.registry.get(self.desc.model_hparams)
        output_location = self.desc.run_path(self.global_seed)

        if self.desc.model_hparams.model_name.startswith("fully_connected_specified"):
            current_step = Step.from_epoch(0, 0, 512)
            model.save(output_location, current_step)
            return

        
        train.standard_train(
            model, output_location,
            self.desc.dataset_hparams, self.desc.training_hparams, evaluate_every_epoch=self.evaluate_every_epoch)

        plot_metrics(folder_path=self.desc.run_path(self.global_seed), metric_name='test_accuracy', vlims=[0.9, 1.0])
        plot_metrics(folder_path=self.desc.run_path(self.global_seed), metric_name='train_accuracy', vlims=[0.9, 1.0])
        plot_metrics(folder_path=self.desc.run_path(self.global_seed), metric_name='test_loss', logscale=True)
        plot_metrics(folder_path=self.desc.run_path(self.global_seed), metric_name='train_loss', logscale=True)

