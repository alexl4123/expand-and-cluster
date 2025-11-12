import argparse
import os
import logging
import pandas as pd
import numpy as np

from platforms.platform import get_platform
from dataclasses import dataclass

from cli import shared_args
from foundations.runner import Runner
from extraction.desc import ExtractionDesc

import dataclasses

from datetime import datetime

from foundations.hparams import DatasetHparams

import torch
from models.students_mnist_lenet import Model
import models.activation_functions as activation_functions
import models.initializers as initializers
import models.bn_initializers as bn_initializers
import datasets
from foundations.hparams import DatasetHparams

from models import registry as models_registry

sep = '='*140
sep2 = '-'*140

@dataclass
class EvaluateModelRunner(Runner):
    global_seed: int
    desc: ExtractionDesc
    verbose: bool = True
    evaluate_every_epoch: bool = True

    @staticmethod
    def description():
        return "Evaluates model."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        shared_args.JobArgs.add_args(parser)
        ExtractionDesc.add_args(parser, shared_args.maybe_get_default_hparams())

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'TrainingRunner':
        return EvaluateModelRunner(args.global_seed, ExtractionDesc.create_from_args(args),
                                not args.quiet, not args.evaluate_only_at_end)

    def display_output_location(self):
        print(self.desc.run_path(self.global_seed))


    def get_output_location(self):
        return self.desc.run_path(self.global_seed)

    def run(self):

        df = None 
        individual_losses_df = []

        dataset_hparams = self.desc.dataset_hparams
        #folder_path = os.path.join("data","20240510_512_TEACHER")
        #folder_path = os.path.join("data","20240524_512_TEACHER")
        #folder_path = os.path.join("data","20240528_512_TEACHER")
        folder_path = os.path.join("data","20240609_512_2")
        #debug_iter = 0

        for folder in sorted(os.listdir(folder_path)):
            instance_folder_path = os.path.join(folder_path, folder)
            if os.path.isdir(instance_folder_path):
                #instance_folder_name = "20240510-1000-mnist"
                instance_folder_name = folder
                print("")
                print(f"<<<<<NOW DOING: {instance_folder_name}>>>>>>>>>>>>>>")

                instance_path = os.path.join(folder_path, instance_folder_name, "seed_1")



                original_datagen, final_layer_size, training_hidden_layer_size, after_tuning_model_path,\
                    before_tuning_best_average_sim_w, before_tuning_best_max_sim_w, before_tuning_best_average_sim_a, before_tuning_best_max_sim_a,\
                    after_tuning_best_average_sim_w, after_tuning_best_max_sim_w, after_tuning_best_average_sim_a, after_tuning_best_max_sim_a = self.find_model_parameters_and_backup(instance_path)

                after_training_model_path = self.find_model_in_path(instance_path)

                after_tuning_losses, after_tuning_std_devs, _ = self.load_and_eval_model(final_layer_size, after_tuning_model_path,
                                                               original_datagen, dataset_hparams)
                after_training_losses, after_training_std_devs, individual_losses = self.load_and_eval_model(training_hidden_layer_size, after_training_model_path,
                                                                 original_datagen, dataset_hparams, number_students=30)
                
                data_for_row = {
                    'Augmentation-Technique': instance_folder_name,
                    #
                    'TUN-Specific-Data': after_tuning_losses[0],
                    'TUN-MNSIT-Train': after_tuning_losses[1],
                    'TUN-MNIST-Test': after_tuning_losses[2],
                    'TUN-FashionMNIST-Train': after_tuning_losses[3],
                    'TUN-FashionMNIST-Test': after_tuning_losses[4],
                    #
                    'TUN-Specific-Data-Std-Dev': after_tuning_std_devs[0],
                    'TUN-MNSIT-Train-Std-Dev': after_tuning_std_devs[1],
                    'TUN-MNIST-Test-Std-Dev': after_tuning_std_devs[2],
                    'TUN-FashionMNIST-Train-Std-Dev': after_tuning_std_devs[3],
                    'TUN-FashionMNIST-Test-Std-Dev': after_tuning_std_devs[4],
                    #
                    'TRN-Specific-Data': after_training_losses[0],
                    'TRN-MNSIT-Train': after_training_losses[1],
                    'TRN-MNIST-Test': after_training_losses[2],
                    'TRN-FashionMNIST-Train': after_training_losses[3],
                    'TRN-FashionMNIST-Test': after_training_losses[4],
                    #
                    'TRN-Specific-Data-Std-Dev': after_training_std_devs[0],
                    'TRN-MNSIT-Train-Std-Dev': after_training_std_devs[1],
                    'TRN-MNIST-Test-Std-Dev': after_training_std_devs[2],
                    'TRN-FashionMNIST-Train-Std-Dev': after_training_std_devs[3],
                    'TRN-FashionMNIST-Test-Std-Dev': after_training_std_devs[4],
                    #
                    'before_tuning_best_average_sim_w': before_tuning_best_average_sim_w,
                    'before_tuning_best_max_sim_w': before_tuning_best_max_sim_w,
                    'before_tuning_best_average_sim_a': before_tuning_best_average_sim_a,
                    'before_tuning_best_max_sim_a': before_tuning_best_max_sim_a,
                    'after_tuning_best_average_sim_w': after_tuning_best_average_sim_w,
                    'after_tuning_best_max_sim_w': after_tuning_best_max_sim_w,
                    'after_tuning_best_average_sim_a': after_tuning_best_average_sim_a,
                    'after_tuning_best_max_sim_a': after_tuning_best_max_sim_a,
                    }
                
                for individual_loss_index in range(len(individual_losses[0])):
                    data_for_row[f'TRN-Specific-Data-IndividualLoss-{individual_loss_index}'] = individual_losses[0][individual_loss_index]
                    data_for_row[f'TRN-MNIST-Train-IndividualLoss-{individual_loss_index}'] = individual_losses[1][individual_loss_index]
                    data_for_row[f'TRN-MNIST-Test-IndividualLoss-{individual_loss_index}'] = individual_losses[2][individual_loss_index]
                    data_for_row[f'TRN-FashionMNIST-Train-IndividualLoss-{individual_loss_index}'] = individual_losses[3][individual_loss_index]
                    data_for_row[f'TRN-FashionMNIST-Test-IndividualLoss-{individual_loss_index}'] = individual_losses[4][individual_loss_index]

                new_row = pd.DataFrame([data_for_row])

                if df is not None: 
                    df = pd.concat([df, new_row], ignore_index=True)
                else:
                    df = new_row

                #if debug_iter >= 1:
                #    break
                #debug_iter += 1

        now = datetime.now()
        formatted_date_time = now.strftime("%Y-%m-%d-%H-%M")

        df.to_csv(f'{formatted_date_time}-output.csv', index=False)

    def load_and_eval_model(self, final_layer_size, after_tuning_model_path, original_datagen, dataset_hparams, number_students=1):
        print("TRY MODEL LOAD")
        model_hparams_copy = dataclasses.replace(self.desc.model_hparams)
        model_hparams_copy.model_name = f"students_mnist_lenet({number_students})_{final_layer_size}"
        model = models_registry.get(model_hparams_copy)
        temp = torch.load(after_tuning_model_path, map_location=torch.device("cpu"))
        model.load_state_dict(temp)
        print("AFTER LOAD")

        datagens = [
            original_datagen,
            "mnist",
            "mnist_test",
            "fashion_mnist",
            "fashion_mnist_test",
                    ]
        losses = []
        std_devs = []
        individual_losses_arr = []
        for datagen in datagens:
            dataset_hparams_copy = dataclasses.replace(dataset_hparams)
            dataset_hparams_copy.teacher_name = "593ba334b9"
            dataset_hparams_copy.datagen = datagen
            data_loader = datasets.registry.get(dataset_hparams_copy, train=False)

            loss, std_dev, individual_losses = self.evaluate_model(data_loader, model)
            losses.append(loss)
            std_devs.append(std_dev)
            individual_losses_arr.append(individual_losses)

        return losses, std_devs, individual_losses_arr




    def evaluate_model(self, data_loader, model):

        device = get_platform().torch_device

        example_count = torch.tensor(0.0).to(device)
        total_loss = torch.tensor(0.0).to(device)
        individual_losses = torch.zeros(model.N).to(device)

        model.eval()

        with torch.no_grad():
            for examples, labels in data_loader:
                examples = examples.to(device)
                labels = labels.squeeze().to(device)
                output = model(examples)
                #output = output.to(device)
                #model.to(device)

                labels_size = torch.tensor(len(labels))
                labels_size = labels_size.to(device)

                example_count += labels_size
                tmp_loss = model.loss_criterion(output, labels)
                tmp_loss_2 = tmp_loss * labels_size
                total_loss +=  tmp_loss_2
                
                tmp_ind_loss = model.individual_losses(output, labels)
                tmp_ind_loss_2 = tmp_ind_loss * labels_size
                individual_losses +=  tmp_ind_loss_2

        # Share the information if distributed.
        # if get_platform().is_distributed:
        #     torch.distributed.reduce(total_loss, 0, op=torch.distributed.ReduceOp.SUM)
        #     torch.distributed.reduce(example_count, 0, op=torch.distributed.ReduceOp.SUM)

        total_loss = total_loss.cpu().item()
        example_count = example_count.cpu().item()
        individual_losses = individual_losses.cpu()

        eval_name="eval"
        step=0

        standard_dev = np.std(individual_losses.numpy() / example_count)

        if get_platform().is_primary_process:
            print(f"ADD LOG FOR: {eval_name}")
            print(f"{eval_name} for step:{step}: {total_loss / (example_count * model.N)}")
            print(f"{eval_name} number of examples: {example_count}")
            print(f"std-losses: {np.std(individual_losses.numpy() / example_count)}")

            #print('{}_individual_losses'.format(eval_name), step, individual_losses / example_count)
            #print('{}_examples'.format(eval_name), step, example_count)

        return (total_loss / (example_count * model.N)), standard_dev, list(individual_losses.numpy() / example_count)


    def find_model_parameters_and_backup(self, instance_path):

        datagen = ""
        hparams_path = os.path.join(instance_path, "hparams.log")
        with open(hparams_path) as file:
            for line in file:
                line = line.rstrip()
                if "datagen" in line:
                    datagen = (line.split("=>")[1]).strip()
                if "model_name" in line:
                    tmp = (line.split("=>")[1]).strip()
                    training_hidden_layer_size = int(((tmp.split("_"))[-1]).strip())

        final_layer_size = 0
        print_logger_path = os.path.join(instance_path, "print_logger.log")

        before_tuning_best_average_sim_w = None
        before_tuning_best_max_sim_w = None
        before_tuning_best_max_sim_a = None
        before_tuning_best_average_sim_a = None

        after_tuning_best_average_sim_w = None
        after_tuning_best_max_sim_w = None
        after_tuning_best_max_sim_a = None
        after_tuning_best_average_sim_a = None

        is_tuning_parameters = False
        with open(print_logger_path) as file:
            for line in file:
                line = line.rstrip()
                if "Final layer size:" in line:
                    final_layer_size = (line.split("Final layer size:")[1]).strip()

                if "Best average sim w:" in line:
                    if is_tuning_parameters == True:
                        after_tuning_best_average_sim_w = float((line.split("Best average sim w:")[1]).strip())
                    else:
                        before_tuning_best_average_sim_w = float((line.split("Best average sim w:")[1]).strip())

                if "Best max sim w:" in line:
                    if is_tuning_parameters == True:
                        after_tuning_best_max_sim_w = float((line.split("Best max sim w:")[1]).strip())
                    else:
                        before_tuning_best_max_sim_w = float((line.split("Best max sim w:")[1]).strip())

                if "Best average sim a:" in line:
                    if is_tuning_parameters == True:
                        after_tuning_best_average_sim_a = float((line.split("Best average sim a:")[1]).strip())
                    else:
                        before_tuning_best_average_sim_a = float((line.split("Best average sim a:")[1]).strip())

                if "Best max sim a:" in line:
                    if is_tuning_parameters == True:
                        after_tuning_best_max_sim_a = float((line.split("Best max sim a:")[1]).strip())
                    else:
                        before_tuning_best_max_sim_a = float((line.split("Best max sim a:")[1]).strip())
                        # This one is always last in logging, therefore this sets the flag differently:
                        is_tuning_parameters = True

        # Find path to model:
        model_path = None 
        for entry in os.listdir(instance_path):
            cluster_path = os.path.join(instance_path, entry)
            if os.path.isdir(cluster_path) and "clustering" in entry:
                model_path = self.find_model_in_path(cluster_path)

            if model_path is not None:
                break

        return datagen, final_layer_size, training_hidden_layer_size, model_path,\
            before_tuning_best_average_sim_w, before_tuning_best_max_sim_w, before_tuning_best_average_sim_a, before_tuning_best_max_sim_a,\
            after_tuning_best_average_sim_w, after_tuning_best_max_sim_w, after_tuning_best_average_sim_a, after_tuning_best_max_sim_a,

    def find_model_in_path(self, path):
        # Find path to model:
        model_path = None
        for item in os.listdir(path):
            temp_model_path = os.path.join(path, item)
            
            # Check if the item is a file and has a .csv extension
            if os.path.isfile(temp_model_path) and item.endswith('.pth') and "ep0" not in item:
                model_path = temp_model_path
                break

        return model_path
