"""
 # Created on 08.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Description: Expand-and-Cluster experiment script
 #
"""
import copy
import warnings
import logging

import torch
import os.path
import os
import dataclasses
import time
import yaml

import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

import datasets.registry
import models.registry
import matplotlib.pyplot as plt

from datasets.teacher_dataset import Dataset
from foundations import hparams
from foundations import paths
from foundations.step import Step
from models.activation_functions import get_symmetry
from models.base import Model, DataParallel
from platforms.platform import get_platform
from training import optimizers
from training.checkpointing import restore_checkpoint
from training.metric_logger import MetricLogger
from training.plotting import plot_metrics
from training.standard_callbacks import ec_callbacks, ec_linear_callbacks
from training.train import train
from extraction.layer_reconstruction import reconstruct_layer, compare_with_teacher, compare_with_teacher_conv
from datasets.registry import registered_datasets

from extraction.plotting import plot_all_features
from models import initializers, students_mnist_lenet_linear
from training.wandb_init import log_figure_wandb, log_histogram_wandb, log_metric_wandb, sync_wandb
from utils.utils import find_final_model_step

from extraction.special_callbacks import trajectory_callback

try:
    import apex
    NO_APEX = False
except ImportError:
    NO_APEX = True

sep = '='*140
sep2 = '-'*140

logger = logging.getLogger('')

def fill_network_weights(models, students, layer_index, w_rec, b_rec, a_rec, model_hparams):

    splits = (model_hparams.model_name.split("_"))
    layers_specifications = splits[3:]

    layers_specifications[layer_index] = str(w_rec.shape[1])

    new_hparams_model_name = "_".join(splits[:3] + layers_specifications)

    model_hparams.model_name = new_hparams_model_name

    new_students = models.registry.get(model_hparams)

    for l in range(len(students.fc_layers)):

        if l == 0:
            print("LAYER 0 VALUES FOR COMPARISON:")
            print((students.fc_layers[l].fc.data).detach().numpy())



        if l == layer_index:
            # Freeze current layer
            new_students.fc_layers[l].fc.data = torch.tensor(w_rec, requires_grad=False, dtype=torch.float)
            new_students.fc_layers[l].b.data = torch.tensor(b_rec, requires_grad=False, dtype=torch.float)
        elif l == layer_index + 1:
            new_students.fc_layers[l].fc.data = torch.tensor(a_rec, requires_grad=True, dtype=torch.float)
            new_students.fc_layers[l].b.data = students.fc_layers[l].b.data
        elif l < layer_index:
            # Freeze lower layers
            new_students.fc_layers[l].fc.data = torch.tensor(students.fc_layers[l].fc.data, requires_grad = False, dtype=torch.float)
            new_students.fc_layers[l].b.data = torch.tensor(students.fc_layers[l].b.data, requires_grad = False, dtype=torch.float)
        else:
            new_students.fc_layers[l].fc.data = students.fc_layers[l].fc.data
            new_students.fc_layers[l].b.data = students.fc_layers[l].b.data
    
    return new_students, model_hparams

def save_model_for_weights_analysis(elapsed_time, losses, student_index, experiment_name, experiment_data_index, teacher_model, model, all_data, all_labels, name_prefix = "100001"):
    format_precision = '%.6f'
    results = {}
    results["training_time"] = elapsed_time
    results["loss"] = float(losses[student_index])

    results_file_path = ["output", experiment_name, f"{experiment_name}_{name_prefix}{student_index+1:05}_{experiment_data_index}_results.yaml"]
    results_file_path = os.path.join(*results_file_path)

    directory_path = os.path.dirname(results_file_path)
    os.makedirs(directory_path, exist_ok=True)

    with open(results_file_path, 'w') as file:
        yaml.dump(results, file)


    for layer_index in range(len(teacher_model.fc_layers)):

        layer_ = (teacher_model.fc_layers[layer_index].weight.data).clone().detach().numpy()
        bias_ = (teacher_model.fc_layers[layer_index].bias.data).clone().detach().numpy()

        layer_ = np.hstack((layer_, bias_[:,np.newaxis]))

        if len(layer_.shape) > 1:
            header = [f"x{index + 1}" for index in range(layer_.shape[1])]
        else:
            header = [f"x{index + 1}" for index in range(layer_.shape[0])]


        teacher_file_path = ["output", experiment_name, f"{experiment_name}_{name_prefix}{student_index+1:05}_{experiment_data_index}_teacher_{layer_index+1}.csv"]
        teacher_file_path = os.path.join(*teacher_file_path)

        np.savetxt(teacher_file_path, layer_, delimiter=',', fmt=format_precision,  header=",".join(header), comments='')


    layer_ = (teacher_model.fc.weight.data).clone().detach().numpy()
    bias_ = (teacher_model.fc.bias.data).clone().detach().numpy()

    layer_ = np.hstack((layer_, bias_[:,np.newaxis]))

    if len(layer_.shape) > 1:
        header = [f"x{index + 1}" for index in range(layer_.shape[1])]
    else:
        header = [f"x{index + 1}" for index in range(layer_.shape[0])]

    teacher_file_path = ["output", experiment_name, f"{experiment_name}_{name_prefix}{student_index+1:05}_{experiment_data_index}_teacher_{len(teacher_model.fc_layers)+1}.csv"]
    teacher_file_path = os.path.join(*teacher_file_path)

    np.savetxt(teacher_file_path, layer_, delimiter=',', fmt=format_precision,  header=",".join(header), comments='')

    # STUDENT SAVING
    for layer_index in range(len(model.fc_layers)):

        if len((model.fc_layers[layer_index].fc.data).clone().detach().numpy().shape) == 3:
            layer_ = ((model.fc_layers[layer_index].fc.data).clone().detach().numpy())[:,:,student_index].T
            bias_ = ((model.fc_layers[layer_index].b.data).clone().detach().numpy())[:,student_index]
        else:
            layer_ = ((model.fc_layers[layer_index].fc.data).clone().detach().numpy()).T
            bias_ = ((model.fc_layers[layer_index].b.data).clone().detach().numpy())

        layer_ = np.hstack((layer_, bias_[:,np.newaxis]))

        student_file_path = ["output", experiment_name, f"{experiment_name}_{name_prefix}{student_index+1:05}_{experiment_data_index}_student_{layer_index+1}.csv"]
        student_file_path = os.path.join(*student_file_path)

        if len(layer_.shape) > 1:
            header = [f"x{index + 1}" for index in range(layer_.shape[1])]
        else:
            header = [f"x{index + 1}" for index in range(layer_.shape[0])]

        np.savetxt(student_file_path, layer_, delimiter=',', fmt=format_precision,  header=",".join(header), comments='')


    data_file_path = ["output", experiment_name, f"{experiment_name}_{name_prefix}{student_index+1:05}_{experiment_data_index}_data.csv"]
    data_target_file_path = ["output", experiment_name, f"{experiment_name}_{name_prefix}{student_index+1:05}_{experiment_data_index}_dataTarget.csv"]

    data_file_path = os.path.join(*data_file_path)
    data_target_file_path = os.path.join(*data_target_file_path)

    # flatten data
    if len(all_data.shape) > 2:
        all_data = all_data.reshape(all_data.shape[0], -1)
    # Assuming rows are data (otw. need to do more stuff)
    header = [f"x{index + 1}" for index in range(all_data.shape[0])]
    np.savetxt(data_file_path, all_data.T, delimiter=',', fmt=format_precision,  header=",".join(header), comments='')

    header = [f"x{index + 1}" for index in range(all_labels.shape[0])]
    np.savetxt(data_target_file_path, all_labels.T, delimiter=',', fmt=format_precision,  header=",".join(header), comments='')



def train_students(
    model: Model,
    output_location: str,
    dataset_hparams: hparams.DatasetHparams,
    training_hparams: hparams.TrainingHparams,
    start_step: Step = None,
    verbose: bool = True,
    evaluate_every_epoch: bool = True,
    model_hparams: hparams.ModelHparams = None,
    teacher_model = None):

    """Train using the students through the standard_train procedure."""

    # If the model file for the end of training already exists in this location, do not train.
    iterations_per_epoch = datasets.registry.iterations_per_epoch(dataset_hparams)
    train_end_step = Step.from_str(training_hparams.training_steps, iterations_per_epoch)


    if "random_generation" not in str(dataset_hparams.teacher_name):
        train_loader = datasets.registry.get(dataset_hparams, train=True, output_location = f"{output_location}/train_", train_hparams=training_hparams)
    else:
        print("----")
        print(teacher_model)
        train_loader = datasets.registry.get(dataset_hparams, train=True, output_location = f"{output_location}/train_", model = teacher_model, train_hparams=training_hparams)

    if teacher_model is None:
        if train_loader.dataset._teacher is not None:
            teacher_model = train_loader.dataset._teacher
   
    eval_loaders = []
    if model_hparams is not None and not model_hparams.model_name.startswith("students_fully_connected_specified"):
        if dataset_hparams.original_dataset_name != None:
            dataset_hparams_copy = dataclasses.replace(dataset_hparams)
            dataset_hparams_copy.datagen = dataset_hparams_copy.original_dataset_name

            eval_loaders.append((dataset_hparams.original_dataset_name, datasets.registry.get(dataset_hparams_copy, train=True)))

        dataset_hparams_copy = dataclasses.replace(dataset_hparams)
        dataset_hparams_copy.datagen = "fashion_mnist"
        eval_loaders.append(("fashion_mnist_train", datasets.registry.get(dataset_hparams_copy, train=True)))

        dataset_hparams_copy = dataclasses.replace(dataset_hparams)
        dataset_hparams_copy.datagen = "fashion_mnist_test"
        eval_loaders.append(("fashion_mnist_test", datasets.registry.get(dataset_hparams_copy, train=False)))

        dataset_hparams_copy = dataclasses.replace(dataset_hparams)
        dataset_hparams_copy.datagen = "mnist_test"
        eval_loaders.append(("mnist_test", datasets.registry.get(dataset_hparams_copy, train=False)))

    if (models.registry.exists(output_location, train_end_step) and
    get_platform().exists(paths.logger(output_location))) and training_hparams.further_training is None:
        device = get_platform().torch_device
        state_dict = get_platform().load_model(paths.model(output_location, train_end_step),
                                               map_location=device)
        model.load_state_dict(state_dict)
        losses = 0
        for examples, labels in train_loader:
            losses += model.individual_losses(model(examples), labels) / len(labels)
        return model, losses.detach().cpu().numpy()


    if "random_generation" not in str(dataset_hparams.teacher_name):
        callbacks = ec_callbacks(
                                training_hparams, 
                                train_loader, 
                                verbose=verbose, 
                                start_step=start_step,
                                evaluate_every_1000_steps=evaluate_every_epoch, 
                                eval_loader=eval_loaders,
                                evaluate_every_100_epochs=evaluate_every_epoch,
                                evaluate_every_10=hasattr(model, 'ConvNet'))  # If ConvNet, evaluate every 10 epochs
        
        callbacks += trajectory_callback()
    else:
        callbacks = []

    start_time = time.time()
    train(training_hparams, model, train_loader, output_location, callbacks, start_step=start_step)
    end_time = time.time()
    model.cpu()

    all_data = None
    all_labels = None

    losses = 0
    for examples, labels in train_loader:
        losses += model.individual_losses(model(examples), labels)/len(labels)

        if all_data is None:
            all_data = examples.clone().detach().numpy()
        else:
            all_data = np.vstack((all_data, examples.clone().detach().numpy()))

        if all_labels is None:
            all_labels = labels.clone().detach().numpy()
        else:
            all_labels = np.vstack((all_labels, labels.clone().detach().numpy()))






    elapsed_time = end_time - start_time
    experiment_name = dataset_hparams.experiment_name
    if experiment_name is None:
        experiment_name = dataset_hparams.wandb_name
    experiment_data_index = dataset_hparams.experiment_data_index
    if experiment_data_index is None:
        experiment_data_index = 1

    number_students = model.N

    for student_index in range(number_students):
        save_model_for_weights_analysis(elapsed_time, losses, student_index, experiment_name, experiment_data_index, teacher_model, model, all_data, all_labels)

    if "random_generation" in str(dataset_hparams.teacher_name):
        logger.info("---- SMALL TEACHER TRAINING FINISHED - EXITING ----")
        exit()



    return model, losses.detach().cpu().numpy()


def reconstruct(model: Model,
                losses: torch.Tensor,
                output_location: str,
                extraction_hparams: hparams.ExtractionHparams,
                dataset_hparams: hparams.DatasetHparams,
                training_hparams: hparams.TrainingHparams,
                model_hparams: hparams.ModelHparams,
                verbose: bool = True,
                layer: int = None):
    
    logger = logging.getLogger('')

    """Expand-and-Cluster procedure, for the moment is only for fully connected"""
    plots_folder = os.path.join(output_location, "ECplots")
    os.makedirs(plots_folder, exist_ok=True)
    finetune_checkpoints_folder = os.path.join(output_location, "finetune_checkpoints")
    os.makedirs(finetune_checkpoints_folder, exist_ok=True)
    reconstructed_folder = os.path.join(output_location, "reconstructed_model")
    os.makedirs(reconstructed_folder, exist_ok=True)
    alignment_reconstruction =  os.path.join(reconstructed_folder, "after_reconstruction")
    os.makedirs(alignment_reconstruction, exist_ok=True)
    alignment_tuning = os.path.join(reconstructed_folder, "after_tuning")
    os.makedirs(alignment_tuning, exist_ok=True)
    beta = np.pi/extraction_hparams.beta
    train_loader = datasets.registry.get(dataset_hparams, train=True, output_location = f"{output_location}/reconstruct_")

    eval_loaders = []
    if not model_hparams.model_name.startswith("students_fully_connected_specified"):
        if dataset_hparams.original_dataset_name != None:
            dataset_hparams_copy = dataclasses.replace(dataset_hparams)
            dataset_hparams_copy.datagen = dataset_hparams_copy.original_dataset_name

            eval_loaders.append((dataset_hparams.original_dataset_name, datasets.registry.get(dataset_hparams_copy, train=True)))

        dataset_hparams_copy = dataclasses.replace(dataset_hparams)
        dataset_hparams_copy.datagen = "fashion_mnist"
        eval_loaders.append(("fashion_mnist_train", datasets.registry.get(dataset_hparams_copy, train=True)))

        dataset_hparams_copy = dataclasses.replace(dataset_hparams)
        dataset_hparams_copy.datagen = "fashion_mnist_test"
        eval_loaders.append(("fashion_mnist_test", datasets.registry.get(dataset_hparams_copy, train=False)))

        dataset_hparams_copy = dataclasses.replace(dataset_hparams)
        dataset_hparams_copy.datagen = "mnist_test"
        eval_loaders.append(("mnist_test", datasets.registry.get(dataset_hparams_copy, train=False)))

    teacher_folder = os.path.join(get_platform().root, "train_" + dataset_hparams.teacher_name,
                                "seed_" + dataset_hparams.teacher_seed, "main")
    teacher = Dataset.get_specified_model(teacher_folder)
    # Iterate through the different layers of model
    parameter_list = copy.deepcopy(list(model.parameters()))
    N = parameter_list[0].shape[-1]

    layer_no = int(len(parameter_list)/2)
    final_layer = False

    students = models.registry.get(model_hparams)
    symmetry = get_symmetry(students.act_fun)

    # losses = 0
    # for examples, labels in train_loader:
    #     losses += students.individual_losses(students(examples), labels) / len(labels)

    # if symmetry == "even_linear_positive_scaling" or symmetry == "even_linear":
    #     students = students_mnist_lenet_linear.Model.load_from_student_mnist_lenet(students)

    # load boruta mask (assumes we know the dataset the teacher was trained on)
    cluster_mask = None
    if extraction_hparams.boruta is not None:
        dataset_class = registered_datasets[extraction_hparams.boruta].Dataset
        cluster_mask = dataset_class.get_boruta_mask()
        cluster_mask = np.concatenate([cluster_mask, [1.0]])

    for l in range(layer_no):  # TODO: avoid re-clustering when it is already saved
        logger.info(sep2 + '\n' + f'Layer {l}' + '\n' + sep2 + '\n')
        if l+1 == layer_no-1:  # if last hidden layer, set final_layer to True
            final_layer = True

        i = l*2  # parameter_list index
        w, b, a = parameter_list[i].data, parameter_list[i+1].data,  parameter_list[i+2].data
        w, b, a = [copy.deepcopy(x.cpu().numpy()) for x in [w, b, a]]

        if hasattr(students, 'ConvNet'):
            # compact 3D kernels into 1D vectors
            # w = w.reshape([w.shape[0], -1, w.shape[-1]]).transpose(1, 0, 2)
            # a = a.reshape([a.shape[0], -1, a.shape[-1]]).transpose(1, 0, 2)
            w = w.transpose(0, 4, 1, 2, 3)  # DIMS: [out_channels, in_channels, kernel_size, kernel_size, #networks]
            w = w.reshape([w.shape[0], w.shape[1], -1]).transpose(2, 0, 1)
            a = None

        # concatenate the biases to the weights
        w_cat = np.concatenate([w, b[np.newaxis, :, :]], axis=0)

        # move weight norms to a if symmetry is even_linear_positive_scaling
        if symmetry == 'even_linear_positive_scaling':
            w_norms = np.linalg.norm(w_cat[:, :, :], axis=0)
            w_cat /= w_norms
            a = np.einsum("hon,hn->hon", a, w_norms)

        # compute cosine similarity between weight vectors of w_cat
        # n = 3  # network index
        # sim = np.zeros((w_cat.shape[1], w_cat.shape[1]))
        # for j in range(w_cat.shape[1]):
        #     for i in range(w_cat.shape[1]):
        #         sim[j, i] = np.dot(w_cat[:-1, j, n], w_cat[:-1, i, n]) / \
        #                     (np.linalg.norm(w_cat[:-1, j, n]) * np.linalg.norm(w_cat[:-1, i, n]))
        # fig, ax = plt.subplots(); plt.imshow(sim, cmap='bwr'); plt.colorbar(); fig.show()

        w_rec, a_rec = reconstruct_layer(w_cat, N, extraction_hparams.gamma, beta, losses, A=a,
                                         symmetry=symmetry, verbose=verbose, cluster_mask=cluster_mask,
                                         plots_folder=plots_folder, exp_name=f"L{l+1}", final_layer=final_layer)

        # save w_rec in the extraction folder
        if hasattr(students, 'ConvNet'):
            if l == 1:  # Manually fix second layer permutation of conv (something scrambled in the process)
                perm = [0, 1, 2, 9, 11, 12, 13, 14, 15, 3, 4, 5, 6, 10, 8, 7]
                perm = np.argsort(perm)
                w_perm = w_rec[:-1,:].T.reshape(16, 16, 3, 3)
                w_perm = w_perm[:, perm, :, :].reshape(16, -1).T
                b_perm = w_rec[-1]
                b_perm = b_perm[perm]
                w_rec = np.concatenate([w_perm, b_perm[np.newaxis, :]])

            np.save(os.path.join(output_location, f"w_rec_L{l+1}"), w_rec)
            best_permutation = teacher_comparison_conv(w_rec, teacher, l, symmetry, plots_folder, verbose=verbose,
                                                       permutation= best_permutation if l > 0 else None)
            np.save(os.path.join(output_location, f"best_permutation_L{l + 1}"), best_permutation)
            if l == layer:
                return
            else:
                continue

        # put the new clustered layer across the N networks
        w_rec = np.stack([w_rec]*N, axis=2)
        w_rec, b_rec = w_rec[:-1, :, :], w_rec[-1, :, :]
        if len(a_rec.shape) == 2 and l+1 < layer_no-1:
            a_rec = np.stack([a_rec]*N, axis=2)

        #students = fill_current_layer(students, i, w_rec, b_rec, a_rec, symmetry)
        new_students, new_model_hparams = fill_network_weights(models, students, l, w_rec, b_rec, a_rec, model_hparams)
        students = new_students
        model_hparams = new_model_hparams

        if l+1 == layer_no-1:  # if last hidden layer, break
            break

        # Retrain the upper layers of the network
        start_step = Step.zero(train_loader.iterations_per_epoch)
        finetune_hparams = copy.deepcopy(training_hparams)
        finetune_hparams.training_steps = extraction_hparams.finetune_training_steps
        finetune_hparams.lr = extraction_hparams.finetune_lr
        callbacks = ec_callbacks(finetune_hparams, train_loader, verbose=verbose, start_step=start_step,
                                 evaluate_every_1000_steps=True, eval_loader=eval_loaders)
        os.makedirs(os.path.join(finetune_checkpoints_folder, f"L{l + 1}"), exist_ok=True)



        train(finetune_hparams, students, train_loader,
              os.path.join(finetune_checkpoints_folder, f"L{l + 1}"),
              callbacks, start_step=start_step)

        plot_metrics(folder_path=os.path.join(finetune_checkpoints_folder, f"L{l + 1}"),
                     metric_name='train_individual_losses', logscale=True)

        if l==0: cluster_mask = None  # remove cluster_mask after first layer

        # Account for new layer sizes
        parameter_list = copy.deepcopy(list(students.parameters()))

        logger.info("--- AFTER TRAINING LAYER SHAPE AND ALIGNMENT ---")
        for l in range(len(students.fc_layers)):
            logger.info(students.fc_layers[l].fc.data.shape)
        logger.info(layer_no)
        logger.info("--- END AFTER TRAINING LAYER SHAPE AND ALIGNMENT ---")




        # TODO: add filtering of neurons with non-aligned biases (how do we deal with the last layer? there's no OP
        #  after that)

    # FINAL FINE-TUNING
    # condense in one network and make all parameters trainable
    for i, param in enumerate(students.parameters()):
        if i % 2 == 0 and i != 2*(layer_no-1):  # weight params except the output layer
            param.data = torch.mean(param.data, dim=2).unsqueeze(2)  # this average is not doing anything (note the above np.stack)
        elif i == 2*(layer_no-1):  # output layer
            param.data = param.data.unsqueeze(2)
        elif i % 2 == 1:  # bias params
            param.data = torch.mean(param.data, dim=1).unsqueeze(1)
        param.requires_grad = True

    students.N = 1
    students.to(get_platform().torch_device)

    if symmetry == 'even_linear_positive_scaling' or symmetry == 'even_linear':
        frozen_students = copy.deepcopy(students)
        sample_no = train_loader.dataset._labels.shape[0]
        with torch.no_grad():
            y_frozen = []
            list_examples = []
            list_labels = []
            for examples, labels in train_loader:
                examples = examples.to(device=get_platform().torch_device)
                labels = labels.to(device=get_platform().torch_device)
                y_frozen.append(frozen_students(examples))
                list_labels.append(labels)
                list_examples.append(examples)
            y_frozen = torch.cat(y_frozen, dim=0).squeeze()
            y = (torch.vstack(list_labels) - y_frozen).cpu().numpy()
            examples = torch.vstack(list_examples).cpu().numpy()
        x = np.concatenate([examples.reshape(sample_no, -1), np.ones([sample_no, 1])], axis=1)
        thetas = np.linalg.lstsq(x, y, rcond=None)[0]
        logger.info(f"MSE of linear component after reconstruction {(((x @ thetas) - y)**2).mean()}")


    # PRINTING END SHAPE:
    logger.info("--- FINAL SHAPE ---")
    for l in range(len(students.fc_layers)):
        logger.info(students.fc_layers[l].fc.data.shape)
    logger.info(layer_no)
    deep_comparison_and_compression(teacher, students, symmetry, cluster_mask, alignment_reconstruction, layer_no, models, model_hparams)
    logger.info("--- END FINAL SHAPE ---")

    if layer_no <= 2:
        teacher_comparison(teacher, students, symmetry, cluster_mask, alignment_reconstruction)
    else: # Deep comparison
        students, distances = deep_comparison_and_compression(teacher, students, symmetry, cluster_mask, alignment_reconstruction, layer_no, models, model_hparams)

    start_step = Step.zero(train_loader.iterations_per_epoch)
    finetune_hparams = copy.deepcopy(training_hparams)
    finetune_hparams.training_steps = extraction_hparams.finetune_training_steps
    finetune_hparams.lr = extraction_hparams.finetune_lr

    if symmetry == 'even_linear_positive_scaling' or symmetry == 'even_linear':
        print("IN LINEAR/EVEN-LINEAR STUFF")
        callbacks = ec_linear_callbacks(finetune_hparams, train_loader, verbose=verbose, start_step=start_step,
                                        evaluate_every_1000_steps=True, eval_loader=eval_loaders)
        callbacks += trajectory_callback()
        loss = parallel_train(finetune_hparams, students, train_loader, reconstructed_folder, callbacks,
                    thetas, start_step=start_step)
        # TODO: Implement proper linear parallel model and avoid specifying a different train function.
        # TODO: what about the align_final_bias?
    else:
        print("DEFAULT STUFF")
        callbacks = ec_callbacks(finetune_hparams, train_loader, verbose=verbose, start_step=start_step,
                                evaluate_every_1000_steps=True, eval_loader=eval_loaders)

        callbacks += trajectory_callback()
        align_final_bias(students, train_loader, verbose=verbose)
        train(finetune_hparams, students, train_loader, reconstructed_folder, callbacks, start_step=start_step)

        for it, (examples, labels) in enumerate(train_loader):
            loss = students.loss_criterion(students(examples), labels)
            break


    teacher_comparison(teacher, students, symmetry, cluster_mask, alignment_tuning)
    deep_comparison_and_compression(teacher, students, symmetry, cluster_mask, alignment_reconstruction, layer_no, models, model_hparams)

    plot_metrics(folder_path=reconstructed_folder,
                metric_name='train_individual_losses', logscale=True)


    elapsed_time = 0
    student_index = 0
    experiment_name = dataset_hparams.experiment_name
    if experiment_name is None:
        experiment_name = dataset_hparams.wandb_name
    experiment_data_index = dataset_hparams.experiment_data_index
    if experiment_data_index is None:
        experiment_data_index = 1
    if loss is None:
        loss = 0

    all_data = None
    all_labels = None

    losses = 0
    for examples, labels in train_loader:
        losses += model.individual_losses(model(examples), labels)/len(labels)

        if all_data is None:
            all_data = examples.clone().detach().numpy()
        else:
            all_data = np.vstack((all_data, examples.clone().detach().numpy()))

        if all_labels is None:
            all_labels = labels.clone().detach().numpy()
        else:
            all_labels = np.vstack((all_labels, labels.clone().detach().numpy()))


    save_model_for_weights_analysis(elapsed_time, losses, student_index, experiment_name, 0, teacher, students, all_data, all_labels, name_prefix="200002")



def align_final_bias(model, loader, verbose=False):
    """
    Aligns the final bias of the model to the dataset bias.
    :param model:
    :param trainloader:
    :return:
    """
    example_count = torch.tensor(0.0).to(get_platform().torch_device)
    total_loss = torch.tensor(0.0).to(get_platform().torch_device)
    total_output = 0.0
    dataset_bias = 0.0

    model.eval()
    with torch.no_grad():
        for examples, labels in loader:
            examples = examples.to(get_platform().torch_device)
            labels = labels.squeeze().to(get_platform().torch_device)
            output = model(examples)

            labels_size = torch.tensor(len(labels), device=get_platform().torch_device)
            example_count += labels_size
            total_loss += model.loss_criterion(output, labels) * labels_size
            total_output += output.sum(dim=0)
            dataset_bias += labels.sum(dim=0)

    example_count = example_count.cpu().item()
    total_loss = total_loss.cpu().item() / example_count
    total_output /= example_count
    dataset_bias /= example_count

    if verbose: logger.info(f"Loss before bias alignment: {total_loss:.3e}")

    if len(dataset_bias.shape) > 0:
        list(model.parameters())[-1].data += -total_output + dataset_bias[:, None]
    else:
        list(model.parameters())[-1].data += -total_output + dataset_bias

    total_loss = torch.tensor(0.0).to(get_platform().torch_device)
    example_count = torch.tensor(0.0).to(get_platform().torch_device)
    with torch.no_grad():
        for examples, labels in loader:
            examples = examples.to(get_platform().torch_device)
            labels = labels.squeeze().to(get_platform().torch_device)
            output = model(examples)

            labels_size = torch.tensor(len(labels), device=get_platform().torch_device)
            example_count += labels_size
            total_loss += model.loss_criterion(output, labels) * labels_size

    total_loss = total_loss.cpu().item() / example_count
    if verbose: logger.info(f"Loss after bias alignment: {total_loss:.3e}")


def fill_current_layer(students, i, w_rec, b_rec, a_rec, symmetry):
    param_list = list(students.parameters())
    param_list[i].data = torch.Tensor(w_rec)
    param_list[i+1].data = torch.Tensor(b_rec)
    param_list[i+2].data = torch.Tensor(a_rec)
    return student

def deep_comparison_and_compression(teacher, students, symmetry, cluster_mask, alignment_reconstruction, layer_no, models, model_hparams, create_new_student = False):

    distances = []

    for layer_index in range(layer_no - 1):

        print(f"-------- LAYER {layer_index} ------")

        parameter_list = copy.deepcopy(list(students.parameters()))

        #i = layer_index*2  # parameter_list index
        #w, b, a = parameter_list[i].data, parameter_list[i+1].data,  parameter_list[i+2].data
        #w, b, a = [copy.deepcopy(x.cpu().numpy()) for x in [w, b, a]]

        student_weights = (students.fc_layers[layer_index].fc.data).cpu().numpy()
        student_bias = ((students.fc_layers[layer_index].b.data).cpu().numpy())
        student_a = (students.fc_layers[layer_index + 1].fc.data).cpu().numpy()


        #student_weights = ((students.fc_layers[layer_index].fc.data).cpu().numpy())[:,:,0]
        #student_bias = ((students.fc_layers[layer_index].b.data).cpu().numpy())[:,:,0]
        #student_a = ((students.fc_layers[layer_index + 1].fc.data).cpu().numpy())[:,:,0]

        # concatenate the biases to the weights
        #w_cat = np.concatenate([w, b[np.newaxis, :, :]], axis=0)

        logger.info("--- STUDENT SHAPES ---")
        logger.info(student_weights.shape)
        logger.info(student_bias.shape)
        logger.info(student_a.shape)

        student_weights_cat = np.concatenate([student_weights, student_bias[np.newaxis, :]], axis=0)

        logger.info(student_weights_cat.shape)

        teacher_weights = ((teacher.fc_layers[layer_index].weight.data).cpu().numpy()).T
        teacher_biases = (teacher.fc_layers[layer_index].bias.data).cpu().numpy()
        if layer_index < layer_no - 2:
            teacher_a = ((teacher.fc_layers[layer_index + 1].weight.data).cpu().numpy()).T
        else:
            teacher_a = ((teacher.fc.weight.data).cpu().numpy()).T

        logger.info("--- TEACHER SHAPES ---")
        logger.info(teacher_weights.shape)
        logger.info(teacher_biases.shape)
        logger.info(teacher_a.shape)

        teacher_cat = np.concatenate([teacher_weights, teacher_biases[np.newaxis,:]], axis=0)

        logger.info(teacher_cat.shape)

        closest_student_indices_list = []
        closest_student_distances = []

        new_student_weights_cat = np.zeros(tuple(list(teacher_cat.shape) + [1]))
        new_student_a = np.zeros((teacher_a.shape[0],student_a.shape[1],1))


        for teacher_neuron_index in range(teacher_cat.shape[1]):

            teacher_neuron = teacher_cat[:, teacher_neuron_index]

            closest_distance = None
            closest_index = None

            for student_neuron_index in range(student_weights_cat.shape[1]):


                student_neuron = student_weights_cat[:, student_neuron_index, 0]

                # Remove bias for deep identifiability:
                tmp_teacher_neuron = np.sort(teacher_neuron[:-1])[::-1]
                tmp_student_neuron = np.sort(student_neuron[:-1])[::-1]

                cos_sim = 1 - np.dot(tmp_teacher_neuron, tmp_student_neuron)/(np.linalg.norm(tmp_teacher_neuron)*np.linalg.norm(tmp_student_neuron))

                if closest_distance is None:
                    closest_distance = cos_sim
                    closest_index = student_neuron_index
                elif cos_sim < closest_distance:
                    closest_distance = cos_sim
                    closest_index = student_neuron_index

            closest_student_indices_list.append(closest_index)
            closest_student_distances.append(closest_distance)

            new_student_weights_cat[:,teacher_neuron_index,0] = student_weights_cat[:, closest_index,0]
            new_student_a[teacher_neuron_index,:,0] = student_a[closest_index,:,0]

        print(new_student_weights_cat.shape)
        print(new_student_a.shape)

        w_rec, b_rec = new_student_weights_cat[:-1, :, :], new_student_weights_cat[-1, :, :]

        new_students, new_model_hparams = fill_network_weights(models, students, layer_index, w_rec, b_rec, new_student_a, model_hparams)
        students = new_students
        model_hparams = new_model_hparams

        distances.append(closest_student_distances)

    all_dists = []
    for layer_distances_index in range(len(distances)):

        layer_distances = distances[layer_distances_index]

        avg_ = sum(layer_distances) / len(layer_distances)
        max_ = max(layer_distances)

        all_dists += layer_distances

        logger.info(f"For layer {layer_distances_index} the avg. and max distances are:")
        logger.info(f"AVG: {avg_}") 
        logger.info(f"MAX: {max_}")
        logger.info("")

    logger.info("Overall avg and max distances are:")
    logger.info(f"AVG: {sum(all_dists)/len(all_dists)}")
    logger.info(f"MAX: {max(all_dists)}")
    logger.info("")

    return students, distances


def teacher_comparison(teacher, students, symmetry, cluster_mask, plots_folder):
    teacher_params = copy.deepcopy(list(teacher.parameters()))
    students_params = copy.deepcopy(list(students.parameters()))

    wt, bt, at = [teacher_params[i].data.cpu().numpy().squeeze().T for i in range(3)]
    ws, bs, as_ = [students_params[i].data.cpu().numpy().squeeze() for i in range(3)]

    out = compare_with_teacher(wt, bt, at, ws, bs, as_, symmetry, cluster_mask=cluster_mask, verbose=True)
    fig, best_sims_w, best_sims_a, (student_size, teacher_size) = out
    log_histogram_wandb(np.log10(best_sims_w)+1e-12, "clustering/student_alignment_W", "log cosine distance")
    log_histogram_wandb(np.log10(best_sims_a)+1e-16, "clustering/student_alignment_A", "log cosine distance")
    log_metric_wandb("clustering/student_size", student_size)
    log_metric_wandb("clustering/teacher_size", teacher_size)
    sync_wandb()

    fig.savefig(os.path.join(plots_folder, f"student_alignments.pdf"))
    out = compare_with_teacher(wt, bt, at, ws, bs, as_, symmetry, cluster_mask=cluster_mask, log=False)
    fig, best_sims_w, best_sims_a, (student_size, teacher_size) = out

    fig.savefig(os.path.join(plots_folder, f"student_alignments_nolog.pdf"))
    plt.close(fig)


def teacher_comparison_conv(w_rec, teacher, l, symmetry, plots_folder, permutation=None, verbose=False):
    wt = copy.deepcopy(list(teacher.parameters())[l*2].data.cpu().numpy().squeeze())
    inverse_permutation = np.argsort(permutation) if permutation is not None else None
    if inverse_permutation is not None:
        wt = wt[:, inverse_permutation, :, :]  # Align input channels to previous best permutation
    wt = wt.reshape(wt.shape[0], -1).T

    bt = copy.deepcopy(list(teacher.parameters())[l*2+1].data.cpu().numpy().squeeze())
    ws = copy.deepcopy(w_rec)
    ws, bs = ws[:-1], ws[-1]

    out = compare_with_teacher_conv(wt, bt, ws, bs, symmetry, verbose=True)
    fig, best_sims_w, best_permutation, (student_size, teacher_size) = out
    log_histogram_wandb(np.log10(best_sims_w)+1e-12, "clustering/student_alignment_W", "log cosine distance")
    log_metric_wandb("clustering/student_size", student_size)
    log_metric_wandb("clustering/teacher_size", teacher_size)
    sync_wandb()
    fig.savefig(os.path.join(plots_folder, f"student_alignments_nolog_L{l+1}.pdf"))
    plt.close(fig)

    out = compare_with_teacher_conv(wt, bt, ws, bs, symmetry, verbose=False, log=True)
    fig, _, _, (_, _) = out
    fig.savefig(os.path.join(plots_folder, f"student_alignments_L{l+1}.pdf"))
    plt.close(fig)

    return best_permutation


def parallel_train(
    training_hparams: hparams.TrainingHparams,
    model: Model,
    train_loader,
    output_location: str,
    callbacks,
    thetas,
    start_step: Step = None,
    end_step: Step = None,
):

    """The main training loop modified to allow for parallel linear path.

    Args:
      * training_hparams: The training hyperparameters whose schema is specified in hparams.py.
      * model: The model to train. Must be a models.base.Model
      * train_loader: The training data. Must be a datasets.base.DataLoader
      * output_location: The string path where all outputs should be stored.
      * callbacks: A list of functions that are called before each training step and once more
        after the last training step. Each function takes five arguments: the current step,
        the output location, the model, the optimizer, and the logger.
        Callbacks are used for running the test set, saving the logger, saving the state of the
        model, etc. They provide hooks into the training loop for customization so that the
        training loop itself can remain simple.
      * start_step: The step at which the training data and learning rate schedule should begin.
        Defaults to step 0.
      * end_step: The step at which training should cease. Otherwise, training will go for the
        full `training_hparams.training_steps` steps.
    """

    # Create the output location if it doesn't already exist.
    if not get_platform().exists(output_location) and get_platform().is_primary_process:
        get_platform().makedirs(output_location)

    # Get the optimizer and learning rate schedule.
    model.to(get_platform().torch_device)
    thetas = torch.nn.Parameter(torch.Tensor(thetas).to(get_platform().torch_device))
    optimizer = torch.optim.Adam(list(model.parameters()) + [thetas], lr=training_hparams.lr*10)
    step_optimizer = optimizer
    lr_schedule = optimizers.get_lr_schedule(training_hparams, optimizer, train_loader.iterations_per_epoch)

    # Adapt for FP16.
    if training_hparams.apex_fp16:
        if NO_APEX: raise ImportError('Must install nvidia apex to use this model.')
        model, step_optimizer = apex.amp.initialize(model, optimizer, loss_scale='dynamic', verbosity=0)

    # Handle parallelism if applicable.
    # if get_platform().is_distributed:
    #     model = DistributedDataParallel(model, device_ids=[get_platform().rank])
    elif get_platform().is_parallel:
        model = DataParallel(model)

    # Get the random seed for the data order.
    data_order_seed = training_hparams.data_order_seed

    # Restore the model from a saved checkpoint if the checkpoint exists.
    cp_step, cp_logger = restore_checkpoint(output_location, model, optimizer, train_loader.iterations_per_epoch)
    start_step = cp_step or start_step or Step.zero(train_loader.iterations_per_epoch)
    logger = cp_logger or MetricLogger()

    if not isinstance(lr_schedule, ReduceLROnPlateau):
        with warnings.catch_warnings():  # Filter unnecessary warning.
            # warnings.filterwarnings("ignore", category=UserWarning)
            for _ in range(start_step.iteration): lr_schedule.step()

    # Determine when to end training.
    end_step = end_step or Step.from_str(training_hparams.training_steps, train_loader.iterations_per_epoch)
    if end_step <= start_step: return

    # if training was stopped prematurely (lr too low) then skip training.
    maybe_train_end_step = find_final_model_step(output_location, train_loader.iterations_per_epoch)
    if maybe_train_end_step is not None:  # load saved model before returning
        state_dict = get_platform().load_model(paths.model(output_location, maybe_train_end_step),
                                               map_location=get_platform().torch_device)
        model = model.load_state_dict(state_dict)
        return

    # The training loop.
    for ep in range(start_step.ep, end_step.ep + 1):

        # Ensure the data order is different for each epoch.
        train_loader.shuffle(None if data_order_seed is None else (data_order_seed + ep))
        loss_ep = 0

        for it, (examples, labels) in enumerate(train_loader):

            # Advance the data loader until the start epoch and iteration.
            if ep == start_step.ep and it < start_step.it: continue

            # Run the callbacks.
            step = Step.from_epoch(ep, it, train_loader.iterations_per_epoch)
            for callback in callbacks: callback(output_location, step, [model, thetas], optimizer, logger)
            # for callback in callbacks: callback(output_location, step, model, optimizer, logger)

            # Exit at the end step.
            if ep == end_step.ep and it == end_step.it: return

            # Otherwise, train.
            examples = examples.to(device=get_platform().torch_device)
            labels = labels.to(device=get_platform().torch_device)

            x = torch.cat([examples.reshape(examples.shape[0], -1),
                           torch.ones(examples.shape[0], 1).to(get_platform().torch_device)], dim=1)

            step_optimizer.zero_grad()
            model.train()
            loss = model.loss_criterion(model(examples) + (x@thetas).unsqueeze(-1), labels)
            if training_hparams.apex_fp16:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            loss_ep += loss.item()

            # Step forward. Ignore extraneous warnings that the lr_schedule generates.
            step_optimizer.step()
            if not isinstance(lr_schedule, ReduceLROnPlateau):
                with warnings.catch_warnings():  # Filter unnecessary warning.
                    warnings.filterwarnings("ignore", category=UserWarning)
                    lr_schedule.step()

        loss_ep /= train_loader.iterations_per_epoch
        if isinstance(lr_schedule, ReduceLROnPlateau):
            if step_optimizer.param_groups[0]["lr"] < lr_schedule.min_lrs[0]+lr_schedule.eps:  # End if lr is minimal
                current_step = Step.from_epoch(ep, it, train_loader.iterations_per_epoch)
                logger.save(output_location)
                model.save(output_location, current_step)
                break
            with warnings.catch_warnings():  # Filter unnecessary warning.
                warnings.filterwarnings("ignore", category=UserWarning)
                lr_schedule.step(loss_ep)


    get_platform().barrier()
    return loss
