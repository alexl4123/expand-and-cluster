"""
 # Created on 08.09.23
 #
 # Author: Flavio Martinelli, EPFL.
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Defines all callbacks
 #
"""
import os
import time
import torch
import logging

from datasets.base import DataLoader
from foundations import hparams
from foundations.step import Step
from platforms.platform import get_platform
from training import checkpointing
from training.wandb_init import log_metric_wandb, sync_wandb, get_wandb_prefix

print_logger = logging.getLogger('')

# Standard callbacks.
def save_model(output_location, step, model, optimizer, logger):
    model.save(output_location, step)


def save_model_linear(output_location, step, model, optimizer, logger):
    model[0].save(output_location, step)
    torch.save(model[1], os.path.join(output_location, "affine.pth"))


def save_logger(output_location, step, model, optimizer, logger):
    logger.save(output_location)


def sync_wandb_callback(output_location, step, model, optimizer, logger):
    sync_wandb()


def create_timekeeper_callback():
    time_of_last_call = None

    def callback(output_location, step, model, optimizer, logger):
        if get_platform().is_primary_process:
            print_logger = logging.getLogger('')
            nonlocal time_of_last_call
            t = 0.0 if time_of_last_call is None else time.time() - time_of_last_call
            print_logger.info(f'Ep {step.ep}\tIt {step.it}\tTime Elapsed {t:.2f}')
            log_metric_wandb('time_elapsed', t)
            time_of_last_call = time.time()
        get_platform().barrier()

    return callback


def create_eval_callback(eval_name: str, loader: DataLoader, verbose=False):
    """This function returns a callback."""

    time_of_last_call = None

    def eval_callback(output_location, step, model, optimizer, logger):
        example_count = torch.tensor(0.0).to(get_platform().torch_device)
        total_loss = torch.tensor(0.0).to(get_platform().torch_device)
        total_correct = torch.tensor(0.0).to(get_platform().torch_device)

        def correct(labels, outputs):
            return torch.sum(torch.eq(labels, output.argmax(dim=1)))

        model.eval()

        with torch.no_grad():
            for examples, labels in loader:
                examples = examples.to(get_platform().torch_device)
                labels = labels.squeeze().to(get_platform().torch_device)
                output = model(examples)

                labels_size = torch.tensor(len(labels), device=get_platform().torch_device)
                example_count += labels_size
                total_loss += model.loss_criterion(output, labels) * labels_size
                total_correct += correct(labels, output)

        # Share the information if distributed.
        if get_platform().is_distributed:
            torch.distributed.reduce(total_loss, 0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(total_correct, 0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(example_count, 0, op=torch.distributed.ReduceOp.SUM)

        total_loss = total_loss.cpu().item()
        total_correct = total_correct.cpu().item()
        example_count = example_count.cpu().item()

        if get_platform().is_primary_process:
            logger.add('{}_loss'.format(eval_name), step, total_loss / example_count)
            logger.add('{}_accuracy'.format(eval_name), step, total_correct / example_count)
            logger.add('{}_examples'.format(eval_name), step, example_count)

            if verbose:
                print_logger = logging.getLogger('')
                nonlocal time_of_last_call
                elapsed = 0 if time_of_last_call is None else time.time() - time_of_last_call
                print_logger.info('{}\tep {:03d}\tit {:03d}\tloss {:.3f}\tacc {:.2f}%\tex {:d}\ttime {:.2f}s'.format(
                    eval_name, step.ep, step.it, total_loss/example_count, 100 * total_correct/example_count,
                    int(example_count), elapsed))
                time_of_last_call = time.time()

        wandb_prefix = get_wandb_prefix(output_location)
        log_metric_wandb(f'{wandb_prefix}/{eval_name}_loss', total_loss / example_count)
        log_metric_wandb(f'{wandb_prefix}/{eval_name}_accuracy', total_correct / example_count)
        log_metric_wandb(f'{wandb_prefix}/learning_rate', optimizer.param_groups[0]['lr'])
        log_metric_wandb(f'{wandb_prefix}/examples', example_count)
        log_metric_wandb(f'{wandb_prefix}/iteration_step', step.iteration)

    return eval_callback


def create_eval_reg_callback(eval_name: str, loader: DataLoader, verbose=False):
    """Same as above but does not compute accuracy, for regression the loss is enough."""

    time_of_last_call = None

    def eval_callback(output_location, step, model, optimizer, logger):
        example_count = torch.tensor(0.0).to(get_platform().torch_device)
        total_loss = torch.tensor(0.0).to(get_platform().torch_device)

        model.eval()

        with torch.no_grad():
            for examples, labels in loader:
                examples = examples.to(get_platform().torch_device)
                labels = labels.squeeze().to(get_platform().torch_device)
                output = model(examples)

                labels_size = torch.tensor(len(labels), device=get_platform().torch_device)
                example_count += labels_size
                total_loss += model.loss_criterion(output, labels) * labels_size

        # Share the information if distributed.
        if get_platform().is_distributed:
            torch.distributed.reduce(total_loss, 0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(example_count, 0, op=torch.distributed.ReduceOp.SUM)

        total_loss = total_loss.cpu().item()
        example_count = example_count.cpu().item()

        if get_platform().is_primary_process:
            logger.add('{}_loss'.format(eval_name), step, total_loss / example_count)
            logger.add('{}_examples'.format(eval_name), step, example_count)

            if verbose:
                print_logger = logging.getLogger('')
                nonlocal time_of_last_call
                elapsed = 0 if time_of_last_call is None else time.time() - time_of_last_call
                print_logger.info('{}\tep {:03d}\tit {:03d}\tloss {:.2e}\tex {:d}\ttime {:.2f}s'.format(
                    eval_name, step.ep, step.it, total_loss/example_count,
                    int(example_count), elapsed))
                time_of_last_call = time.time()

        wandb_prefix = get_wandb_prefix(output_location)
        log_metric_wandb(f'{wandb_prefix}/{eval_name}_loss', total_loss / example_count)
        log_metric_wandb(f'{wandb_prefix}/learning_rate', optimizer.param_groups[0]['lr'], step)
        log_metric_wandb(f'{wandb_prefix}/examples', example_count)
        log_metric_wandb(f'{wandb_prefix}/iteration_step', step.iteration)

    return eval_callback


def create_ec_eval_callback(eval_name: str, loader: DataLoader, verbose=False, other_prefix: str = ""):
    """This function returns a callback for the students training experiments."""

    time_of_last_call = None

    def eval_callback(output_location, step, model, optimizer, logger):

        device = get_platform().torch_device

        # TODO: hack to handle the fact that the model is a tuple only sometimes.
        model, thetas = model if type(model) is list else (model, None)

        example_count = torch.tensor(0.0).to(device)
        total_loss = torch.tensor(0.0).to(device)
        individual_losses = torch.zeros(model.N).to(device)

        model.eval()

        with torch.no_grad():
            for examples, labels in loader:
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

        if get_platform().is_primary_process:
            print(f"ADD LOG FOR: {eval_name}")
            logger.add('{}_loss'.format(eval_name), step, total_loss / (example_count * model.N))
            logger.add('{}_individual_losses'.format(eval_name), step, individual_losses / example_count)
            logger.add('{}_examples'.format(eval_name), step, example_count)

            if verbose:
                print_logger = logging.getLogger('')
                nonlocal time_of_last_call
                elapsed = 0 if time_of_last_call is None else time.time() - time_of_last_call
                print_logger.info('{}-{}\tep {:05d}\tit {:03d}\tloss {:.3e}\tex {:d}\ttime {:.2f}s'.format(
                    eval_name, other_prefix, step.ep, step.it, total_loss/example_count, int(example_count), elapsed))
                time_of_last_call = time.time()

        wandb_prefix = get_wandb_prefix(output_location)
        log_metric_wandb(f'{wandb_prefix}{other_prefix}/{eval_name}_loss', total_loss / (example_count * model.N))
        log_metric_wandb(f'{wandb_prefix}{other_prefix}/{eval_name}_individual_losses', individual_losses / example_count)
        log_metric_wandb(f'{wandb_prefix}{other_prefix}/learning_rate', optimizer.param_groups[0]['lr'])
        log_metric_wandb(f'{wandb_prefix}{other_prefix}/examples', example_count)
        log_metric_wandb(f'{wandb_prefix}{other_prefix}/iteration_step', step.iteration)

    return eval_callback


def create_ec_linear_eval_callback(eval_name: str, loader: DataLoader, verbose=False, other_prefix: str = ""):
    """This function returns a callback for the students training experiments."""

    time_of_last_call = None

    def eval_callback_3(output_location, step, model, optimizer, logger):


        device = get_platform().torch_device
        model, thetas = model
        example_count = torch.tensor(0.0).to(get_platform().torch_device)
        total_loss = torch.tensor(0.0).to(get_platform().torch_device)
        individual_losses = torch.zeros(model.N).to(get_platform().torch_device)

        model.eval()
        #model = model.to(device)

        with torch.no_grad():
            for examples, labels in loader:
                examples = examples.to(device)
                labels = labels.squeeze().to(device)
                output = model(examples)

                labels_size = torch.tensor(len(labels), device=device)
                example_count += labels_size
                x = torch.cat([examples.reshape(examples.shape[0], -1),
                               torch.ones(examples.shape[0], 1).to(device)], dim=1)
                total_loss += model.loss_criterion(output + (x@thetas).unsqueeze(-1), labels) * labels_size
                individual_losses += model.individual_losses(output + (x@thetas).unsqueeze(-1), labels) * labels_size

        # Share the information if distributed.
        # if get_platform().is_distributed:
        #     torch.distributed.reduce(total_loss, 0, op=torch.distributed.ReduceOp.SUM)
        #     torch.distributed.reduce(example_count, 0, op=torch.distributed.ReduceOp.SUM)

        total_loss = total_loss.cpu().item()
        example_count = example_count.cpu().item()
        individual_losses = individual_losses.cpu()

        if get_platform().is_primary_process:
            logger.add('{}_loss'.format(eval_name), step, total_loss / (example_count * model.N))
            logger.add('{}_individual_losses'.format(eval_name), step, individual_losses / example_count)
            logger.add('{}_examples'.format(eval_name), step, example_count)

            if verbose:
                print_logger = logging.getLogger('')
                nonlocal time_of_last_call
                elapsed = 0 if time_of_last_call is None else time.time() - time_of_last_call
                print_logger.info('{}\tep {:05d}\tit {:03d}\tloss {:.3e}\tex {:d}\ttime {:.2f}s'.format(
                    eval_name, step.ep, step.it, total_loss/example_count, int(example_count), elapsed))
                time_of_last_call = time.time()

        wandb_prefix = get_wandb_prefix(output_location)
        log_metric_wandb(f'{wandb_prefix}{other_prefix}/{eval_name}_loss', total_loss / (example_count * model.N))
        log_metric_wandb(f'{wandb_prefix}{other_prefix}/{eval_name}_individual_losses', individual_losses / example_count)
        log_metric_wandb(f'{wandb_prefix}{other_prefix}/learning_rate', optimizer.param_groups[0]['lr'])
        log_metric_wandb(f'{wandb_prefix}{other_prefix}/examples', example_count)
        log_metric_wandb(f'{wandb_prefix}{other_prefix}/iteration_step', step.iteration)



    return eval_callback_3


# Callback frequencies. Each takes a callback as an argument and returns a new callback
# that runs only at the specified frequency.
def run_every_epoch(callback):
    def modified_callback(output_location, step, model, optimizer, logger):
        if step.it != 0:
            return
        callback(output_location, step, model, optimizer, logger)
    return modified_callback


def run_every_100_epochs(callback):
    def modified_callback(output_location, step, model, optimizer, logger):
        if step.it != 0 or step.ep % 100 != 0:
            return
        callback(output_location, step, model, optimizer, logger)
    return modified_callback

def run_every_1000_steps(callback):
    def modified_callback(output_location, step, model, optimizer, logger):
        if step.iteration % 5000 == 0:
            return callback(output_location, step, model, optimizer, logger)
        else:
            return
    return modified_callback

def run_every_100_steps(callback):
    def modified_callback(output_location, step, model, optimizer, logger):
        if step.iteration % 100 == 0:
            return callback(output_location, step, model, optimizer, logger)
        else:
            return
    return modified_callback


def run_every_10_epochs(callback):
    def modified_callback(output_location, step, model, optimizer, logger):
        if step.it != 0 or step.ep % 10 != 0:
            return
        callback(output_location, step, model, optimizer, logger)
    return modified_callback

def run_every_25_epochs(callback):
    def modified_callback(output_location, step, model, optimizer, logger):
        if step.it != 0 or step.ep % 25 != 0:
            return
        callback(output_location, step, model, optimizer, logger)
    return modified_callback



def run_every_step(callback):
    return callback


def run_at_step(step1, callback):
    def modified_callback(output_location, step, model, optimizer, logger):
        if step != step1:
            return
        callback(output_location, step, model, optimizer, logger)
    return modified_callback


# The standard set of callbacks that should be used for a normal training run.
def standard_callbacks(training_hparams: hparams.TrainingHparams, train_set_loader: DataLoader,
                       test_set_loader: DataLoader, eval_on_train: bool = False, verbose: bool = True,
                       start_step: Step = None, evaluate_every_1000_steps: bool = True):
    start = start_step or Step.zero(train_set_loader.iterations_per_epoch)
    end = Step.from_str(training_hparams.training_steps, train_set_loader.iterations_per_epoch)
    test_eval_callback = create_eval_callback('test', test_set_loader, verbose=verbose)
    train_eval_callback = create_eval_callback('train', train_set_loader, verbose=verbose)

    run_every_X = run_every_25_epochs

    # Basic checkpointing and state saving at the beginning and end.
    result = [
        run_at_step(start, save_model),
        run_at_step(end, save_model),
        run_at_step(end, save_logger),
        run_at_step(end, sync_wandb_callback),
        run_every_X(checkpointing.save_checkpoint_callback),
        run_every_X(sync_wandb_callback)
    ]

    # Test every epoch if requested.
    if evaluate_every_1000_steps: result = [run_every_X(test_eval_callback)] + result
    elif verbose: result.append(run_every_X(create_timekeeper_callback()))

    # Ensure that testing occurs at least at the beginning and end of training.
    if start.it != 0 or not evaluate_every_1000_steps: result = [run_at_step(start, test_eval_callback)] + result
    if end.it != 0 or not evaluate_every_1000_steps: result = [run_at_step(end, test_eval_callback)] + result

    # Do the same for the train set if requested.
    if eval_on_train:
        if evaluate_every_1000_steps: result = [run_every_X(train_eval_callback)] + result
        if start.it != 0 or not evaluate_every_1000_steps: result = [run_at_step(start, train_eval_callback)] + result
        if end.it != 0 or not evaluate_every_1000_steps: result = [run_at_step(end, train_eval_callback)] + result

    return result


def ec_callbacks(training_hparams: hparams.TrainingHparams, train_set_loader: DataLoader,
                 verbose: bool = True, start_step: Step = None, evaluate_every_1000_steps: bool = True,
                 evaluate_every_100_epochs: bool = True,
                 eval_loader = None, evaluate_every_10: bool = False):
    start = start_step or Step.zero(train_set_loader.iterations_per_epoch)
    end = Step.from_str(training_hparams.training_steps, train_set_loader.iterations_per_epoch)
    train_eval_callback = create_ec_eval_callback('train', train_set_loader, verbose=verbose)

    run_every = run_every_10_epochs if evaluate_every_10 is True else run_every_1000_steps

    # Basic checkpointing and state saving at the beginning and end.
    result = [
        run_at_step(start, save_model),
        run_at_step(end, save_model),
        run_at_step(end, save_logger),
        run_at_step(end, sync_wandb_callback),
        run_every(checkpointing.save_checkpoint_callback),
        run_every(sync_wandb_callback)
    ]

    # Compute loss every 100 epoch if requested.
    if evaluate_every_1000_steps: result = [run_every(train_eval_callback)] + result
    elif verbose: result.append(run_every(create_timekeeper_callback()))

    if eval_loader is not None and len(eval_loader) > 0:
        for eval_loader_individual in eval_loader:
            eval_name = f"eval-{eval_loader_individual[0]}"
            train_eval_eval_callback = create_ec_eval_callback(eval_name, eval_loader_individual[1],
                                                               verbose=verbose,
                                                               other_prefix=eval_name)
            result = [run_every(train_eval_eval_callback)] + result

    # Ensure that training loss computation occurs at least at the beginning and end of training.
    if start.it != 0 or not evaluate_every_1000_steps: result = [run_at_step(start, train_eval_callback)] + result
    if end.it != 0 or not evaluate_every_1000_steps: result = [run_at_step(end, train_eval_callback)] + result

    return result


def ec_linear_callbacks(training_hparams: hparams.TrainingHparams, train_set_loader: DataLoader,
                        verbose: bool = True, start_step: Step = None, evaluate_every_1000_steps: bool = True,
                        eval_loader = None, evaluate_every_10: bool = False):
    start = start_step or Step.zero(train_set_loader.iterations_per_epoch)
    end = Step.from_str(training_hparams.training_steps, train_set_loader.iterations_per_epoch)
    train_eval_callback = create_ec_linear_eval_callback('train', train_set_loader, verbose=verbose)

    run_every = run_every_10_epochs if evaluate_every_10 is True else run_every_1000_steps
    # Basic checkpointing and state saving at the beginning and end.
    result = [
        run_at_step(start, save_model_linear),
        run_at_step(end, save_model_linear),
        run_at_step(end, save_logger),
        run_at_step(end, sync_wandb_callback)
    ]

    # Compute loss every 100 epoch if requested.
    if evaluate_every_1000_steps: result = [run_every(train_eval_callback)] + result
    elif verbose: result.append(run_every(create_timekeeper_callback()))

    if eval_loader is not None and len(eval_loader) > 0:
        for eval_loader_individual in eval_loader:
            eval_name = f"eval-{eval_loader_individual[0]}"
            train_eval_eval_callback = create_ec_eval_callback(eval_name, eval_loader_individual[1],
                                                               verbose=verbose,
                                                               other_prefix=eval_name)
            result = [run_every(train_eval_eval_callback)] + result


    # Ensure that training loss computation occurs at least at the beginning and end of training.
    if start.it != 0 or not evaluate_every_1000_steps: result = [run_at_step(start, train_eval_callback)] + result
    if end.it != 0 or not evaluate_every_1000_steps: result = [run_at_step(end, train_eval_callback)] + result

    return result
