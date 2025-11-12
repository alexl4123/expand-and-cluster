"""
 # Created on 06.12.2023
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: functions for control exps
 #
"""
import os
import warnings

import numpy as np
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

from extraction.plotting import plot_all_features
from foundations import paths
from foundations.step import Step
from training.checkpointing import restore_checkpoint
from training.metric_logger import MetricLogger
from utils.utils import find_final_model_step


def train_oto(training_hparams, model, train_loader, output_location, callbacks, start_step, end_step, optimizer,
              platform, lr_schedule):
    step_optimizer = optimizer
    # Restore the model from a saved checkpoint if the checkpoint exists.
    cp_step, cp_logger = restore_checkpoint(output_location, model, optimizer, train_loader.iterations_per_epoch)
    start_step = cp_step or start_step or Step.zero(train_loader.iterations_per_epoch)
    logger = cp_logger or MetricLogger()

    # Get the random seed for the data order.
    data_order_seed = training_hparams.data_order_seed

    # Determine when to end training.
    end_step = end_step or Step.from_str(training_hparams.training_steps, train_loader.iterations_per_epoch)
    if end_step <= start_step: print("network already trained, aborting..."); return

    # if training was stopped prematurely (lr too low) then skip training.
    maybe_train_end_step = find_final_model_step(output_location, train_loader.iterations_per_epoch)
    if maybe_train_end_step is not None:  # load saved model before returning
        state_dict = platform.load_model(paths.model(output_location, maybe_train_end_step),
                                         map_location=platform.torch_device)
        model = model.load_state_dict(state_dict)
        print("network already trained, aborting...")
        return

    model.to(device=platform.torch_device)
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
            for callback in callbacks: callback(output_location, step, model, optimizer, logger)

            # Exit at the end step.
            if ep == end_step.ep and it == end_step.it: return

            # Otherwise, train.
            examples = examples.to(device=platform.torch_device)
            labels = labels.to(device=platform.torch_device)

            step_optimizer.zero_grad()
            model.train()
            loss = model.loss_criterion(model(examples), labels)
            loss.backward()

            loss_ep += loss.item()

            # Step forward. Ignore extraneous warnings that the lr_schedule generates.
            # print(f"it: {it}, ep: {ep}, loss: {loss.item()}")
            step_optimizer.step()
            if not isinstance(lr_schedule, ReduceLROnPlateau):
                with warnings.catch_warnings():  # Filter unnecessary warning.
                    warnings.filterwarnings("ignore", category=UserWarning)
                    lr_schedule.step()

        loss_ep /= train_loader.iterations_per_epoch
        if isinstance(lr_schedule, ReduceLROnPlateau):
            if step_optimizer.param_groups[0]["lr"] < lr_schedule.min_lrs[0] + lr_schedule.eps:  # End if lr is minimal
                current_step = Step.from_epoch(ep, it, train_loader.iterations_per_epoch)
                model.save(output_location, current_step)
                logger.save(output_location)
                break
            with warnings.catch_warnings():  # Filter unnecessary warning.
                warnings.filterwarnings("ignore", category=UserWarning)
                lr_schedule.step(loss_ep)

        if ep % 100 == 0:
            group_sparsity, omega = step_optimizer.compute_group_sparsity_omega()
            try:  # somehow in the gpu it throws an .cpu() error when the optimizer is not yet sparsifying...
                group_norm_gp, group_norm_gnp, _, _ = step_optimizer.compute_norm_group_partitions()
            except Exception as e:
                group_norm_gp, group_norm_gnp = 0, 0
                print(e)
            print(f"group_sparsity: {group_sparsity}, group_norm_gp: {group_norm_gp}, group_norm_gnp: {group_norm_gnp},"
                  f" omega: {omega}")


def teacher_comparison(teacher, students, symmetry, cluster_mask, plots_folder):
    teacher_params = list(teacher.parameters())
    students_params = list(students.parameters())

    wt, bt, at = [teacher_params[i].data.cpu().numpy().squeeze().T for i in range(3)]
    ws, bs, as_ = [students_params[i].data.cpu().numpy().squeeze() for i in range(3)]

    fig, ws_matched, ws_missing = compare_with_teacher(wt, bt, at, ws.T, bs, as_.T, symmetry, cluster_mask=cluster_mask)
    fig.savefig(os.path.join(plots_folder, f"student_alignments.pdf"))
    plt.close(fig)

    # viz features

    if symmetry == "even_linear_positive_scaling":
        wt_matched = wt / np.linalg.norm(wt, axis=0, ord=2)
    else:
        wt_matched = wt
    ax, fig = plot_all_features(wt_matched[:, :, np.newaxis])
    fig.suptitle("teacher features")
    fig.savefig(os.path.join(plots_folder, f"teacher_features.pdf"))
    plt.close(fig)

    ws_matched = ws_matched.T[:-1, :]
    if symmetry == "even_linear_positive_scaling":
        ws_matched /= np.linalg.norm(ws_matched, axis=0, ord=2)
    ax, fig = plot_all_features(ws_matched[:, :, np.newaxis])
    fig.suptitle("student features matching teacher")
    fig.savefig(os.path.join(plots_folder, f"student_features.pdf"))
    plt.close(fig)

    ws_missing = ws_missing.T[:-1, :]
    ws_missing = ws_missing[:, np.linalg.norm(ws_missing, axis=0) != 0]
    if symmetry == "even_linear_positive_scaling":
        ws_missing /= np.linalg.norm(ws_missing, axis=0, ord=2)
    ax, fig = plot_all_features(ws_missing[:, :, np.newaxis])
    fig.suptitle("student features extra")
    fig.savefig(os.path.join(plots_folder, f"student_features_extra.pdf"))
    plt.close(fig)

    fig, _, _ = compare_with_teacher(wt, bt, at, ws.T, bs, as_.T, symmetry, cluster_mask=cluster_mask, log=False)
    fig.savefig(os.path.join(plots_folder, f"student_alignments_nolog.pdf"))
    plt.close(fig)



def compare_with_teacher(wt, bt, at, ws, bs, as_, symmetry, cluster_mask=1, log=True):
    if symmetry == 'odd' or symmetry == 'odd_constant':
        dist_fun_w = abs_cosine_dissimilarity
        dist_fun_a = abs_cosine_dissimilarity
    elif symmetry == 'even_linear' or symmetry == 'even_linear_positive_scaling':
        dist_fun_w = abs_cosine_dissimilarity
        dist_fun_a = cosine_dissimilarity
    else:
        dist_fun_w = cosine_dissimilarity
        dist_fun_a = cosine_dissimilarity

    wt = np.concatenate([wt, bt[np.newaxis, :]], axis=0)
    ws = np.concatenate([ws, bs[np.newaxis, :]], axis=0)

    # sort teacher weights from highest to lowest output neuron norm
    idx_sorted = np.argsort(np.linalg.norm(at, ord=2, axis=1))[::-1]
    wt = wt[:, idx_sorted].astype(np.double)  # We need higher precision for cosine similarity
    at = at[idx_sorted, :].astype(np.double)

    best_sims_w = []
    student_idx_matched = []
    best_sims_a = []
    for wtt, att in zip(wt.T, at):
        sim_w = np.array([dist_fun_w(cluster_mask * wtt, cluster_mask * wss) for wss in ws.T])
        best_sims_w.append(sim_w.min())
        student_idx_matched.append(sim_w.argmin())
        sim_a = dist_fun_a(att, as_[sim_w.argmin()])
        best_sims_a.append(sim_a)

    ws_matched = ws.T[student_idx_matched]

    fig = plt.figure(figsize=(7, 5), dpi=200)
    gs = fig.add_gridspec(2, 4)
    ax1 = fig.add_subplot(gs[0, 0:3])
    ax2 = fig.add_subplot(gs[1, 0:3])
    ax3 = fig.add_subplot(gs[:, 3])

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), dpi=200, sharex=True)
    ax1.bar(range(len(best_sims_w)), best_sims_w)
    ax2.bar(range(len(best_sims_a)), best_sims_a)
    ax1.set_title("Distance from teacher neurons")
    ax1.set_ylabel("dist( w_t , w_s )")
    ax2.set_xlabel("Teacher neuron index")
    ax2.set_ylabel("dist( a_t , a_s )")
    if log:
        ax1.set_ylim([1e-6, 1])
        ax2.set_ylim([1e-6, 1])
        ax1.set_yscale("log")
        ax2.set_yscale("log")
    else:
        ax1.set_ylim([0, 1])
        ax2.set_ylim([0, 1])

    # disable top and right spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    ax2.invert_yaxis()
    ax2.xaxis.tick_top()
    ax2.set_xticklabels([])

    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    # Looking at the missing student neurons
    missing_students_idx = set(range(as_.shape[0])) - set(student_idx_matched)
    best_sims_w = []
    teacher_idx_matched = []
    student_out_norms = []
    for s_id in missing_students_idx:
        wss = ws.T[s_id]
        if np.linalg.norm(wss) == 0:
            continue
        sim_w = np.array([dist_fun_w(cluster_mask * wtt, cluster_mask * wss) for wtt in wt.T])
        best_sims_w.append(sim_w.min())
        teacher_idx_matched.append(sim_w.argmin())
        student_out_norms.append(np.linalg.norm(as_[s_id], ord=2))

    ws_missing = ws.T[list(missing_students_idx)]

    # sort teacher weights from highest to lowest output neuron norm
    idx_sorted = np.argsort(teacher_idx_matched)
    teacher_idx_matched = np.array(teacher_idx_matched)[idx_sorted]
    student_out_norms = np.array(student_out_norms)[idx_sorted]
    excess_neurons = len(teacher_idx_matched)
    best_sims_w = np.array(best_sims_w)[idx_sorted]
    facecolors = ["C0" if n > 0.1 else "white" for n in student_out_norms]
    linestyles = ["-" if n > 0.1 else "--" for n in student_out_norms]

    if len(best_sims_w) < 10:
        best_sims_w = np.concatenate([best_sims_w, np.zeros(10 - len(best_sims_w))])
        teacher_idx_matched = np.concatenate([teacher_idx_matched, np.zeros(10 - len(teacher_idx_matched))-1]).astype(
            int)
        facecolors = np.concatenate([facecolors, ["white"] * (10 - len(facecolors))])
        linestyles = np.concatenate([linestyles, ["--"] * (10 - len(linestyles))])

    ax3.barh(range(len(teacher_idx_matched)), best_sims_w, edgecolor="C0")
    # change each bar's facecolor based on facecolors
    for i, patch in enumerate(ax3.patches):
        patch.set_facecolor(facecolors[i])
        patch.set_linestyle(linestyles[i])
    ax3.set_title(f"{excess_neurons} excess neurons")
    ax3.yaxis.tick_right()
    ax3.set_yticks(range(len(teacher_idx_matched)))
    ax3.set_yticklabels([l if l != -1 else '' for l in teacher_idx_matched])
    ax3.set_xlabel("dist( w_t , w_s )")
    ax3.set_ylabel("Teacher neuron index")
    ax3.yaxis.set_label_position("right")

    if log:
        ax3.set_xlim([1e-6, 1])
        ax3.set_xscale("log")
    else:
        ax3.set_xlim([0, 1])
    ax3.invert_xaxis()

    fig.tight_layout()
    return fig, ws_matched, ws_missing



def cosine_dissimilarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:  # if one of the vectors is 0, return orthogonal
        return 1
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def abs_cosine_dissimilarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:  # if one of the vectors is 0, return orthogonal
        return 1
    return 1 - np.abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
