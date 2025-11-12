"""
 # Created on 18.12.2023
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Description: control plot OTO vs. EC
 #
"""
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from context import *
import platforms.registry
import platforms.platform
from foundations import paths
from training.metric_logger import MetricLogger
from utils.utils import find_final_model_step

import matplotlib
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# # matplotlib.rcParams['svg.fonttype'] = "none"
# matplotlib.rcParams['font.size'] = 12


teacher_seeds = ['1']

platform = platforms.registry.get('local')(num_workers=0)
platforms.platform._PLATFORM = platform
home_location = pathlib.Path().resolve()
controls_location = home_location.joinpath('controls', 'oto_sweep')
ec_location = home_location.joinpath('data', 'sims', 'ec_400b4ca63c', 'seed_1', 'main',
                                     'clustering_5298cd3394', 'reconstructed_model')
lth_location = home_location.joinpath('data', 'sims', 'lottery_c032b769b2', 'seed_1')
han_location = home_location.joinpath('data', 'sims', 'lottery_f172431ac2', 'seed_1')

lambdas = []
sizes = []
losses = []

for teacher_seed in teacher_seeds:
    oto_location = controls_location.joinpath(f"teacher_seed_{teacher_seed}")
    # loop through directories of all lambdas
    for lambda_dir in oto_location.iterdir():
        lbda = float(lambda_dir.name.split('_')[-1])
        maybe_train_end_step = find_final_model_step(lambda_dir, 118)  # 118 is the number of it per epoch
        if maybe_train_end_step is not None:  # load saved model before returning
            state_dict = platform.load_model(paths.model(lambda_dir, maybe_train_end_step),
                                             map_location=platform.torch_device)
            w = state_dict["fc_layers.0.weight"]
            final_size = w[torch.linalg.norm(w, axis=1) != 0].shape[0]

            # load logger from lambda_dir
            logger = MetricLogger.create_from_file(lambda_dir)
            final_loss = logger.get_data('train_loss')[-1][1]
            lambdas.append(lbda)
            sizes.append(final_size)
            losses.append(final_loss)

relative_sizes = np.array(sizes)/128
# sort losses and sizes by lambdas
lambdas, relative_sizes, losses = zip(*sorted(zip(lambdas, relative_sizes, losses)))

## load EC logger
ec_logger = MetricLogger.create_from_file(ec_location)
final_loss_EC = ec_logger.get_data('train_loss')[-1][1]
## get size of EC model
maybe_train_end_step = find_final_model_step(ec_location, 118)  # 118 is the number of it per epoch
state_dict_EC = platform.load_model(paths.model(ec_location, maybe_train_end_step),
                                    map_location=platform.torch_device)
reconstruction_size = state_dict_EC["fc_layers.0.fc"].shape[1]
relative_size_EC = reconstruction_size/128

## load LTH data

lth_sizes = []
lth_losses = []
lth_params = []

for level in range(11):
    lth_location_level = lth_location.joinpath(f"level_{level}", "main")
    # load logger from lambda_dir
    logger = MetricLogger.create_from_file(lth_location_level)
    final_loss = logger.get_data('train_loss')[-1][1]
    lth_losses.append(final_loss)
    # load model from lambda_dir
    mask = torch.load(os.path.join(lth_location_level, "mask.pth"))
    w = mask["fc_layers.0.fc"].squeeze().float()
    final_size = w[:, torch.linalg.norm(w, axis=0) != 0].shape[1]
    param_count = w[w != 0].shape[0]
    lth_sizes.append(final_size)
    lth_params.append(param_count)
lth_params = np.array(lth_params)/(128*784)
lth_sizes = np.array(lth_sizes)/128

## load Han data

han_sizes = []
han_losses = []
han_params = []

for level in range(11):
    han_location_level = han_location.joinpath(f"level_{level}", "main")
    # load logger from lambda_dir
    logger = MetricLogger.create_from_file(han_location_level)
    final_loss = logger.get_data('train_loss')[-1][1]
    han_losses.append(final_loss)
    # load model from lambda_dir
    mask = torch.load(os.path.join(han_location_level, "mask.pth"))
    w = mask["fc_layers.0.fc"].squeeze().float()
    final_size = w[:, torch.linalg.norm(w, axis=0) != 0].shape[1]
    param_count = w[w != 0].shape[0]
    han_sizes.append(final_size)
    han_params.append(param_count)
han_params = np.array(han_params)/(128*784)
han_sizes = np.array(han_sizes)/128

## Load EC parameter sweep data
ec_parent_location = home_location.joinpath('data', 'sims', 'ec_400b4ca63c', 'seed_1', 'main')

ec_sweeps = []
for dir in ec_parent_location.iterdir():
    if not dir.name.startswith('clustering'):
        continue
    params = torch.load(os.path.join(dir, "extraction_hparams_dict"))['extraction_hparams']
    gamma = params.gamma
    beta = params.beta
    logs = MetricLogger.create_from_file(os.path.join(dir, "reconstructed_model"))
    final_loss = logs.get_data('train_loss')[-1][1]
    maybe_train_end_step = find_final_model_step(os.path.join(dir, "reconstructed_model"), 118)
    state_dict = platform.load_model(paths.model(os.path.join(dir, "reconstructed_model"), maybe_train_end_step),
                                     map_location=platform.torch_device)
    w = state_dict["fc_layers.0.fc"]
    final_size = w.shape[1]/128
    ec_sweeps.append((gamma, beta, final_loss, final_size))

# np.sqrt() all losses
losses = np.sqrt(np.array(losses))
han_losses = np.sqrt(np.array(han_losses))
lth_losses = np.sqrt(np.array(lth_losses))

fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
cmap = plt.get_cmap('winter')
for i, (loss, size) in enumerate(zip(losses, relative_sizes)):
    ax.scatter(loss, size, marker='s', color=cmap(i/len(losses)), edgecolor='k', s=70, alpha=0.75)
cmap = plt.get_cmap('cool_r')
for i, (loss, size) in enumerate(zip(han_losses, han_params)):
    ax.scatter(loss, size, marker='o', color=cmap(i/len(han_losses)), edgecolor='k', s=70, alpha=0.75)
cmap = plt.get_cmap('autumn')
for i, (loss, size) in enumerate(zip(lth_losses, lth_params)):
    ax.scatter(loss, size, marker='^', color=cmap(i/(2*(len(lth_losses)-1))+0.5), edgecolor='k', s=70, alpha=0.75)
ax.axhline(y=1, color='k', linestyle='--', alpha=0.25)

# add OTO axes inside the current one
cmap = plt.get_cmap('winter')
oto_ax = ax.inset_axes([0.1, 0.88, 0.2, 0.05])
oto_ax.imshow(np.arange(len(lambdas))[np.newaxis, :], cmap=cmap)
# set xticks and xticklabels
oto_ax.set_xticks([0, 3, 6, 9])
oto_ax.set_xticklabels(['0.1', '1', '10', '100'])
oto_ax.tick_params(axis='both', which='both', bottom=True, top=False, left=False, right=False,
                   labelbottom=True, labelleft=False)
ax.annotate('1e-4', (6e-3, 4.43), fontsize=9)
ax.annotate('OTOv2', (8e-4, 4.95), fontsize=12)
ax.scatter(3e-3, 5.025, marker='s', color="white", edgecolor='k', s=70)
ax.annotate('$\lambda$', (5e-4, 4.7))

# add LTH axes inside the current one
cmap = plt.get_cmap('autumn')
lth_ax = ax.inset_axes([0.445, 0.88, 0.2, 0.05])

lth_ax.imshow(np.arange(11)[np.newaxis, :], cmap=cmap, vmin=-10, vmax=10)
lth_ax.set_xticks([0, 5, 10])
lth_ax.set_xticklabels(['0', '5', '10'])
lth_ax.tick_params(axis='both', which='both', bottom=True, top=False, left=False, right=False,
                   labelbottom=True, labelleft=False)
ax.annotate('LTH', (2e-2, 4.95), fontsize=12)
ax.scatter(5e-2, 5.025, marker='^', color="white", edgecolor='k', s=70)
ax.annotate('iter.', (1e-2, 4.7))

# add Han axes inside the current one
cmap = plt.get_cmap('cool_r')
han_ax = ax.inset_axes([0.725, 0.88, 0.2, 0.05])
han_ax.imshow(np.arange(11)[np.newaxis, :], cmap=cmap)
han_ax.set_xticks([0, 5, 10])
han_ax.set_xticklabels(['0', '5', '10'])
han_ax.tick_params(axis='both', which='both', bottom=True, top=False, left=False, right=False,
                   labelbottom=True, labelleft=False)
ax.annotate('Han', (2e-1, 4.95), fontsize=12)
ax.scatter(0.6, 5.025, marker='o', color="white", edgecolor='k', s=70)
ax.annotate('iter.', (1.2e-1, 4.7))


# ax.scatter(final_loss_EC, relative_size_EC, color='r', marker='*', s=125, edgecolor='k')
ax.annotate('EC (our work)', (np.sqrt(final_loss_EC)+3e-4, relative_size_EC+0.1), color='r', fontsize=12)

for gamma, beta, loss, size in ec_sweeps:
    if gamma < 0.4 or beta > 20:
        continue
    ax.scatter(np.sqrt(loss), size, edgecolor='r', marker='*', s=10, facecolor='none', zorder=-1)
    if size < 1.04:
        ax.scatter(np.sqrt(loss), size, edgecolor='k', marker='*', s=120, facecolor='r', zorder=2, linewidths=0.5)
    # ax.annotate(f'EC (sweep)', (loss, size+0.1), color='r', fontsize=12)


ax.set_xscale('log')
ax.set_xlabel('RMSE')
ax.set_ylabel('Excess parameters')
ax.set_yticklabels(['', 'x0', 'x1', 'x2', 'x3', 'x4'])
ax.set_ylim([0, 5.3])
# remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.set_title('EC vs. OTO')
fig.tight_layout()
fig.savefig(os.path.join(home_location, 'controls', 'oto_sweep', 'control_params.svg'))
fig.show()