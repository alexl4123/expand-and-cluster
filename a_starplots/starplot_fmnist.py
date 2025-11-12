"""
 # Created on 25.03.2024
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: 
 #
"""
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import platforms.platform
import platforms.registry
from training.metric_logger import MetricLogger

# names of folders based on activation functions
act_funs = ['g', 'sigmoid', 'tanh', 'softplus', 'relu']

# names of folders based on activation functions. Tuple: (noOP, OP, teacher size, student size, reconstructed size,
# reconstructed loss)
dirs = {'g': ('ee42afba65', 'a1a79fd602', '64', '256', '71', 1.14e-3)}
# setting the paths

platform = platforms.registry.get('local')(num_workers=0)
platforms.platform._PLATFORM = platform
output_location = pathlib.Path().resolve().joinpath('a_starplots', 'plots')

try: platform.makedirs(output_location)
except: pass

fig, axes = plt.subplots(1, 1, figsize=(3, 3), dpi=350)
for i, (act_fun, ax) in enumerate(zip(act_funs, [axes])):
    noOP, OP, t_size, s_size, r_size, r_loss = dirs[act_fun]
    noOP_path = os.path.join(platform.root, f'ec_{noOP}', 'seed_1', 'main')
    OP_path = os.path.join(platform.root, f'ec_{OP}', 'seed_1', 'main')

    logger_noOP = MetricLogger.create_from_file(noOP_path)
    loss_noOP = logger_noOP.get_data('train_individual_losses')[-1][1].sqrt()
    # loss_noOP = [np.sqrt(0.03)]*20

    logger_OP = MetricLogger.create_from_file(OP_path)
    loss_OP = logger_OP.get_data('train_individual_losses')[-1][1].sqrt()

    jitter = (np.random.rand(len(loss_noOP))-0.5) * 0.6
    ax.scatter(jitter, loss_noOP, label='noOP', color='blue', alpha=0.5)
    ax.scatter(1, r_loss, label='EC', color='orange', marker='*', s=200)
    jitter = (np.random.rand(len(loss_noOP)) - 0.5) * 0.6 + 2
    ax.scatter(jitter, loss_OP, label='OP', color='red', alpha=0.5)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([str(t_size), str(r_size), str(s_size)])
    if i==0:
        ax.set_ylabel('RMSE')
    ax.set_xlabel('hidden layer size')
    ax.set_title(f'$\sigma=${act_fun}')
    ax.set_yscale('log')
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(4e-4, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if i == 4:
        ax.legend()

# fig.suptitle('MNIST sampled with FashionMNIST')
fig.tight_layout()
fig.show()
fig.savefig(output_location.joinpath('fmnist_mnist.pdf'))

print("end")
