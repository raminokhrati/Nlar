
import os

from experiments.config.config import shared_path
os.chdir(shared_path)

import tensorflow as tf#

from src.adamhd_optimizer import AdamHD
from src.nlar_optimizers import Nlarc, Nlars
from experiments.src_epxeriments.training import run_experiment_cls_rgr
from experiments.src_epxeriments.utils.utils_plot import  plot_experiment_cls_rgr

from tensorflow.keras.optimizers import Adam


os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

##
# determine the floating accuracies
tf.keras.backend.set_floatx('float64')
data_source = 'cifar10'
experiment_name = 'cifar10_vgg11'

##

# model configuration
use_bias_input = True
lambda_value = 0.0001

# training parameters
validation_split = 0
epochs = 50
verbose = 1
batch_size = 300

# general parameters
use_saved_initials_seed = False # to use saved initial seeds
use_saved_initial_weights = False # to use saved initial weights of the training variables
reg_train_loss = False # to save training losses
reg_train_loss_batch = False # to save batch losses on training
reg_learning_rate = False # to save the history of learning rates
overwrite = True # to recall saved results_experiments

# number of seeds
n_seeds = 3

# determine optimizers' parameters
momentums = [1]
lr_rates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1]
betas = [1e-7, 1e-4]

optimizers = [Nlars, Nlarc, Adam, AdamHD]
sns_analysis_vars = None

##

run_experiment_cls_rgr(
                        # experiments
                        experiment_name=experiment_name,
                        data_source=data_source,
                        n_seeds=n_seeds,
                        lambda_value=lambda_value,
                        use_bias_input=use_bias_input,

                        # cpu gpu specifications
                        m_threshold=0.1, # memory threshold
                        cpu_parallel=False,
                        gpu_parallel=True,

                        # training parameters
                        validation_split=validation_split,
                        epochs=epochs,
                        verbose=verbose,
                        batch_size=batch_size,

                        # loging
                        use_saved_initials_seed=use_saved_initials_seed,
                        use_saved_initial_weights=use_saved_initial_weights,
                        reg_train_loss=reg_train_loss,
                        reg_train_loss_batch=reg_train_loss_batch,
                        reg_learning_rate=reg_learning_rate,
                        overwrite=overwrite,

                        optimizers=optimizers,

                        # optimizers' parameters
                        momentums=momentums,
                        lr_rates=lr_rates,
                        betas=betas,
                        sns_analysis_vars=None
                    )

##
# plot accuracies or losses versus epochs
# first_column references to the first column of the group pictures need to have axes titles

for beta in betas:
    for optimizers in [[Nlars, Adam]]:
        for curve_type in [['accuracy', 'val_accuracy']]:
            plot_experiment_cls_rgr(
                # Data and Loss Configuration
                curve_types=curve_type,
                momentums=momentums,
                betas=[beta],
                xlimit_batch=10,
                xlimit=epochs + 1,

                # Data for Plotting
                optimizers=optimizers,
                lr_rates=lr_rates,
                maxval_limit=2, # put a limit on the learning_raet for adam and adamhd so they can be visualized better

                # Seeds and Initialization
                n_seeds=n_seeds,
                use_saved_initial_seed=use_saved_initials_seed,
                experiment_name=experiment_name,
                data_source=data_source,

                # training parameters
                validation_split=validation_split,
                epochs=epochs,
                verbose=verbose,
                batch_size=batch_size)
