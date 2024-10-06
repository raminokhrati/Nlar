
import os

from experiments.config.config import shared_path
os.chdir(shared_path)

import tensorflow as tf

from src.nlar_optimizers import Nlarc, Nlars
from experiments.src_epxeriments.training import run_experiment_cls_rgr
from experiments.src_epxeriments.utils.utils_plot import  plot_experiment_cls_rgr

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

##

# determine the floating accuracies
tf.keras.backend.set_floatx('float64')
data_source = 'cifar10'
experiment_name = 'cifar10_mlp7h'

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
use_saved_initials_seed = True # to use saved initial seeds
use_saved_initial_weights = True # to use saved initial weights of the training variables
reg_train_loss = False # to save training losses
reg_train_loss_batch = False # to save batch losses on training
reg_learning_rate = False # to save the history of learning rates
overwrite = False # to recall saved results_experiments

# number of seeds
n_seeds = 3

# determine optimizers' parameters
momentums = [1]
lr_rates = [0.1]
betas = [1e-7, 1e-4]

optimizers = [Nlars, Nlarc]

# sns_analysis_vars = {'momentum':[None, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95, 1 ]}

# other examples of sensitivity parameters
sns_analysis_vars = {'sigma':[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-15]} #
# sns_analysis_vars = {'k0':[-1e10, -1e5, -1, -1e-5, -1e-10, 1e-10, 1e-5, 1, 1e5, 1e10]}#
# sns_analysis_vars = {'lower_clip':[1e-150, 1e-100, 1e-50, 1e-25, 1e-20, 1e-15, 1e-10, 1e-5, 1e-2, 1e-1]} #
# sns_analysis_vars = {'global_clipnorm': [1e5]}#1e-10, 1e-5, 1e-2, 1e-1, 1, 1e1, 1e2, 1e5, 1e10, 1e15]} #

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
                        sns_analysis_vars=sns_analysis_vars
                        )

##
# plot accuracies or losses versus epochs
# first_column references to the first column of the group pictures need to have axes titles

plot_experiment_cls_rgr(
    # Data and Loss Configuration
    curve_types=['val_accuracy'],
    momentums=momentums,
    betas=betas,
    xlimit_batch=10,
    xlimit=epochs + 1,

    # Data for Plotting
    optimizers=optimizers,
    lr_rates=lr_rates,
    maxval_limit=2,

    # Seeds and Initialization

    n_seeds=n_seeds,
    use_saved_initial_seed=use_saved_initials_seed,
    experiment_name=experiment_name,
    data_source=data_source,

    # training parameters
    validation_split=validation_split,
    epochs=epochs,
    verbose=verbose,
    batch_size=batch_size,
    sns_analysis_vars=sns_analysis_vars
)
