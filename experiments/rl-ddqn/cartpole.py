

import os
from src.config import shared_path
os.chdir(shared_path)

import tensorflow as tf
tf.keras.backend.set_floatx('float64')

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import gym

import random

from src.utils.utils_general import save_obj, load_obj, parallel_computing_strategy
from src.utils.utils_experiments_setup import model_generate
from src.utils.utils_plot import plot_experiment_rl
from src.training import run_experiment_lr


from src.optimizers import Nlarc, Nlars, AdamHD
from tensorflow.keras.optimizers import Adam

##

game_name = 'CartPole-v0'
# Environment setup
env = gym.make(game_name)
env_eval = gym.make(game_name)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
gamma = 0.95  # discount rate
epsilon = 1.  # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995

buffer_size = 10000
train_start = 1000

batch_size = 64
step_episode = 200
n_iterations = 1000

eval_interval = 1
num_eval_episodes = 10

dqn_mlp_conf = [state_size, 128, action_size]

target_network_update_freq = 40
update_freq = 1

##

data_source = game_name
experiment_name = 'dqn_cartpole'
n_seeds = 3

# model configuration
use_bias_input = True
lambda_value = 0.0001
is_regression = True

model_conf = dict(network_type='mlp', mlp_conf=dqn_mlp_conf, lambda_value=lambda_value, use_bias_input=use_bias_input,
                  is_regression=is_regression, seed=None)

# general parameters
use_saved_initials_seed = True
use_saved_initial_weights = True
overwrite = False

# seed selection
seeds = []

for _ in range(n_seeds):
    if not use_saved_initials_seed:
        seeds.append(random.randint(0, 2 ** 32 - 1))

if not use_saved_initials_seed:
    save_obj(seeds, 'initial_parameters/' + 'seeds/' + experiment_name + '/' + data_source)
else:
    seeds = load_obj('initial_parameters/' + 'seeds/' + experiment_name + '/' + data_source)

##
# determine loss and initial weight generation

loss = tf.keras.losses.mean_squared_error

for seed in seeds:
    if not use_saved_initial_weights:
        model_conf['seed'] = seed
        model = model_generate(**model_conf)

        model.compile(loss=loss)
        model_orig_w = model.get_weights()

        save_obj(model_orig_w, 'initial_parameters/' + 'init_weights/' + experiment_name + '/' + "model_orig_w_" + data_source + str(seed))

##

momentums = [1]
lr_rates =  [0.0001, 0.001, 0.01, 0.1, 0.5, 1]
betas = [1e-4, 1e-7]

nlars_args = {
    'learning_rate': None,
    'sigma': 1e-30,
    'k0': 1,
    'momentum': None,
    'global_clipnorm': 1,
    'lower_clip': 1e-150
}

nlarc_args = {
    'learning_rate': None,
    'sigma': 1e-30,
    'k0': 1,
    'momentum': None,
    'global_clipnorm': 1,
}

adam_args = {
    'learning_rate': None,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-7,
    'global_clipnorm': 1,
    'amsgrad': False
}

adamhd_args = {
    'beta': None,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-08,
    'global_clipnorm': 1,
    'learning_rate': None
}

optimizers_args = dict(adam_args=adam_args,
                       adamhd_args=adamhd_args,
                       nlarc_args=nlarc_args,
                       nlars_args=nlars_args)

training_params = dict(batch_size=batch_size, step_episode=step_episode, n_iterations=n_iterations,
                       eval_interval=eval_interval, num_eval_episodes=num_eval_episodes, buffer_size=buffer_size,
                       gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
                       target_network_update_freq=target_network_update_freq, update_freq=update_freq,
                       train_start=train_start)


##
optimizers = [Nlars, Nlarc, Adam, AdamHD]

run_experiment_lr(
                    # Environment and Agent Configuration
                    state_size=state_size,
                    action_size=action_size,
                    env=env,
                    env_eval=env_eval,

                    # Optimization and Learning
                    optimizers=optimizers,
                    optimizers_args=optimizers_args,

                    momentums=momentums,
                    betas=betas,
                    lr_rates=lr_rates,

                    loss=loss,
                    training_params=training_params,

                    # Model and Memory
                    model_conf=model_conf,

                    # Evaluation and Training Control

                    # Miscellaneous
                    # is_regression=is_regression,
                    seeds=seeds,
                    experiment_name=experiment_name,
                    data_source=data_source,
                    overwrite=overwrite
                    )

##

callbacks=[]
min_lrs = [None]
momentums = [1]
optimizers = [Nlars, Nlarc, AdamHD, Adam]
return_type = 'testing'
betas = [1e-04, 1e-07]
lr_rates = [0.0001, 0.001, 0.01, 0.1, 0.5, 1]
curve_type = 'cumulative_reward'

plot_experiment_rl(
    # Optimizer Settings
    optimizers=optimizers,
    optimizers_args=optimizers_args,
    lr_rates=lr_rates,
    min_lrs=min_lrs,
    momentums=momentums,
    betas=betas,

    # Training Settings
    training_params=training_params,
    seeds=seeds,
    callbacks=callbacks,

    # Experiment Identification
    experiment_name=experiment_name,
    game_name=game_name,
    return_type=return_type,
    curve_type=curve_type
)

##

plot_experiment_rl(
    # Optimizer Settings
    optimizers=optimizers,
    optimizers_args=optimizers_args,
    lr_rates=lr_rates,
    min_lrs=min_lrs,
    momentums=momentums,
    betas=betas,

    # Training Settings
    training_params=training_params,
    seeds=seeds,
    callbacks=callbacks,

    # Evaluation Settings
    # num_eval_episodes=num_eval_episodes,
    # eval_interval=eval_interval,

    # Experiment Identification
    experiment_name=experiment_name,
    game_name=game_name,
    return_type=return_type,
    curve_type=curve_type
)

