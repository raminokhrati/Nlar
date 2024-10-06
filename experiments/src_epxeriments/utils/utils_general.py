
import os
from experiments.config.config import shared_path
os.chdir(shared_path)

import numpy as np

import pickle
import tensorflow as tf

import pynvml


##
def int_dtype():
    """Check the backend floating precession and return the equivalent int one"""

    if tf.keras.backend.floatx() == 'float32':
        return 'int32'
    elif tf.keras.backend.floatx() == 'float64':
        return 'int64'
    else:
        raise ValueError('Please set the floating precision accuracy to be either 32 or 64')

##

def get_ler_history(ler_seq=None, layer=None, node_in_layer=None, node_in_pre_layer=None):
    """The first hidden layer is layer number zero, if it exists. If there is no hidden layer, layer zero
     is a reference to the output layer, so for instance in a logistic regression model (785,10)
       layer_no=0 is a reference to the output layer made of 10 nodes. So in this example,
       we have layer=0; 0<=node_in_layer<=9
       and 0<=node_pre_layer<=784.
       In general, Layer_no=0,1,..... If there is a bias term, special attention shall be made towards the bias learning
       rates in the history. For instance, if there is bias learning rates,
       layer=1 is not necessarily the hidden layer."""

    path = None
    len_history = len(ler_seq)
    if node_in_layer is not None and node_in_pre_layer is not None:
        path = np.array([ler_seq[_][layer] for _ in range(len_history)])[:, node_in_pre_layer, node_in_layer]
    elif node_in_layer is None and node_in_pre_layer is None:
        path = np.mean(np.array([ler_seq[_][layer] for _ in range(len_history)]),
                       axis=0)
    return path

##

@tf.function
def loss_x_y(x, y, model, loss, training):
    """This function calculate the loss value for inputs (x,y), model, and a loss function"""
    y_pred = model(x, training=training)
    loss_value = loss(y, y_pred)

    return loss_value

def custome_mean_loss(dataset, model=None, loss=None, training=False):
    """This function calculate the loss value for dataset, modle, and loss."""

    losses = []
    no_samples = 0
    for x_batch, y_batch in dataset:
        batch_size = len(y_batch)
        no_samples += batch_size

        loss_value = loss_x_y(x_batch, y_batch, model, loss, training)
        losses.append(batch_size * loss_value)

    # Compute the overall loss by averaging the batch losses
    loss_value = tf.reduce_sum(losses) / no_samples
    return loss_value.numpy()

##

def save_obj(obj, file_name):
    """It saves obj in pickle format using file_name."""
    with open(file_name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_name):
    """It retrieves a saved object in pickle format with file name, file_name"""
    with open(file_name + '.pkl', 'rb') as f:
            return pickle.load(f)

# %%

def f_name_gen(dic, withkey=True):
    """Given dictionary dic; it generates a file name. This function creates concise file name
    to save the results_experiments."""
    if withkey:
        try:
            return "_".join([f"{key}={value:.2}" for key, value in dic.items()])
        except:
            return "_".join([f"{key}={value}" for key, value in dic.items()])
    else:
        try:
            return "_".join([f"{value:.2}" for value in dic.values()])
        except:
            return "_".join([f"{value}" for value in dic.values()])


# %%

def save_results(optimizer=None, accus_score=None, experiment_name=None, seeds=None, optimizer_args=None,
                 hists=None, exe_times=None, callbacks=None, training_params=None):
    """This function saves the results_experiments."""
    assert seeds is not None and training_params is not None

    path_temp = (experiment_name + '/' + optimizer.__name__ +
                 f_name_gen(dict(n_seeds=len(seeds))) + str(hash(tuple(seeds))) +
                 f_name_gen(optimizer_args) +  f_name_gen(training_params, withkey=False))

    min_lr = None
    if callbacks is not None and len(callbacks) >= 1:
        for callback in callbacks:
            if callback.__name__ == 'ReduceLROnPlateau':
                min_lr = callback.min_lr

        callbacks_names = [_.__name__ for _ in callbacks]

        if 'ModelCheckpoint' not in callbacks_names:
            save_obj(accus_score, 'experiments/results_experiments/' + 'accuss/' +
                     path_temp + '_accus_call_minlr_' + str(min_lr) + 'no_model_check' + tf.keras.backend.floatx())
            save_obj(hists, 'experiments/results_experiments/' + 'histories/' +
                     path_temp + '_hist_call_minlr_' + str(min_lr) + 'no_model_check' + tf.keras.backend.floatx())
            save_obj(exe_times, 'experiments/results_experiments/' + 'exe_times/' +
                     path_temp + '_exe_time_call_minlr_' + str(min_lr) + 'no_model_check' + tf.keras.backend.floatx())
        else:
            save_obj(accus_score, 'experiments/results_experiments/' + 'accuss/' +
                     path_temp + '_accus_call_minlr_' + str(min_lr) + tf.keras.backend.floatx())
            save_obj(hists, 'experiments/results_experiments/' + 'histories/' +
                     path_temp + '_hist_call_minlr_' + str(min_lr) + tf.keras.backend.floatx())
            save_obj(exe_times, 'experiments/results_experiments/' + 'exe_times/' +
                     path_temp + '_exe_time_call_minlr_' + str(min_lr) + tf.keras.backend.floatx())
    else:
        save_obj(accus_score, 'experiments/results_experiments/' + 'accuss/' + path_temp + tf.keras.backend.floatx())
        save_obj(hists, 'experiments/results_experiments/' + 'histories/' + path_temp + tf.keras.backend.floatx())
        save_obj(exe_times, 'experiments/results_experiments/' + 'exe_times/' + path_temp + tf.keras.backend.floatx())


def load_results(optimizer=None, experiment_name=None, seeds=None, training_params=None,
                 optimizer_args=None, callbacks=None):
    """This function load the results_experiments."""

    assert seeds is not None

    path_temp = (experiment_name + '/' + optimizer.__name__ +
                 f_name_gen(dict(n_seeds=len(seeds))) + str(hash(tuple(seeds))) +
                 f_name_gen(optimizer_args) +  f_name_gen(training_params, withkey=False))

    min_lr = None

    if callbacks is not None and len(callbacks) >= 1:
        for callback in callbacks:
            if callback.__name__ == 'ReduceLROnPlateau':
                min_lr = callback.min_lr

        callbacks_names = [_.__name__ for _ in callbacks]

        if 'ModelCheckpoint' not in callbacks_names:
            accus = load_obj('experiments/results_experiments/' + 'accuss/' +
                             path_temp + '_accus_call_minlr_' + str(min_lr) + 'no_model_check' +
                             tf.keras.backend.floatx())
            hists = load_obj('experiments/results_experiments/' + 'histories/' +
                             path_temp + '_hist_call_minlr_' + str(min_lr) + 'no_model_check' +
                             tf.keras.backend.floatx())
            exe_times = load_obj('experiments/results_experiments/' + 'exe_times/' +
                                 path_temp + '_exe_time_call_minlr_' + str(min_lr) + 'no_model_check' +
                                 tf.keras.backend.floatx())
        else:
            accus = load_obj('experiments/results_experiments/' + 'accuss/' +
                             path_temp + '_accus_call_minlr_' + str(min_lr) + tf.keras.backend.floatx())
            hists = load_obj('experiments/results_experiments/' + 'histories/' +
                             path_temp + '_hist_call_minlr_' + str(min_lr) + tf.keras.backend.floatx())
            exe_times = load_obj('experiments/results_experiments/' + 'exe_times/' +
                                 path_temp + '_exe_time_call_minlr_' + str(min_lr) + tf.keras.backend.floatx())
    else:

        accus = load_obj('experiments/results_experiments/' + 'accuss/' + path_temp + tf.keras.backend.floatx())
        hists = load_obj('experiments/results_experiments/' + 'histories/' + path_temp + tf.keras.backend.floatx())
        exe_times = load_obj('experiments/results_experiments/' + 'exe_times/' + path_temp + tf.keras.backend.floatx())

    return accus, hists, exe_times

##

def compute_avg_return_rl(environment=None, agent=None, num_episodes=None, state_size=None, step_episode=None, seed=None):

    assert seed is not None

    total_return = 0.0

    for e in range(num_episodes):

        state = environment.reset(seed=seed * e + 1)

        state = np.reshape(state[0], [1, state_size])
        total_reward = 0  # Reset total reward for the episode

        for score in range(step_episode):
            action = agent.act(state).numpy()
            try:
                next_state, reward, done, *_ = environment.step(action)
            except: next_state, reward, done, *_ = environment.step([action])

            next_state = np.reshape(next_state, [1, state_size])

            # Penalize the agent for dropping the pole
            # reward = reward if not done or score == step_episode - 1 else -1

            # Store the experience in memory

            state = next_state

            total_reward += reward  # Add reward to total reward

            if done:
                break
        total_return += total_reward

    avg_return = total_return / num_episodes
    return avg_return
##

def parallel_computing_strategy(m_threshold=0.1, cpu_parallel=True, gpu_parallel=True):

    def initialize_nvml():
        try:
            pynvml.nvmlInit()
            return True
        except pynvml.NVMLError as err:
            print(f"Failed to initialize NVML: {err}")
            return False

    if not initialize_nvml():
        return None, None

    def get_device_count():
        return pynvml.nvmlDeviceGetCount()

    def get_gpu_info(index):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        return memory_info, processes

    def is_gpu_available(index, m_threshold_inner):
        memory_info, processes = get_gpu_info(index)
        # Consider a GPU available if memory usage is below a threshold and no significant processes are running
        if memory_info.used < (memory_info.total * m_threshold_inner):# and len(processes) == 0:
            return True
        return False

    device_count = get_device_count()
    print(f"Number of GPUs Available: {device_count}")

    available_gpus_indexes = []
    for i in range(device_count):
        if is_gpu_available(i, m_threshold):
            available_gpus_indexes.append(i)

        print(f"GPU {i}:")
        memory_info, processes = get_gpu_info(i)
        print(f"  Memory Usage: {memory_info.used / 1024**2:.2f} MiB / {memory_info.total / 1024**2:.2f} MiB")
        for process in processes:
            print(f"  Process {process.pid}:")
            print(f"    Memory Usage: {process.usedGpuMemory / 1024**2:.2f} MiB")

    if len(available_gpus_indexes) >= 1 and gpu_parallel:
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            # setting 2 GPUs
            tf.config.experimental.set_visible_devices(gpus[0:2], 'GPU')
        except RuntimeError as e:
            print(e)
        strategy = tf.distribute.MirroredStrategy()
    elif cpu_parallel:
        strategy = tf.distribute.experimental.CentralStorageStrategy()
    else:
        strategy = None
    pynvml.nvmlShutdown()

    print(f"Available GPU indexes: {available_gpus_indexes}; computing strategy: {strategy}")

    return available_gpus_indexes, strategy

