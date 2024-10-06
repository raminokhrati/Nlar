# Author: Ramin OKhrati


import os
from experiments.config.config import  shared_path
os.chdir(shared_path)
import tensorflow as tf
import time
import copy
from tqdm import tqdm
from experiments.src_epxeriments.ddqn_agent import  DQNAgent
import random

from experiments.src_epxeriments.utils.utils_general import (load_obj, save_results, load_results, custome_mean_loss, compute_avg_return_rl,
                                                             int_dtype)
from experiments.src_epxeriments.utils.utils_callbacks import BatchLossCallback, BatchLerCallback, EpochLossCallback
from experiments.src_epxeriments.data.load_data import shuffle_data_with_seed
from experiments.src_epxeriments.models import model_generate
from experiments.src_epxeriments.utils.utils_experiments_setup import  setup_experiment
import numpy as np
from src.adamhd_optimizer import AdamHD
from src.nlar_optimizers import Nlarc, Nlars
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


##

def train_cls_rgr_perseed(
    # Dataset Parameters
    train_dataset=None,
    test_dataset=None,
    val_dataset=None,
    x_train=None,
    y_train=None,

    # Training Configuration
    loss=None,
    metrics=None,
    callbacks=None,
    callbacks_args=None,
    optimizer=None,
    model_conf=None,
    training_params=None,
    optimizer_args=None,
    seed=None,

    # Recording losses: binaries
    reg_train_loss=False,
    reg_train_loss_batch=None,
    reg_learning_rate=None,

    # Logging and Strategy
    experiment_name=None,
    data_source=None,
    strategy=None
):

    """This function perform either a classification or regression and save the history of the model and the optimizer
    for a specific seed."""

    optimizer_args_temp = optimizer_args.copy()

    model_conf['seed'] = seed
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if strategy is not None:
        with strategy.scope():
            model = model_generate(**model_conf)
            model.compile(optimizer=optimizer(**optimizer_args_temp), loss=loss, metrics=metrics)
            try:
                model.set_weights(load_obj('experiments/initial_parameters/' + 'init_weights/' + experiment_name
                                       + '/' + "model_orig_w_" +
                                       data_source + str(seed)))
            except Exception as e:
                print(e)
                print(f"Ensure that the path to load the weights exist. If this is the first run, you possibly need to set"
                      f"use_saved_initial_weights to be False so that new weights are created.")

    else:
        model = model_generate(**model_conf)
        model.compile(optimizer=optimizer(**optimizer_args_temp), loss=loss, metrics=metrics)
        try:
            model.set_weights(load_obj('experiments/initial_parameters/' + 'init_weights/' + experiment_name
                                       + '/' + "model_orig_w_" +
                                       data_source + str(seed)))
        except Exception as e:
            print(e)
            print(f"Ensure that the path to load the weights exist. If this is the first run, you possibly need to set"
                  f"use_saved_initial_weights to be False so that new weights are created.")

    if reg_train_loss:
        # Since model.evaluate()[0] calculate losses not in an inference mode; we use a custom made loss calculator
        # to calculate the actual losses
        loss_values_train = custome_mean_loss(train_dataset, model=model, loss=loss, training=False)
    else: loss_values_train = None
    if type(loss).__name__ in ['BinaryCrossentropy', 'CategoricalCrossentropy', 'SparseCategoricalCrossentropy']:
        train_accuracy = model.evaluate(train_dataset, verbose=0)[1]
    else: train_accuracy = None

    # We only allow for either to have a test dataset or validation dataset but not both at the same time
    assert test_dataset is None or val_dataset is None

    if test_dataset is not None:
        target_dataset = test_dataset
        target_dataname = 'test dataset'
    elif val_dataset is not None:
        target_dataset = val_dataset
        target_dataname = 'validation dataset'
    else:
        target_dataset = None
        target_dataname = None


    loss_values_target = custome_mean_loss(target_dataset, model=model, loss=loss, training=False)

    if type(loss).__name__ in ['BinaryCrossentropy', 'CategoricalCrossentropy', 'SparseCategoricalCrossentropy']:
        # target_accuracy = model_accuracy_tensor(model, target_dataset)
        target_accuracy = model.evaluate(target_dataset, verbose=0)[1]
    else: target_accuracy = None
    if training_params['verbose'] >= 1:
        print(f"\nEpoch loss on {target_dataname} before the start of training: {loss_values_target}")
        if type(loss).__name__ in ['BinaryCrossentropy', 'CategoricalCrossentropy', 'SparseCategoricalCrossentropy']:
            print(f'\nEpoch accuracy on {target_dataname} before the start of training: {target_accuracy}')

    training_params_temp = training_params.copy()
    callbacks_temp = []
    train_dataset_copy = list(train_dataset).copy()

    # Since default log history of tf.keras model is based on training=True, in order to get the statistical
    # training losses, call backs are required.
    if reg_train_loss_batch:
        callbacks_temp.append(BatchLossCallback(name='reg_train_loss_batch', data=train_dataset_copy, training=False))
    if reg_train_loss:
        callbacks_temp.append(EpochLossCallback(name='reg_train_loss', data=(x_train, y_train), training=False, loss=loss))
    if reg_learning_rate:
        callbacks_temp.append(BatchLerCallback(name='reg_learning_rate'))

    if callbacks is not None:
        for callback in callbacks:
            if callback.__name__ == 'ModelCheckpoint':
                callback_args = callbacks_args['ModelCheckpoint']
                path_optimizer_args = "_".join(f"{key}-{value}" for key, value in optimizer_args_temp.items())
                callback_args['filepath'] = ('initial_parameters/save_best_weights/' + experiment_name + '/' + "model_best_w_"
                                             + data_source + optimizer.__name__ + path_optimizer_args + str(seed))
                model_checkpoint_callback = ModelCheckpoint(**callback_args)
                callbacks_temp.append(model_checkpoint_callback)

            elif callback.__name__ == 'ReduceLROnPlateau':
                callback_args = callbacks_args['ReduceLROnPlateau']
                callback_redlr = ReduceLROnPlateau(**callback_args)
                callbacks_temp.append(callback_redlr)
            else: callbacks_temp.append(callback)

    tf.print(f"Model's fitting using optimizer {optimizer.__name__} with parameters {optimizer_args_temp}")

    if strategy is not None:
        with strategy.scope():
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            history = model.fit(train_dataset, callbacks=callbacks_temp, **training_params_temp,
                                validation_data=target_dataset, shuffle=False)
    else:
        history = model.fit(train_dataset, callbacks=callbacks_temp, **training_params_temp,
                            validation_data=target_dataset, shuffle=False)

    history = history.history
    history['val_loss'].insert(0, loss_values_target)
    if type(loss).__name__ in ['BinaryCrossentropy', 'CategoricalCrossentropy', 'SparseCategoricalCrossentropy']:
       history['val_accuracy'].insert(0, target_accuracy)

    # When there is EpochLossCallback in the list of callbacks, we are in the inference mode, and the training loss
    # is an actual loss. However, if this is not the case, then we only collect the losses from the training and
    # append the initial loss.
    if len(callbacks_temp) == 0 or EpochLossCallback not in callbacks:
        history['loss'].insert(0, loss_values_train)
    if type(loss).__name__ in ['BinaryCrossentropy', 'CategoricalCrossentropy', 'SparseCategoricalCrossentropy']:
        history['accuracy'].insert(0, train_accuracy)

    for callback in callbacks_temp:
        if type(callback).__name__ == 'BatchLossCallback' and reg_train_loss_batch:
            history['loss_batch'] = callback.batch_losses
        if type(callback).__name__ == 'BatchLerCallback' and reg_learning_rate:
            history['lers'] = callback.lers
        if type(callback).__name__ == 'EpochLossCallback' and reg_train_loss:
            history['loss'] = callback.epoch_losses_end
            history['loss'].insert(0, loss_values_train)
        if type(callback).__name__ == 'ModelCheckpoint':
            checkpoint_filepath = callback.filepath
            model.load_weights(checkpoint_filepath)

    history_seed = copy.deepcopy(history)

    if type(loss).__name__ in ['BinaryCrossentropy', 'CategoricalCrossentropy', 'SparseCategoricalCrossentropy']:

        # Evaluate the model on the target dataset
        # Note that if ModelCheckpoint i used, the value of accu_seed is not necessarily the same value
        # as val_accuracy in history.history
        accu_seed = model.evaluate(target_dataset, verbose=0)[1]
        # a custom made function can also be used
        # accu_seed = model_accuracy_tensor(model, target_dataset)
    else: accu_seed = custome_mean_loss(target_dataset, model=model, loss=loss, training=False)

    return history_seed, accu_seed

##
def train_cls_rgr(
    # Data Parameters
    x_train=None, y_train=None,
    x_test=None, y_test=None,
    data_config=None,

    # Model and Training Configuration
    model_conf=None,
    training_params=None,
    loss=None, metrics=None,
    optimizer=None, optimizer_args=None,

    # Recording losses: binaries
    reg_train_loss=None, reg_train_loss_batch=None,
    reg_learning_rate=None,

    # Seeds for Initialization
    seeds=None,

    # Callbacks and Strategy
    callbacks=None, callbacks_args=None,
    strategy=None,

    # Logging and Data Source
    experiment_name=None, data_source=None,

    # Overwrite: if True, no saved results_experiments are called.
    overwrite=None,
):

    """This function train the model over seeds; where for each seed, it trains the model using train_cls_rgr_perseed
    method"""

    batch_size = training_params['batch_size']
    validation_split = training_params['validation_split']

    accus = []
    hists = []
    exe_times = []

    if overwrite:
        for seed_idx, seed in enumerate(seeds):

            data = shuffle_data_with_seed(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                          seed=seed, data_config=data_config, validation_split=validation_split,
                                          batch_size=batch_size)

            init_time = time.time()
            hist_temp, accu_temp = train_cls_rgr_perseed(train_dataset=data['train_dataset'],
                                                         test_dataset=data['test_dataset'],
                                                         val_dataset=data['val_dataset'],
                                                         x_train=data['x_train'],
                                                         y_train=data['y_train'],
                                                         loss=loss, metrics=metrics,
                                                         callbacks=callbacks,
                                                         callbacks_args=callbacks_args,
                                                         optimizer=optimizer,
                                                         model_conf=model_conf,
                                                         training_params=training_params, optimizer_args=optimizer_args,
                                                         reg_train_loss=reg_train_loss,
                                                         reg_train_loss_batch=reg_train_loss_batch,
                                                         reg_learning_rate=reg_learning_rate,
                                                         seed=seed, experiment_name=experiment_name, data_source=data_source,
                                                         strategy=strategy)
            exe_time_temp = time.time() - init_time

            print(f"The performance of {optimizer.__name__}, using callbacks {[_.__name__ for _ in callbacks]}, on unseen dataset with four decimal places of accuracy, "
                  f"for seed number {seed_idx+1},"
                  f" is {round(accu_temp, 4)}. \n")

            accus.append(accu_temp)
            print(f"The average performance of {optimizer.__name__}, using callbacks {[_.__name__ for _ in callbacks]}, on unseen dataset, "
                  f"with four decimal places of accuracy, up until seed number {seed_idx+1},"
                  f" is {round(np.mean(accus), 4)}. ")

            hists.append(hist_temp)
            exe_times.append(exe_time_temp)

        save_results(optimizer=optimizer, accus_score=accus, experiment_name=experiment_name, seeds=seeds,
                     optimizer_args=optimizer_args, hists=hists, exe_times=exe_times, callbacks=callbacks,
                     training_params=training_params)

    else:
        try:
            accus, hists, exe_times = load_results(optimizer=optimizer, experiment_name=experiment_name, seeds=seeds,
                                                   optimizer_args=optimizer_args, callbacks=callbacks, training_params=training_params)

            print(f"\n The average time, per epoch, to execute experiment {experiment_name} using optimizer {optimizer.__name__} is {np.mean(exe_times) / training_params['epochs']} seconds")
        except Exception as e:
            print(e)
            print(f"The value of overwrite is {overwrite}; if there are no saved results (for instance if this is the first run), please set the value of overwrite to be True.")

    return accus, hists, exe_times

##

def get_callbacks():
    # determine callbacks
    callbacks_args = {}
    callbacks_args['ModelCheckpoint'] = dict(
        filepath='',
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    callbacks = [ModelCheckpoint]
    return callbacks_args, callbacks

##
def run_experiment_cls_rgr(
                        # experiments
                        experiment_name=None,
                        data_source=None,
                        n_seeds=None,
                        lambda_value=None,
                        use_bias_input=None,

                        # cpu gpu specifications
                        m_threshold=None,
                        cpu_parallel=None,
                        gpu_parallel=None,

                        # training parameters
                        validation_split=None,
                        epochs=None,
                        verbose=None,
                        batch_size=None,

                        # loging
                        use_saved_initials_seed=None,
                        use_saved_initial_weights=None,
                        reg_train_loss=None,
                        reg_train_loss_batch=None,
                        reg_learning_rate=None,
                        overwrite=None,

                        optimizers=None,

                        # optimizers' parameters
                        momentums=None,
                        lr_rates=None,
                        betas=None,

                        sns_analysis_vars=None
                        ):

    """This method implements the experiments."""

    (x_train, y_train, x_test, y_test, data_config, model_conf, training_params,
     seeds, callbacks, callbacks_args, strategy, optimizers_args, loss, metrics) = setup_experiment(
                                                                    experiment_name=experiment_name,
                                                                    data_source=data_source,
                                                                    lambda_value=lambda_value,
                                                                    use_bias_input=use_bias_input,

                                                                    # cpu gpu specifications
                                                                    m_threshold=m_threshold,
                                                                    cpu_parallel=cpu_parallel,
                                                                    gpu_parallel=gpu_parallel,

                                                                    # training parameters
                                                                    validation_split=validation_split,
                                                                    epochs=epochs,
                                                                    verbose=verbose,
                                                                    batch_size=batch_size,

                                                                    # general parameters
                                                                    use_saved_initials_seed=use_saved_initials_seed,
                                                                    use_saved_initial_weights=use_saved_initial_weights,

                                                                    # number of seeds
                                                                    n_seeds=n_seeds
                                                                    )

    for optimizer in optimizers:
        optimizer_args = optimizers_args[optimizer.__name__.lower() + '_args']
        for momentum in momentums:
            if optimizer in [Nlars, Nlarc]:
                optimizer_args['momentum'] = momentum
            else:
                if 'momentum' in optimizer_args:
                    optimizer_args.pop('momentum')

            for beta in betas:

                if optimizer == AdamHD:
                    optimizer_args['beta'] = beta
                else:
                    if 'beta' in optimizer_args:
                        optimizer_args.pop('beta')

                accus = {}
                for lr in lr_rates:
                    optimizer_args['learning_rate'] = lr
                    if sns_analysis_vars is None:
                        sns_analysis_vars_temp = {'none': [None]}
                    else: sns_analysis_vars_temp = sns_analysis_vars
                    for sns_var_name, sns_vars in sns_analysis_vars_temp.items():
                        for sns_var in sns_vars:
                            if sns_analysis_vars is not None and sns_var_name in optimizer_args:
                                assert sns_var_name not in ['beta', 'optimizer', 'lr', 'learning_rate']
                                optimizer_args[sns_var_name] = sns_var
                            accus_temp, *_ = train_cls_rgr(
                                # Data and Model Configuration
                                x_train=x_train,
                                y_train=y_train,
                                x_test=x_test,
                                y_test=y_test,
                                data_config=data_config,
                                model_conf=model_conf,

                                # Training Parameters
                                training_params=training_params,
                                loss=loss, metrics=metrics,
                                optimizer=optimizer,
                                optimizer_args=optimizer_args,
                                reg_train_loss=reg_train_loss,
                                reg_train_loss_batch=reg_train_loss_batch,
                                reg_learning_rate=reg_learning_rate,

                                # Seeds and Initialization
                                seeds=seeds,

                                # Callbacks and Logging
                                callbacks=callbacks, callbacks_args=callbacks_args,
                                experiment_name=experiment_name,
                                data_source=data_source,
                                overwrite=overwrite,
                                strategy=strategy
                            )

                    accus[lr] = accus_temp

                print(
                    f"The average accuracy of {optimizer.__name__} for parameters "
                    f"{[(_, optimizer_args[_]) for _ in optimizer_args if _ != 'learning_rate']}, on the test set is\n "
                    f"{[(_, np.round(np.mean(accus[_]), 4)) for _ in accus]}")

                if optimizer != AdamHD:
                    print(f"Optimizer {optimizer.__name__} does not have beta parameter.")
                    break

            if optimizer in [Adam, AdamHD]:
                print(f"Optimizer {optimizer.__name__} does not have momentum parameter.")
                break

##

def train_ddqn_agent(
        # Model and Training Parameters
        state_size=None,
        action_size=None,
        optimizer=None,
        optimizer_args=None,
        loss=None,
        model_conf=None,
        training_params=None,

        # Task Type and Seeding
        seeds=None,

        # File and Data Settings
        experiment_name=None,
        data_source=None,

        # Environment Settings
        env=None,
        env_eval=None,
        overwrite=None
):
    """This function trains a DDQN agent over three different series of seeds."""

    batch_size = training_params['batch_size']
    step_episode = training_params['step_episode']
    n_iterations = training_params['n_iterations']
    eval_interval = training_params['eval_interval']
    num_eval_episodes = training_params['num_eval_episodes']

    buffer_size = training_params['buffer_size']
    gamma = training_params['gamma']
    target_network_update_freq = training_params['target_network_update_freq']
    update_freq = training_params['update_freq']
    train_start = training_params['train_start']

    assert buffer_size > train_start, "Buffer size is smaller than the train_start size"

    if not overwrite:

        accus, hists, exe_times = load_results(optimizer=optimizer, experiment_name=experiment_name, seeds=seeds,
                                               callbacks=None, training_params=training_params,
                                               optimizer_args=optimizer_args)

        return accus, hists, exe_times

    else:
        score = None
        exe_times = []
        episode_rewards_lists = {'train_episode_rewards_list':[], 'eval_episode_rewards_list':[]}
        final_step_score = {'train_final_step_score':None, 'eval_final_step_score': None}

        for seed in seeds:
            training_params_temp = training_params.copy()

            print(f"Start training for game {experiment_name} using optimizer {optimizer.__name__} "
                  f"with parameters {optimizer_args} \n")

            np.random.seed(seed)
            random.seed(seed)
            optimizer_args_temp = optimizer_args.copy()

            epsilon = training_params_temp['epsilon']
            epsilon_min = training_params_temp['epsilon_min']
            epsilon_decay = training_params_temp['epsilon_decay']

            agent = DQNAgent(action_size=action_size,
                             optimizer=optimizer, optimizer_args=optimizer_args_temp,
                             seed=seed,
                             buffer_size=buffer_size, model_conf=model_conf,
                             gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min,
                             epsilon_decay=epsilon_decay, loss=loss
                             )

            agent.model.set_weights(load_obj('initial_parameters/' + 'init_weights/' + experiment_name + '/'
                                             + "model_orig_w_" + data_source + str(seed)))
            agent.update_target_model()

            # Evaluate the agent's policy before training.
            avg_return = compute_avg_return_rl(environment=env_eval, agent=agent, num_episodes=num_eval_episodes,
                                               state_size=state_size, step_episode=step_episode, seed=seed)

            eval_returns = [avg_return]
            episode_rewards = []

            for e in tqdm(range(n_iterations)):
                start_time = time.time()
                state = env.reset(seed=seed + e + 1)
                env_eval.reset()

                state = np.reshape(state[0], [1, state_size])
                total_reward = 0  # Reset total reward for the episode

                for score in range(step_episode):

                    # step_episode is the maximum number of steps per episode which depends on a game

                    action = agent.act(state).numpy()

                    try:
                        next_state, reward, done, *_ = env.step(action)
                    except: next_state, reward, done, *_ = env.step([action])
                    next_state = np.reshape(next_state, [1, state_size])

                    # Extracting just the base game name without the version
                    full_game_name = env.spec.id
                    base_game_name = full_game_name.split('-')[0]
                    if base_game_name == 'CartPole':
                        reward = reward if not done or score == step_episode - 1 else -1
                    
                    # Store the experience in memory
                    agent.memory.add((state, action, reward, next_state, done))
                    state = next_state
                    total_reward += reward  # Add reward to total reward

                    if done:
                        agent.update_target_model()
                        print(f"Episode: {e}/{n_iterations}, Game Done, Score: {score}, "
                              f"AR: {np.sum(episode_rewards) / (len(episode_rewards) + 1)}, "
                              f"Epsilon: {float(agent.epsilon):.2}")
                        break
                    if len(agent.memory.buffer) > train_start and e % update_freq == 0:
                        minibatch = agent.memory.sample(batch_size)

                        # Extract components of experiences
                        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))
                        if experiment_name == 'dqn_pendulum':
                            actions = np.minimum(actions, 2.)
                            actions = np.maximum(actions, -2.)

                        # Convert to tensor dataset
                        states = tf.convert_to_tensor(states, dtype=tf.keras.backend.floatx())
                        actions = tf.convert_to_tensor(actions, dtype=tf.keras.backend.floatx())
                        rewards = tf.convert_to_tensor(rewards, dtype=tf.keras.backend.floatx())
                        next_states = tf.convert_to_tensor(next_states, dtype=tf.keras.backend.floatx())
                        dones = tf.convert_to_tensor(dones, dtype=int_dtype())

                        agent.replay(states, actions, rewards, next_states, dones)

                        if agent.epsilon > agent.epsilon_min:
                            agent.epsilon *= agent.epsilon_decay

                    if e % target_network_update_freq == 0:
                        agent.update_target_model()

                if e % eval_interval == 0 and e != 0:

                    avg_return = compute_avg_return_rl(environment=env_eval, agent=agent, num_episodes=num_eval_episodes,
                                                       state_size=state_size, step_episode=step_episode, seed=seed + e + 1)
                    eval_returns.append(avg_return)
                    print(f"Episode: {e}/{n_iterations}, AR on Unseen Env.: "
                          f"{np.sum(eval_returns) / (len(eval_returns) + 1)}, Epsilon: {float(agent.epsilon):.2}")

                episode_rewards.append(total_reward)
                exe_times.append(time.time() - start_time)

                if e % 10 == 0:
                    print(f"\n Total rewards in episode {e}: {total_reward}")

            # Close the environment after training
            env.close()

            del agent

            episode_rewards_lists['train_episode_rewards_list'].append(episode_rewards)
            final_step_score['train_final_step_score'] = score
            episode_rewards_lists['eval_episode_rewards_list'].append(eval_returns)
            final_step_score['eval_final_step_score'] = avg_return

        save_results(optimizer=optimizer, accus_score=final_step_score, experiment_name=experiment_name, seeds=seeds,
                     optimizer_args=optimizer_args, training_params=training_params,
                     hists=episode_rewards_lists, exe_times=exe_times, callbacks=None)

        return final_step_score, episode_rewards_lists, exe_times
#%%

def run_experiment_lr(
        # Environment and Agent Configuration
        state_size=None,
        action_size=None,
        env=None,
        env_eval=None,

        # Optimization and Learning
        optimizers=None,
        optimizers_args=None,

        momentums=None,
        betas=None,
        lr_rates=None,

        loss=None,

        training_params=None,

        # Model and Memory
        model_conf=None,

        # Miscellaneous
        seeds=None,
        experiment_name=None,
        data_source=None,
        overwrite=None):

    score = None
    for optimizer in optimizers:
        optimizer_args = optimizers_args[optimizer.__name__.lower() + '_args']

        for momentum in momentums:
            if optimizer in [Nlars, Nlarc]:
                optimizer_args['momentum'] = momentum
            else:
                if 'momentum' in optimizer_args:
                    optimizer_args.pop('momentum')

            for beta in betas:

                if optimizer == AdamHD:
                    optimizer_args['beta'] = beta
                else:
                    if 'beta' in optimizer_args:
                        optimizer_args.pop('beta')

                for lr in lr_rates:
                    optimizer_args['learning_rate'] = lr

                    score, *_ = train_ddqn_agent(
                        # Environment and Agent Configuration
                        state_size=state_size,
                        action_size=action_size,
                        env=env,
                        env_eval=env_eval,

                        # Optimization and Learning
                        optimizer=optimizer,
                        optimizer_args=optimizer_args,
                        loss=loss,
                        training_params=training_params,

                        # Model and Memory
                        model_conf=model_conf,

                        # Miscellaneous
                        seeds=seeds,
                        experiment_name=experiment_name,
                        data_source=data_source,
                        overwrite=overwrite
                    )

                if optimizer != AdamHD:
                    print(f"Optimizer {optimizer.__name__} does not have beta parameter.")
                    break

            print(f"Score of {optimizer.__name__} for parameters {optimizer_args}, on the unseen env is {score}")

            if optimizer in [Adam, AdamHD]:
                print(f"Optimizer {optimizer.__name__} does not have momentum parameter.")
                break

##

def run_experiment_cls_rgr_sensitivity(
                        # experiments
                        experiment_name=None,
                        data_source=None,
                        n_seeds=None,
                        lambda_value=None,
                        use_bias_input=None,

                        # cpu gpu specifications
                        m_threshold=None,
                        cpu_parallel=None,
                        gpu_parallel=None,

                        # training parameters
                        validation_split=None,
                        epochs=None,
                        verbose=None,
                        batch_size=None,

                        # loging
                        use_saved_initials_seed=None,
                        use_saved_initial_weights=None,
                        reg_train_loss=None,
                        reg_train_loss_batch=None,
                        reg_learning_rate=None,
                        overwrite=None,

                        optimizers=None,

                        # optimizers' parameters
                        momentums=None,
                        lr_rates=None,
                        betas=None
                        ):
    """This method implements the experiments."""

    (x_train, y_train, x_test, y_test, data_config, model_conf, training_params,
     seeds, callbacks, callbacks_args, strategy, optimizers_args, loss, metrics) = setup_experiment(
                                                                    experiment_name=experiment_name,
                                                                    data_source=data_source,
                                                                    lambda_value=lambda_value,
                                                                    use_bias_input=use_bias_input,

                                                                    # cpu gpu specifications
                                                                    m_threshold=m_threshold,
                                                                    cpu_parallel=cpu_parallel,
                                                                    gpu_parallel=gpu_parallel,

                                                                    # training parameters
                                                                    validation_split=validation_split,
                                                                    epochs=epochs,
                                                                    verbose=verbose,
                                                                    batch_size=batch_size,

                                                                    # general parameters
                                                                    use_saved_initials_seed=use_saved_initials_seed,
                                                                    use_saved_initial_weights=use_saved_initial_weights,

                                                                    # number of seeds
                                                                    n_seeds=n_seeds
                                                                    )

    for optimizer in optimizers:
        optimizer_args = optimizers_args[optimizer.__name__.lower() + '_args']
        for momentum in momentums:
            if optimizer in [Nlars, Nlarc]:
                optimizer_args['momentum'] = momentum
            else:
                if 'momentum' in optimizer_args:
                    optimizer_args.pop('momentum')

            for beta in betas:

                if optimizer == AdamHD:
                    optimizer_args['beta'] = beta
                else:
                    if 'beta' in optimizer_args:
                        optimizer_args.pop('beta')

                accus = {}
                for lr in lr_rates:
                    optimizer_args['learning_rate'] = lr

                    accus_temp, *_ = train_cls_rgr(
                        # Data and Model Configuration
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test,
                        data_config=data_config,
                        model_conf=model_conf,

                        # Training Parameters
                        training_params=training_params,
                        loss=loss, metrics=metrics,
                        optimizer=optimizer,
                        optimizer_args=optimizer_args,
                        reg_train_loss=reg_train_loss,
                        reg_train_loss_batch=reg_train_loss_batch,
                        reg_learning_rate=reg_learning_rate,

                        # Seeds and Initialization
                        seeds=seeds,

                        # Callbacks and Logging
                        callbacks=callbacks, callbacks_args=callbacks_args,
                        experiment_name=experiment_name,
                        data_source=data_source,
                        overwrite=overwrite,
                        strategy=strategy
                    )

                    accus[lr] = accus_temp

                print(
                    f"The average accuracy of {optimizer.__name__} for parameters "
                    f"{[(_, optimizer_args[_]) for _ in optimizer_args if _ != 'learning_rate']}, on the test set is\n "
                    f"{[(_, np.round(np.mean(accus[_]), 4)) for _ in accus]}")

                if optimizer != AdamHD:
                    print(f"Optimizer {optimizer.__name__} does not have beta parameter.")
                    break


            if optimizer in [Adam, AdamHD]:
                print(f"Optimizer {optimizer.__name__} does not have momentum parameter.")
                break