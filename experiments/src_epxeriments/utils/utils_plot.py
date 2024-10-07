import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import math
import numpy as np
import distinctipy

from experiments.src_epxeriments.utils.utils_general import  load_obj, f_name_gen, load_results, get_ler_history
from experiments.src_epxeriments.utils.utils_experiments_setup import setup_optimizers_args, get_seeds
from experiments.src_epxeriments.utils.utils_callbacks import get_experiments_callbacks

import tensorflow as tf

##
def plot_lers(epochs=None, validation_split=None, verbose=None, batch_size=None, range_of_nodes_in_layer=None,
              layer=None, node_in_previous_layer=None, optimizer=None, experiment_name=None, n_seeds=None,
              data_source=None, use_saved_initials_seed=None, learning_rate=None, momentum=None, beta=None):
    """Graph the individual learning rates histories. This current version only works for MLP"""

    key_optz = optimizer.__name__

    seeds = get_seeds(experiment_name=experiment_name, data_source=data_source,
                      n_seeds=n_seeds, use_saved_initials_seed=use_saved_initials_seed)
    optimizers_args = setup_optimizers_args()

    training_params = {
        'epochs': epochs,
        'validation_split': validation_split,
        'verbose': verbose,
        'batch_size': batch_size
    }

    callbacks_args, callbacks  = get_experiments_callbacks()


    # Determine the optimizer arguments based on the optimizer name
    if key_optz == 'Adam':
        optimizer_args = optimizers_args['adam_args']
    elif key_optz == 'Nlars':
        optimizer_args = optimizers_args['nlars_args']
        optimizer_args['momentum'] = momentum
    elif key_optz == 'Nlarc':
        optimizer_args = optimizers_args['nlarc_args']
        optimizer_args['momentum'] = momentum
    elif key_optz == 'AdamHD':
        optimizer_args = optimizers_args['adamhd_args']
        optimizer_args['beta'] = beta
    elif key_optz == 'Adagrad':
        optimizer_args = optimizers_args['adagrad_args']
    else:
        raise ValueError(f"Optimizer {key_optz} is not recognized.")
    optimizer_args['learning_rate'] = learning_rate
    _, ax = plt.subplots()

    # Retrieve histories
    _, opts_hist, _ = load_results(
        optimizer=optimizer,
        experiment_name=experiment_name,
        seeds=seeds,
        training_params=training_params,
        optimizer_args=optimizer_args,
        callbacks=callbacks
    )

    # Process and plot histories
    for node_layer_no in range_of_nodes_in_layer:
        node_hists = []
        for s in range(n_seeds):
            ler_hist = opts_hist[s]['lers']
            node_hist = get_ler_history(ler_seq=ler_hist, layer=layer, node_in_layer=node_layer_no,
                                        node_in_pre_layer=node_in_previous_layer)
            node_hists.append(node_hist)

        node_hist_average = np.mean(np.array(node_hists), axis=0)
        ax.plot(node_hist_average.transpose())

    # Finalize and show plot
    plt.ylabel('Learning Rate')
    plt.xlabel('Iteration')
    figname = key_optz + '_lers'
    plt.savefig('imgs/' + figname, dpi=600)
    plt.show()

##
def plot_optimizer_learning_curves(index=None, item=None, optimizer_args=None, experiment_name=None, opt=None,
                                   seeds=None, training_params=None, ler_value_temp=None, curve_type=None, xlimit=None,
                                   xlimit_batch=None, ler_exponent=None, c=None, ax=None, linestyle=None,
                                   callbacks=None, sns_analysis_vars=None, index_exponent=None, index_value_temp=None):

    """Plot learning curves. The legend values are shown in scientific format, i.e 10^value; unless value is zero,
    in which case it becomes 1."""

    n_seeds = len(seeds)
    optname = opt.__name__
    if index in optimizer_args:
        optimizer_args[index] = item
    if optname == 'AdamHD':
        beta_exponent = int(math.log10(abs(optimizer_args['beta'])))

    path_temp = (experiment_name + '/' + optname +
         f_name_gen(dict(n_seeds=len(seeds))) + str(hash(tuple(seeds))) +
         f_name_gen(optimizer_args) +  f_name_gen(training_params, withkey=False))

    min_lr = None
    if callbacks is not None and len(callbacks) >= 1:
        for callback in callbacks:
            if callback.__name__ == 'ReduceLROnPlateau':
                min_lr = callback.min_lr

        callbacks_names = [_.__name__ for _ in callbacks]

        if 'ModelCheckpoint' not in callbacks_names:

            hists = load_obj(
                'experiments/results_experiments/' + 'histories/' + path_temp + '_hist_call_minlr_' +
                str(min_lr) + 'no_model_check' + tf.keras.backend.floatx())

        else:
            hists = load_obj('experiments/results_experiments/' + 'histories/' + path_temp + '_hist_call_minlr_' + str(min_lr) + tf.keras.backend.floatx())
    else:
        hists = load_obj('experiments/results_experiments/' + 'histories/' + path_temp + tf.keras.backend.floatx())

    s_nr = []
    for s in range(n_seeds):
        if curve_type != 'loss_batch':
            s_nr.append(hists[s][curve_type][:xlimit])
        else:
            s_nr.append(hists[s][curve_type][:xlimit_batch])


    # In order to make losses visible maybe some scaling would be helpful like:
    # s_nr = [np.log10(s_nr[_]) for _ in range(n_seeds)]
    if optname == 'AdamHD':
        if ler_exponent != 0:
            ax.plot(np.mean(np.array(s_nr).reshape(n_seeds, -1), axis=0),
                    linestyle=linestyle, label=rf"{opt.__name__}: $\lambda_0=10^{{{ler_exponent}}}$, $\beta=10^{{{beta_exponent}}}$", color=c)
        else:
            ax.plot(np.mean(np.array(s_nr).reshape(n_seeds, -1), axis=0),
                    linestyle=linestyle, label=rf"{opt.__name__}: $\lambda_0={ler_value_temp}$, $\beta=10^{{{beta_exponent}}}$", color=c)
    elif optname == 'Adam':
        if ler_exponent != 0:
            ax.plot(np.mean(np.array(s_nr).reshape(n_seeds, -1), axis=0),
                    linestyle=linestyle, label=rf"{opt.__name__}: $\lambda_0=10^{{{ler_exponent}}}$", color=c)
        else:
            ax.plot(np.mean(np.array(s_nr).reshape(n_seeds, -1), axis=0),
                    linestyle=linestyle, label=rf"{opt.__name__}: $\lambda_0={ler_value_temp}$", color=c)

    elif optname == 'Nlarc' or optname == 'Nlars':
        if ler_exponent != 0 and index == 'momentum':
            if optimizer_args[index] is not None:
                label = rf"{optname + 'm'}: $\lambda_0=10^{{{ler_exponent}}}$, $\rho={{{optimizer_args[index]}}}$"
            else:
                label = rf"{optname + 'm'}: $\lambda_0=10^{{{ler_exponent}}}$, $\rho={{{0}}}$"
        elif ler_exponent == 0 and index == 'momentum':
            if optimizer_args[index] is not None:
                label = rf"{optname + 'm'}: $\lambda_0={{{ler_value_temp}}}$, $\rho={{{optimizer_args[index]}}}$"
            else: label = rf"{optname + 'm'}: $\lambda_0={{{ler_value_temp}}}$, $\rho={{{0}}}$"
        elif ler_exponent != 0 and index_exponent != 0:
            if sns_analysis_vars is not None:
                if index == 'sigma' and optname == 'Nlarc':
                    label = rf"{optname+'m'}: $\lambda_0=10^{{{ler_exponent}}}$, $c=10^{{{index_exponent}}}$"
                elif index == 'sigma' and optname == 'Nlars':
                    label = rf"{optname+'m'}: $\lambda_0=10^{{{ler_exponent}}}$, $c^\prime=10^{{{index_exponent}}}$"
                elif index == 'k0':
                    label = rf"{optname + 'm'}: $\lambda_0=10^{{{ler_exponent}}}$, $k={{{str(np.sign(index_value_temp))[0] if index_value_temp < 0 else ''}}}10^{{{index_exponent}}}$"
                elif index == 'lower_clip':
                    label = rf"{optname + 'm'}: $\lambda_0=10^{{{ler_exponent}}}$, $B^\prime={{{str(np.sign(index_value_temp))[0] if index_value_temp < 0 else ''}}}10^{{{index_exponent}}}$"
                elif index == 'global_clipnorm':
                    label = rf"{optname + 'm'}: $\lambda_0=10^{{{ler_exponent}}}$, $b={{{str(np.sign(index_value_temp))[0] if index_value_temp < 0 else ''}}}10^{{{index_exponent}}}$"

            else:
                if optimizer_args['momentum'] is not None:
                    label = rf"{optname + 'm'}: $\lambda_0=10^{{{ler_exponent}}}$"
                else: label = rf"{optname}: $\lambda_0=10^{{{ler_exponent}}}$"
        elif ler_exponent == 0 and index_exponent != 0:
            if sns_analysis_vars is not None:
                if index == 'sigma' and optname == 'Nlarc':
                    label = rf"{optname+'m'}: $\lambda_0={ler_value_temp}$, $c=10^{{{index_exponent}}}$"
                elif index == 'sigma' and optname == 'Nlars':
                    label = rf"{optname + 'm'}: $\lambda_0={ler_value_temp}$, $c^\prime=10^{{{index_exponent}}}$"
                elif index == 'k0':
                    label = rf"{optname + 'm'}: $\lambda_0={ler_value_temp}$, $k={{{str(np.sign(index_value_temp))[0] if index_value_temp < 0 else ''}}}10^{{{index_exponent}}}$"
                elif index == 'lower_clip':
                    label = rf"{optname + 'm'}: $\lambda_0={ler_exponent}$, $B^\prime={index_value_temp}$"
                elif index == 'global_clipnorm':
                    label = rf"{optname + 'm'}: $\lambda_0={ler_exponent}$, $b={index_value_temp}$"

            else:
                if optimizer_args['momentum'] is not None:
                    label = rf"{optname + 'm'}: $\lambda_0={ler_value_temp}$"
                else: label = rf"{optname}: $\lambda_0={ler_value_temp}$"
        elif ler_exponent != 0 and index_exponent == 0:
            if sns_analysis_vars is not None:
                if index == 'sigma' and optname == 'Nlarc':
                    label = rf"{optname+'m'}: $\lambda_0=10^{{{ler_exponent}}}$, $c={index_value_temp}$"
                if index == 'sigma' and optname == 'Nlars':
                    label = rf"{optname+'m'}: $\lambda_0=10^{{{ler_exponent}}}$, $c^\prime={index_value_temp}$"
                elif index == 'k0':
                    label = rf"{optname + 'm'}: $\lambda_0=10^{{{ler_exponent}}}$, $k={index_value_temp}$"
                elif index == 'lower_clip':
                    label = rf"{optname + 'm'}: $\lambda_0=10^{{{ler_exponent}}}$, $B^\prime={index_value_temp}$"
                elif index == 'global_clipnorm':
                    label = rf"{optname + 'm'}: $\lambda_0=10^{{{ler_exponent}}}$, $b={index_value_temp}$"
            else:
                if optimizer_args['momentum'] is not None:
                    label = rf"{optname + 'm'}: $\lambda_0=10^{{{ler_exponent}}}$"
                else: label = rf"{optname}: $\lambda_0=10^{{{ler_exponent}}}$"

        elif ler_exponent == 0 and index_exponent == 0:
            if sns_analysis_vars is not None:
                if index == 'sigma' and optname == 'Nlarc':
                    label = rf"{optname+'m'}: $\lambda_0={ler_value_temp}$, $c={index_value_temp}$"
                elif index == 'sigma' and optname == 'Nlars':
                    label = rf"{optname+'m'}: $\lambda_0={ler_value_temp}$, $c^\prime={index_value_temp}$"
                elif index == 'k0':
                    label = rf"{optname + 'm'}: $\lambda_0={ler_value_temp}$, $k={index_value_temp}$"
                elif index == 'lower_clip':
                    label = rf"{optname + 'm'}: $\lambda_0={ler_exponent}$, $B^\prime={index_value_temp}$"
                elif index == 'global_clipnorm':
                    label = rf"{optname + 'm'}: $\lambda_0={ler_exponent}$, $b={index_value_temp}$"
            else:
                if optimizer_args['momentum'] is not None:
                    label = rf"{optname + 'm'}: $\lambda_0={ler_value_temp}$"
                else: label = rf"{optname}: $\lambda_0={ler_value_temp}$"

        ax.plot(np.mean(np.array(s_nr).reshape(n_seeds, -1), axis=0),
                linestyle=linestyle,
                label=label, color=c)


##
def plot_loss_accu(
        # Core Parameters
        optimizers=None,
        optimizers_args=None,
        opt_args_loss=None,  # Example: [{'momentum': [0.95]}]

        # Plot Configuration
        curve_type=None,
        figname=None,
        xlimit=None,
        xlimit_batch=None,
        maxval_limit=None,

        # Learning Rate
        lr_rates=None,
        callbacks=None,

        # Seeds
        seeds=None,
        training_params=None,

        # Data and plot Management
        experiment_name=None,
        first_column=None,
        sns_analysis_vars=None
):

    """This function plots learning curves for a set of optimizers."""

    assert lr_rates is not None

    # Number of colors needed
    num_colors = len(lr_rates) * len(list(opt_args_loss.items())[0][1])

    # Generate distinct colors
    colors = iter(distinctipy.get_colors(num_colors, rng=40)) #40

    _, ax = plt.subplots()

    for lr in lr_rates:
        if sns_analysis_vars is None:
            c = next(colors)

        # To avoid decimals like 1. for lr=1 in the plots
        if lr == 1:
            lr_temp = int(lr)
        else:
            lr_temp = lr

        for idx_opt_args_loss, (key, value) in enumerate(opt_args_loss.items()):

            for idx_value, item_value in enumerate(value):
                if sns_analysis_vars is not None:
                    c = next(colors)
                    # if key == 'sigma':
                    if item_value == 1:
                        index_value_temp = int(item_value)
                    else: index_value_temp = item_value
                    if item_value is not None:
                        index_exponent = int(math.log10(np.abs(item_value)))
                    else: index_exponent = 0
                else:
                    index_exponent = None
                    index_value_temp = None

                exponent = int(math.log10(abs(lr)))
                # c = next(color)
                for opt in optimizers:
                    # c = next(color)
                    if opt.__name__ == 'Adam':
                        optimizer_args = optimizers_args['adam_args']
                        linestyle = 'dotted'
                    elif opt.__name__ == 'AdamHD':
                        optimizer_args = optimizers_args['adamhd_args']
                        linestyle = 'dashed'
                    elif opt.__name__ == 'Adagrad':
                        optimizer_args = optimizers_args['adagrad_args']
                        linestyle = (0, (1, 1)) # 'densely dotted'
                    elif opt.__name__ == 'Nlarc':
                        optimizer_args = optimizers_args['nlarc_args']
                        linestyle =  'solid'
                    elif opt.__name__ == 'Nlars':
                        optimizer_args = optimizers_args['nlars_args']
                        linestyle = 'dashdot' #
                    else: raise ValueError("The optimizer is not recongnized.")

                    if opt.__name__ != 'Nlarc' and opt.__name__ != 'Nlars' and key in optimizer_args and lr <= maxval_limit:
                        optimizer_args['learning_rate'] = lr
                        plot_optimizer_learning_curves(index=key, item=item_value, optimizer_args=optimizer_args,
                                                       experiment_name=experiment_name, opt=opt, seeds=seeds,
                                                       training_params=training_params, ler_value_temp=lr_temp,
                                                       curve_type=curve_type, xlimit=xlimit, xlimit_batch=xlimit_batch,
                                                       ler_exponent=exponent, c=c, ax=ax, linestyle=linestyle,
                                                       callbacks=callbacks, sns_analysis_vars=None, index_exponent=None,
                                                       index_value_temp=None)

                    elif (opt.__name__ != 'Nlarc' and opt.__name__ != 'Nlars' and not key in optimizer_args
                          and not set(opt_args_loss.keys()).intersection(optimizer_args.keys()) and lr <= maxval_limit):
                        # in order to avoid repeating the same values on the figure
                        if idx_value == 0 and idx_opt_args_loss == 0:
                            optimizer_args['learning_rate'] = lr
                            plot_optimizer_learning_curves(index=key, item=item_value, optimizer_args=optimizer_args,
                                                           experiment_name=experiment_name, opt=opt, seeds=seeds,
                                                           training_params=training_params, ler_value_temp=lr_temp,
                                                           curve_type=curve_type, xlimit=xlimit,
                                                           xlimit_batch=xlimit_batch, ler_exponent=exponent, c=c, ax=ax,
                                                           linestyle=linestyle, callbacks=callbacks,
                                                           sns_analysis_vars=None, index_exponent=None, index_value_temp=None)

                    if opt.__name__ == 'Nlarc' and key in optimizer_args:
                        # optimizer_args = nlarc_args.copy()
                        optimizer_args['learning_rate'] = lr
                        optimizer_args[key] = item_value
                        if sns_analysis_vars is not None:
                            optimizer_args['momentum'] = 1
                    elif opt.__name__ == 'Nlars' and key in optimizer_args:
                        optimizer_args['learning_rate'] = lr
                        optimizer_args[key] = item_value
                        if sns_analysis_vars is not None:
                            optimizer_args['momentum'] = 1

                    if (opt.__name__ == 'Nlarc' or opt.__name__ == 'Nlars') and key in optimizer_args:
                        plot_optimizer_learning_curves(index=key, item=item_value, optimizer_args=optimizer_args,
                                                       experiment_name=experiment_name, opt=opt, seeds=seeds,
                                                       training_params=training_params, ler_value_temp=lr_temp,
                                                       curve_type=curve_type, xlimit=xlimit, xlimit_batch=xlimit_batch,
                                                       ler_exponent=exponent, c=c, ax=ax, linestyle=linestyle,
                                                       callbacks=callbacks, sns_analysis_vars=sns_analysis_vars,
                                                       index_exponent=index_exponent, index_value_temp=index_value_temp)

    min_lr = None
    if callbacks is not None and len(callbacks) >= 1:
        for callback in callbacks:
            if callback.__name__ == 'ReduceLROnPlateau':
                min_lr = callback.min_lr

    ax.legend(prop={'size': 7})
    if first_column:
        if curve_type == 'loss_batch':
            plt.ylabel('Batch Training Loss' if min_lr is None else 'Batch Training Loss under RLR')
            plt.xlabel('Iteration')
        elif curve_type == 'loss':
            plt.ylabel('Training Loss' if min_lr is None else 'Training Loss under RLR')
            if min_lr is None or experiment_name == 'cifar10_vgg11': # for cifar10_vgg11, we only consider min_lr=0 so we need to show the lables
                plt.xlabel('Epoch')
        elif curve_type == 'val_loss':
            plt.ylabel('Validation Loss' if min_lr is None else 'Validation Loss under RLR')
        elif curve_type == 'val_accuracy':
            plt.ylabel('Validation Accuracy' if min_lr is None else 'Validation Accuracy under RLR')
        elif curve_type == 'accuracy':
            plt.ylabel('Training Accuracy' if min_lr is None else 'Training Accuracy under RLR')
            if min_lr is None or experiment_name == 'cifar10_vgg11':
                plt.xlabel('Epoch')

    else:

        if curve_type == 'loss_batch':
            plt.xlabel('Iteration')
        elif curve_type == 'loss' and min_lr is None:
            plt.xlabel('Epoch')
        elif curve_type == 'accuracy' and min_lr is None:
            plt.xlabel('Epoch')
        elif curve_type == 'accuracy' and min_lr == 0:
            if experiment_name == 'cifar10_vgg11':
                plt.xlabel('Epoch')


    if sns_analysis_vars is None:
        plt.savefig('imgs/' + figname +  '.pdf', dpi=300) # , format='pdf')
    else: plt.savefig('imgs/' + figname + list(sns_analysis_vars.keys())[0] + '.pdf', dpi=300) # , format='pdf')
    plt.show()


##
def plot_return_rl(optimizers=None, return_type=None, betas=None, figname=None, adam_args=None,
                   nlarc_args=None, nlars_args=None, adamhd_args=None, lr_rates=None,
                   seeds=None,
                   experiment_name=None,
                   first_column=None,
                   training_params=None,
                   callbacks=None
                   ):

    assert return_type == 'training' or return_type == 'testing'


    color = iter(cm.rainbow(np.linspace(0, 1, len(lr_rates))))
    for lr in lr_rates:
        adam_args['learning_rate'] = lr
        nlarc_args['learning_rate'] = lr
        nlars_args['learning_rate'] = lr
        adamhd_args['learning_rate'] = lr

        optimizer_args_hyp_params_adam = {**training_params, **adam_args}
        optimizer_args_hyp_params_nlarc = {**training_params, **nlarc_args}
        optimizer_args_hyp_params_nlars = {**training_params, **nlars_args}
        optimizer_args_hyp_params_adamhd = {**training_params, **adamhd_args}

        c = next(color)
        optimizers_args_dic = {}
        returns_seeds_dic = {}

        for opt in optimizers:
            optname = opt.__name__
            if optname == 'Adam':
                optimizer_args = adam_args
                optimizers_args_dic[optname] = optimizer_args_hyp_params_adam

            elif optname == 'Nlarc':
                optimizer_args = nlarc_args
                optimizers_args_dic[optname] = optimizer_args_hyp_params_nlarc
            elif optname == 'Nlars':
                optimizer_args = nlars_args
                optimizers_args_dic[optname] = optimizer_args_hyp_params_nlars


            elif optname == 'AdamHD':
                optimizer_args = adamhd_args
                optimizers_args_dic[optname] = optimizer_args_hyp_params_adamhd
            else: raise ValueError(f"Optimizer {opt} is not recognized.")

            for beta in betas:
                adamhd_args['beta'] = beta
                accus, hists, exe_times = load_results(optimizer=opt, experiment_name=experiment_name, seeds=seeds,
                                                       training_params=training_params, callbacks=callbacks,
                                                       optimizer_args=optimizer_args)

                if return_type == 'training':
                    returns_seeds_dic[optname] = hists['train_episode_rewards_list']
                elif return_type == 'testing':
                    returns_seeds_dic[optname] = hists['eval_episode_rewards_list']

        # for opt in optimizers:
            # optname = opt.__name__
            # if optname == 'Adam':
            #     optimizer_args = adam_args
            #     optimizers_args_dic[optname] = optimizer_args_hyp_params_adam
            #
            # elif optname == 'Nlarc':
            #     optimizer_args = nlarc_args
            #     optimizers_args_dic[optname] = optimizer_args_hyp_params_nlarc
            # elif optname == 'Nlars':
            #     optimizer_args = nlars_args
            #     optimizers_args_dic[optname] = optimizer_args_hyp_params_nlars
            #
            #
            # elif optname == 'AdamHD':
            #     optimizer_args = adamhd_args
            #     optimizers_args_dic[optname] = optimizer_args_hyp_params_adamhd
            # else:
            #     raise ValueError(f"Optimizer {opt} is not recognized.")

            returns = np.mean(returns_seeds_dic[optname], axis=0)
            average_reutrns =np.cumsum(returns[0:]) / np.arange(1, len(returns[0:]) + 1)

            if optname == 'Adam':
                linestyle = 'dotted'
                plt.plot(average_reutrns, label=rf"{optname}: $\lambda_0={lr}$", linestyle=linestyle, color=c)
            if optname == 'AdamHD':
                linestyle = 'dashed'
                beta_exponent = int(math.log10(abs(adamhd_args['beta'])))
                label = rf"{optname}: $\lambda_0={lr}, \beta=10^{{{beta_exponent}}}$"
                plt.plot(average_reutrns, label=label, linestyle=linestyle, color=c)
            if optname == 'Nlarc':
                linestyle = 'solid'
                plt.plot(average_reutrns, label=rf"{optname+'m'}: $\lambda_0={{{lr}}}, \rho={{{optimizer_args['momentum']}}}$", linestyle=linestyle, color=c) # , \rho={{{nlarc_args['momentum']}}}
            if optname == 'Nlars':
                linestyle = 'dashdot'
                plt.plot(average_reutrns, label=rf"{optname+'m'}: $\lambda_0={{{lr}}}, \rho={{{optimizer_args['momentum']}}}$", linestyle=linestyle, color=c) # , \rho={{{nlars_args['momentum']}}}

            plt.legend()
            if return_type == 'training':
                plt.xlabel('Episode')
            if first_column:
                plt.ylabel('CMA of Rewards during '+ return_type.capitalize())
    plt.savefig('imgs/' + figname + '.pdf', dpi=300)
    plt.show()

##
def plot_experiment_cls_rgr(
                        # Data and Loss Configuration
                        curve_types=None,
                        momentums=None,
                        betas=None,
                        xlimit_batch=None,
                        xlimit=None,

                        # Data for Plotting
                        optimizers=None,
                        lr_rates=None,
                        maxval_limit=None,

                        # Seeds and Initialization
                        n_seeds=None,
                        use_saved_initial_seed=None,
                        experiment_name=None,
                        data_source=None,

                        # training parameters
                        validation_split=None,
                        epochs=None,
                        verbose=None,
                        batch_size=None,
                        sns_analysis_vars=None
                        ):


    seeds = get_seeds(experiment_name=experiment_name, data_source=data_source, n_seeds=n_seeds,
                      use_saved_initials_seed=use_saved_initial_seed)

    optimizers_args = setup_optimizers_args()

    training_params = {
        'epochs': epochs,
        'validation_split': validation_split,
        'verbose': verbose,
        'batch_size': batch_size
    }

    callbacks_args, callbacks = get_experiments_callbacks()

    optimizers_names = [_.__name__ for _ in optimizers]
    callbacks_name = [_.__name__ for _ in callbacks]
    min_lr = None
    for callback in callbacks:
        if 'ReduceLROnPlateau' in callbacks_name:
            min_lr = callback.minlr
    # for min_lr in min_lrs:
    for momentum in momentums:
        for beta in betas:
            for curve_type in curve_types:
                if sns_analysis_vars is None:
                    if 'Adam' in optimizers_names and 'Nlars' in optimizers_names:#momentum is not None and momentum > 0.1: # is not None:
                        first_column = True
                    else: first_column = False
                else:
                    if 'Nlars' in optimizers_names and ('k0' in sns_analysis_vars or 'momentum' in sns_analysis_vars or
                    'sigma' in sns_analysis_vars or 'lower_clip' in sns_analysis_vars or 'global_clipnorm' in sns_analysis_vars_temp):
                        first_column = True
                    else:
                        first_column = False

                if len(optimizers) > 1:
                    opts_names_lower = '_' + optimizers[0].__name__.lower() + '_' + optimizers[1].__name__.lower() + '_'
                else: opts_names_lower = '_' + optimizers[0].__name__.lower() + '_'
                if 'AdamHD' in optimizers_names:
                    figname = experiment_name + opts_names_lower + 'mu=' + str(momentum).replace('.', '_') + '_beta='\
                              + str(beta).replace('.', '_') +\
                              '_minlr=' + str(min_lr) + '_' + f_name_gen(training_params, withkey=False) + curve_type
                else:
                    figname = experiment_name + opts_names_lower + 'mu=' + str(momentum).replace('.', '_') + '_beta=' \
                              + str(None).replace('.', '_') + \
                              '_minlr=' + str(min_lr) + '_' + f_name_gen(training_params, withkey=False) + curve_type

                if sns_analysis_vars is None:
                    sns_analysis_vars_temp = {'none': [None]}
                    opt_args_loss = {'momentum': [momentum], 'beta': [beta]}
                else:
                    sns_analysis_vars_temp = sns_analysis_vars
                    opt_args_loss = {}
                for sns_var_name, sns_vars in sns_analysis_vars_temp.items():
                    if sns_analysis_vars is not None:
                        # first_column = True
                        assert sns_var_name not in ['beta', 'optimizer', 'lr', 'learning_rate']
                        opt_args_loss[sns_var_name] = sns_vars

                    plot_loss_accu(
                        # Data and Loss Configuration
                        curve_type=curve_type,

                        xlimit_batch=xlimit_batch,
                        xlimit=xlimit,
                        figname = figname,

                        # Data for Plotting
                        first_column=first_column,
                        optimizers=optimizers,
                        lr_rates=lr_rates,
                        opt_args_loss=opt_args_loss,
                        maxval_limit=maxval_limit,

                        # Seeds and Initialization
                        seeds=seeds,
                        training_params=training_params,
                        experiment_name=experiment_name,
                        optimizers_args=optimizers_args,
                        callbacks=callbacks,
                        sns_analysis_vars=sns_analysis_vars
                    )

            if 'AdamHD' not in optimizers_names:
                break

##
def plot_experiment_rl(
            # Optimization parameters
            optimizers=None,
            optimizers_args=None,
            betas=None,
            lr_rates=None,
            min_lrs=None,
            momentums=None,

            # Training parameters
            training_params=None,
            seeds=None,

            # Evaluation parameters
            return_type=None,
            callbacks=None,

            # Configuration parameters
            experiment_name=None,
            game_name=None,
            curve_type=None
    ):
    optimizers_names = [_.__name__ for _ in optimizers]
    nlarc_args = optimizers_args['nlarc_args']
    nlars_args = optimizers_args['nlars_args']
    adamhd_args = optimizers_args['adamhd_args']
    adam_args = optimizers_args['adam_args']
    for min_lr in min_lrs:
        for momentum in momentums:
            nlarc_args['momentum'] = momentum
            nlars_args['momentum'] = momentum
            for beta in betas:
                adamhd_args['beta'] = beta
                # if momentum > 0.1:
                #     first_column = False
                if momentum is not None and 'Nlars' in optimizers_names and 'Adam' in optimizers_names:
                    first_column = True
                else: first_column = False
                if len(optimizers) > 1:
                    opts_names = '_' + optimizers[0].__name__.lower() + '_' + optimizers[1].__name__.lower() + '_'
                else:
                    opts_names = '_' + optimizers[0].__name__.lower() + '_'
                if 'AdamHD' in optimizers_names:
                    figname = experiment_name + opts_names + 'mu=' + str(momentum).replace('.', '_') + '_beta=' \
                              + str(beta).replace('.', '_') + \
                              '_minlr=' + str(min_lr) + '_' + game_name + '_' + curve_type + '_' + return_type
                else:
                    figname = experiment_name + opts_names + 'mu=' + str(momentum).replace('.', '_') + '_beta=' \
                              + str(None).replace('.', '_') + \
                              '_minlr=' + str(min_lr) + '_' + game_name + '_' + curve_type + '_' + return_type

                plot_return_rl(optimizers=optimizers, return_type=return_type, betas=[beta], figname=figname,
                               training_params=training_params, adam_args=adam_args,
                               nlarc_args=nlarc_args, nlars_args=nlars_args, adamhd_args=adamhd_args, lr_rates=lr_rates,
                               seeds=seeds,
                               experiment_name=experiment_name, first_column=first_column,
                               callbacks=callbacks)
                if 'AdamHD' not in optimizers_names:
                    break
