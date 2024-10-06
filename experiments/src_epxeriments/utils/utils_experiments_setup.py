
import tensorflow as tf
import random

from src.utils.utils_callbacks import get_experiments_callbacks
from src.utils.utils_general import save_obj, load_obj, parallel_computing_strategy
from src.models import model_generate
from src.data.load_data import get_data
##

def get_seeds(experiment_name=None, data_source=None, n_seeds=None, use_saved_initials_seed=None):
    # seed selection
    if not use_saved_initials_seed:
        seeds = []
        for _ in range(n_seeds):
            if not use_saved_initials_seed:
                seeds.append(random.randint(0, 2 ** 32 - 1))
        save_obj(seeds, 'initial_parameters/seeds/' + experiment_name + '/' + data_source + tf.keras.backend.floatx())
    else:
        seeds = load_obj('initial_parameters/seeds/' + experiment_name + '/' + data_source + tf.keras.backend.floatx())
    return seeds

##
def setup_optimizers_args():
    # optimizers' parameters: the critical parameters are set to be None as they change through the experiments

    nlars_args = {
        'learning_rate': None,
        'momentum': None,
        'sigma': 1e-30,
        'k0': 1,
        'global_clipnorm': 1,
        'lower_clip': 1e-150
    }
    nlarc_args = {
        'learning_rate': None,
        'momentum': None,
        'sigma': 1e-30,
        'k0': 1,
        'global_clipnorm': 1
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
        'learning_rate': None,
        'beta': None,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-08,
        'global_clipnorm': 1
    }

    adagrad_args = {
        'learning_rate': None,
        'global_clipnorm': 1
    }



    optimizers_args = dict(adam_args=adam_args,
                           adamhd_args=adamhd_args,
                           nlarc_args=nlarc_args,
                           nlars_args=nlars_args,
                           adagrad_args=adagrad_args)
    return optimizers_args
##

def setup_experiment(
                    # experiment specifications
                    experiment_name=None,
                    data_source=None,
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

                    # general parameters
                    use_saved_initials_seed=None,
                    use_saved_initial_weights=None,

                    # number of seeds
                    n_seeds=None
                    ):

    # gpu and computing strategy setups
    _, strategy = parallel_computing_strategy(m_threshold=m_threshold, cpu_parallel=cpu_parallel, gpu_parallel=gpu_parallel)

    # training parameters
    training_params = {
        'epochs': epochs,
        'validation_split': validation_split,
        'verbose': verbose,
        'batch_size': batch_size
    }

    # load data
    x_train, y_train, x_test, y_test = get_data(data_source=data_source)

    if experiment_name == 'cifar10_vgg16':
        data_config = (-1, 32, 32, 3)
        network_type = 'cnn-vgg16'
        mlp_conf = None
    elif experiment_name == 'cifar10_vgg11':
        data_config = (-1, 32, 32, 3)
        network_type = 'cnn-vgg11'
        mlp_conf = None

    elif experiment_name == 'mnist_logistic':
        n_features = 28 * 28
        data_config = (-1, n_features)
        network_type = 'mlp'
        mlp_conf = (n_features, 10)
    elif experiment_name == 'mnist_mlp':
        n_features = 28 * 28
        data_config = (-1, n_features)
        network_type = 'mlp'
        mlp_conf = (n_features, 1000, 1000, 10)
    elif experiment_name == 'cifar10_mlp7h':
        n_features = 32 * 32 * 3
        data_config = (-1, n_features)
        network_type = 'mlp'
        mlp_conf = (n_features, 512, 512, 512, 512, 512, 512, 512, 10)
    else: raise ValueError("The experiment is not recognized.")

    model_conf = dict(network_type=network_type, mlp_conf=mlp_conf, lambda_value=lambda_value,
                      use_bias_input=use_bias_input,
                      seed=None)

    seeds = get_seeds(experiment_name, data_source, n_seeds, use_saved_initials_seed)

    if experiment_name in ['cifar10_vgg11', 'cifar10_vgg16', 'mnist_logistic', 'mnist_mlp', 'cifar10_mlp7h']:
        # determine loss, metrics, and initial weight generation
        if strategy is None:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False,
                reduction=tf.keras.losses.Reduction.SUM) # other reductions are available: SUM, Auto, None
        metrics = ['accuracy']
    else:
        raise ValueError(f"The experiment {experiment_name} is not recognized.")

    for seed in seeds:
        if not use_saved_initial_weights:
            model_conf['seed'] = seed
            model = model_generate(**model_conf)

            model.compile(loss=loss)
            model_orig_w = model.get_weights()

            save_obj(model_orig_w,
                     'initial_parameters/init_weights/' + experiment_name + '/' + "model_orig_w_" + data_source + str(
                         seed))

    optimizers_args = setup_optimizers_args()

    callbacks_args, callbacks = get_experiments_callbacks()

    return (x_train, y_train, x_test, y_test, data_config, model_conf, training_params,
            seeds, callbacks, callbacks_args, strategy, optimizers_args, loss, metrics)
