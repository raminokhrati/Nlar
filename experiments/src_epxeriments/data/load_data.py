
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

##

def get_data(data_source=None):

    """This function loads different datasets."""

    float_precision = tf.keras.backend.floatx()

    if data_source == 'cifar10':
        # data and preprocessing
        (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = tf.keras.datasets.cifar10.load_data()

    elif data_source == 'mnist':
        (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = tf.keras.datasets.mnist.load_data()
    else:
        raise ValueError("The data source is not recognized.")

    y_train_orig = np.hstack(y_train_orig)
    y_test_orig = np.hstack(y_test_orig)

    y_train = y_train_orig.astype(float_precision)
    y_test = y_test_orig.astype(float_precision)

    # Normalize pixel values between 0 and 1
    x_train = x_train_orig.astype(float_precision) / 255.0
    x_test = x_test_orig.astype(float_precision) / 255.0

    return x_train, y_train, x_test, y_test


def shuffle_data_with_seed(
        # Dataset parameters
        x_train=None,
        y_train=None,
        x_test=None,
        y_test=None,

        # Configuration and preprocessing
        data_config=None,
        validation_split=None,
        batch_size=None,

        # Random seed for reproducibility
        seed=None
):

    """This function takes, x_train, y_train, x_test, y_test and produces datasets
    train_dataset, val_dataset, test_dataset using random shuffling. Note that even if validation_split is zero,
    the configuration of the dataset would not be the same as the old ones."""

    tf.random.set_seed(seed)

    if seed is not None:
        # Combine the training and testing sets
        x_combined = np.concatenate((x_train, x_test), axis=0)
        y_combined = np.concatenate((y_train, y_test), axis=0)

        # Reshape the input data to a vector
        x_combined = x_combined.reshape(x_combined.shape[0], -1)

        # Split the data into training and testing sets with the same number of samples
        x_train, x_test, y_train, y_test = train_test_split(x_combined, y_combined, train_size=len(x_train),
                                                            random_state=seed)

        x_train = x_train.reshape(data_config)
        x_test = x_test.reshape(data_config)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

    # Split the dataset into training and validation sets
    if validation_split > 0:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_split,
                                                          random_state=seed)
        x_train = x_train.reshape(data_config)
        x_val = x_val.reshape(data_config)
        y_train = np.array(y_train)
        y_val = np.array(y_val)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        # apply batching and prefetching to the validation dataset
        val_dataset = val_dataset.batch(batch_size=batch_size)
        val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        # making test dataset tensors
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        # apply shuffling, batching, and prefetching to the testing dataset
        test_dataset = test_dataset.shuffle(buffer_size=len(x_train), seed=seed)
        test_dataset = test_dataset.batch(batch_size=batch_size)
        test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    else:
        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        # apply batching and prefetching to the validation dataset
        val_dataset = val_dataset.batch(batch_size=batch_size)
        val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        x_val = x_test
        y_val = y_test

        x_test = None
        y_test = None
        test_dataset = None

    # making train dataset tensors
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    # apply shuffling, batching, and prefetching to the training dataset
    train_dataset = train_dataset.shuffle(buffer_size=len(x_train), seed=seed)
    train_dataset = train_dataset.batch(batch_size=batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return {'train_dataset': train_dataset, 'val_dataset': val_dataset, 'test_dataset': test_dataset,
            'x_train': x_train, 'x_test': x_test, 'x_val': x_val, 'y_train': y_train, 'y_test': y_test,
            'y_val': y_val}
