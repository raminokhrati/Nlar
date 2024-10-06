
import tensorflow as tf
from experiments.src_epxeriments.utils.utils_general import custome_mean_loss
from tensorflow.keras.callbacks import ModelCheckpoint


##


class BatchLossCallback(tf.keras.callbacks.Callback):
    __name__ = 'BatchLossCallback'
    """This class takes a tensor data and a boolian training variable and save or print out the batch losses.
       If training=True, then losses are recorded as the model has been trained. This parameter
       make a difference if there are layers like dropout or batch normalization, for which if training=True,
       then the losses are recorded as the training is carried out. However, if training=False, then
       the model is in inference mode, and for statistical the effect of dropout is not considered.
       Hence, for inferences, it is better to use training=False. If verbose >=2, then the average of losses
       are printed out. This is the same as log of tf.keras which also prints the average of losses."""

    def __init__(self, name, data, training):
        super(BatchLossCallback, self).__init__()
        # assert isinstance(data, tf.data.Dataset)

        self.data = data
        self.data_iterator = iter(data)
        self.training = training
        self.batch_losses = []
        self.name = name
        self.lers = []

    def on_batch_begin(self, batch, logs=None):
        try:
            x_batch, y_batch = next(self.data_iterator)
            # Calculate the loss
            x_batch = tf.reshape(x_batch, (-1, *x_batch.shape[1:]))
            predictions = self.model(x_batch, training=self.training)

            loss_temp = self.model.loss(y_batch, predictions)
            self.batch_losses.append(loss_temp.numpy())

            if self.params['verbose'] >= 2:
                print(f'The average loss up to batch {batch} is: {tf.reduce_mean(self.batch_losses).numpy():.4f}')

        except StopIteration:

            self.data_iterator = iter(self.data)

class BatchLerCallback(tf.keras.callbacks.Callback):
    __name__ = 'BatchLerCallback'
    """To return learning rates at the end of each batch."""

    def __init__(self, name):
        super(BatchLerCallback, self).__init__()
        self.name = name
        self.lers = []

    def on_batch_end(self, batch, logs=None):
        if self.model.optimizer.__name__ in ['Nlarc', 'Nlars']:
            ler_temp = [tf.identity(_) for _ in self.model.optimizer._lers]
        else: ler_temp = self.model.optimizer.lr
        self.lers.append(ler_temp)

        if self.params['verbose'] >= 2:
            print(f'The average of all learning rates in batch {batch} is: '
                  f'{tf.reduce_mean([tf.reduce_mean(_).numpy() for _ in ler_temp]):.4f}')


class BatchrhoCallback(tf.keras.callbacks.Callback):
    """To obtain dynamic momentums"""
    __name__ = 'BatchrhoCallback'
    """To return momentums at the end of each batch if they are stored by the optimizer."""

    def __init__(self, name):
        super(BatchrhoCallback, self).__init__()

        self.name = name
        self.rhos = []

    def on_batch_end(self, batch, logs=None):
        if self.model.optimizer.__name__ in ['Nlarc', 'Nlars']:
            rho_temp = [tf.identity(_) for _ in self.model.optimizer._rhos]
        else:
            raise ValueError(f'Optimizer {self.model.optimizer.__name__} does not have a dynamic estimation of'
                             f'momentums')
        self.rhos.append(rho_temp)

        if self.params['verbose'] >= 2:
            print(f'The average of all learning rates in batch {batch} is: '
                  f'{tf.reduce_mean([tf.reduce_mean(_).numpy() for _ in rho_temp]):.4f}')


class EpochLossCallback(tf.keras.callbacks.Callback):
    __name__ = 'EpochLossCallback'

    """This callback records losses on a tensor data. It can do both inference and non inference mode.
    In inference mode, we have training=False."""
    def __init__(self, name, data, training, loss):
        super(EpochLossCallback, self).__init__()

        self.data = data
        self.training = training
        self.loss = loss
        self.epoch_losses_begin = []
        self.epoch_losses_end = []
        self.name = name

    def on_epoch_begin(self, epoch, logs=None):

        # Calculate the loss
        if isinstance(self.data, tf.data.Dataset):
            loss_temp = custome_mean_loss(self.data, self.model, self.model.loss, self.training)
        else:
            dataset_tensor = tf.data.Dataset.from_tensor_slices(self.data)
            dataset_tensor = dataset_tensor.shuffle(buffer_size=10000).batch(300).prefetch(tf.data.AUTOTUNE)
            loss_temp = custome_mean_loss(dataset_tensor, model=self.model, loss=self.model.loss, training=self.training)

        self.epoch_losses_begin.append(loss_temp)
        if self.params['verbose'] >= 2:
            print(f'The loss at the beginning of epoch {epoch} is: {loss_temp:.4f}')

    def on_epoch_end(self, epoch, logs=None):
        # Calculate the loss
        if isinstance(self.data, tf.data.Dataset):
            loss_temp = custome_mean_loss(self.data, self.model, self.model.loss, self.training)
        else:
            dataset_tensor = tf.data.Dataset.from_tensor_slices(self.data)
            dataset_tensor = dataset_tensor.shuffle(buffer_size=10000).batch(300).prefetch(tf.data.AUTOTUNE)
            loss_temp = custome_mean_loss(dataset_tensor, model=self.model, loss=self.model.loss, training=self.training)

        self.epoch_losses_end.append(loss_temp)
        if self.params['verbose'] >= 2:
            print(f'The loss at the end of epoch {epoch} is: {loss_temp:.4f}')


def get_experiments_callbacks():
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
