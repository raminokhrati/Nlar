
import tensorflow as tf

from tensorflow.keras.layers import Dense, BatchNormalization

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

##
class MLP(tf.keras.Model):

    def __init__(self, layers, lambda_value=0, use_bias_input=False, is_regression=False, seed=None):
        assert seed is not None


        tf.random.set_seed(seed)
        super(MLP, self).__init__()
        self.from_layer = 0
        self.seq = []
        self.lambda_value = lambda_value
        self.is_regression = is_regression

        input_size = layers[0]
        for layer, output_size in enumerate(layers[1:]):
            if layer == len(layers) - 2:
                if output_size == 1:
                    active_func = 'linear' if self.is_regression else 'sigmoid'
                else: active_func = 'linear' if self.is_regression else 'softmax'

                self.seq.append(Dense(output_size,
                                      activation= active_func,
                                      input_shape=(input_size,),
                                      use_bias=use_bias_input,
                                      dtype=tf.keras.backend.floatx(), kernel_regularizer=tf.keras.regularizers.l2(self.lambda_value)))
            elif layer == 0:
                self.seq.append(Dense(output_size,
                                      activation='relu',
                                      input_shape=(input_size,),
                                      use_bias=use_bias_input,
                                      dtype=tf.keras.backend.floatx(), kernel_regularizer=tf.keras.regularizers.l2(self.lambda_value)))

            else:
                self.seq.append(Dense(output_size,
                                      activation='relu',
                                      input_shape=(input_size,), use_bias=use_bias_input,
                                      dtype=tf.keras.backend.floatx(), kernel_regularizer=tf.keras.regularizers.l2(self.lambda_value)))

            input_size = output_size
            # Initialize the model's variables
            self.build(input_shape=(None, layers[0]))

    def set_from_layer(self, layer=0):
        self.from_layer = layer

    def call(self, x):
        for layer, fc in enumerate(self.seq):
            if layer >= self.from_layer:
                x = fc(x)
        return x

##

def vgg11(lambda_value=0, use_bias_input=None, seed=None):
    assert seed is not None
    tf.random.set_seed(seed)

    l2_reg = tf.keras.regularizers.l2(l2=lambda_value)

    model = tf.keras.Sequential()
    # Block 1
    model.add(
    layers.Conv2D(64, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg, padding='same',
                  input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    #
    # Block 2
    model.add(layers.Conv2D(128, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg,
                            padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    # Block 3
    model.add(layers.Conv2D(256, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg,
                            padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg,
                            padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    # Block 4
    model.add(layers.Conv2D(512, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg,
                            padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg,
                            padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    # Block 5
    model.add(layers.Conv2D(512, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg,
                            padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg,
                            padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    # Flatten
    model.add(layers.Flatten())

    # Fully connected layers with 4096 nodes of the first two feedforward layers
    model.add(layers.Dense(4096, activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg))
    model.add(layers.Dense(4096, activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg))
    model.add(layers.Dense(10, activation='softmax', use_bias=use_bias_input, kernel_regularizer=l2_reg))

    model.build((None, 32, 32, 3))
    return model
##

def vgg16(lambda_value=0, use_bias_input=None, seed=None):
    assert seed is not None
    tf.random.set_seed(seed)

    l2_reg = tf.keras.regularizers.l2(l2=lambda_value)

    model = tf.keras.Sequential()
    # Block 1
    model.add(
    layers.Conv2D(64, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg, padding='same',
                  input_shape=(32, 32, 3)))
    layers.Conv2D(64, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg, padding='same')

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    #
    # Block 2
    model.add(layers.Conv2D(128, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg,
                            padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg,
                            padding='same'))

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    # Block 3
    model.add(layers.Conv2D(256, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg,
                            padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg,
                            padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg,
                            padding='same'))

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    # Block 4
    model.add(layers.Conv2D(512, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg,
                            padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg,
                            padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg,
                            padding='same'))

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    # Block 5
    model.add(layers.Conv2D(512, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg,
                            padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg,
                            padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg,
                            padding='same'))

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    # Flatten
    model.add(layers.Flatten())

    # Fully connected layers with 4096 nodes of the first two feedforward layers
    model.add(layers.Dense(4096, activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg))
    model.add(layers.Dense(4096, activation='relu', use_bias=use_bias_input, kernel_regularizer=l2_reg))
    model.add(layers.Dense(10, activation='softmax', use_bias=use_bias_input, kernel_regularizer=l2_reg))

    model.build((None, 32, 32, 3))
    return model

##
def model_generate(
    network_type=None,
    mlp_conf=None,
    lambda_value=None,
    use_bias_input=None,
    is_regression=False,
    seed=None
):
    assert seed is not None and network_type is not None
    tf.random.set_seed(seed)

    if network_type == 'mlp':
        assert mlp_conf is not None
        return MLP(
            mlp_conf,
            lambda_value=lambda_value,
            use_bias_input=use_bias_input,
            is_regression=is_regression,
            seed=seed
        )

    elif network_type == 'cnn-vgg11':
        return vgg11(
            lambda_value=lambda_value,
            use_bias_input=use_bias_input,
            seed=seed
        )

    elif network_type == 'cnn-vgg16':
        return vgg16(
            lambda_value=lambda_value,
            use_bias_input=use_bias_input,
            seed=seed
        )


    else:
        raise ValueError("The model is not defined.")
