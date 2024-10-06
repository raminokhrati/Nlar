# Editing and overwriting of Tensorflow and Keras codes are gratefully acknowledged.
# Author: Ramin OKhrati

import tensorflow as tf

import warnings

def custom_formatwarning(msg, *_args, **_kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

##

class AdamHD(tf.keras.optimizers.Optimizer):

    r"""Optimizer that implements the AdamHD algorithm."""

    __name__ = 'AdamHD'
    def __init__(
        self,
        learning_rate=0.001,
        beta=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="AdamHD",
        **kwargs
    ):
        super().__init__(
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs
        )

        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta = beta
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

    def build(self, var_list):

        """Initialize optimizer variables."""

        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    )
          )

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]

        prev_bias_correction1 = 1 - tf.pow(tf.cast(self.beta_1, variable.dtype), local_step - 1)
        prev_bias_correction2 = 1 - tf.pow(tf.cast(self.beta_2, variable.dtype), local_step - 1)
        h = tf.cond(local_step > 1, true_fn=lambda: (tf.tensordot(tf.reshape(gradient, shape=[-1]),
                             tf.reshape(m / (tf.sqrt(v) + self.epsilon), shape=[-1]), axes=1)
                                                     * tf.sqrt(prev_bias_correction2) / prev_bias_correction1),
                             false_fn=lambda: tf.cast(0, variable.dtype))
        # A similar way of calculating h
        #     pre_h = m / (tf.sqrt(v) + self.epsilon)
        #     h = tf.cond(local_step>1, true_fn=lambda:
        #     tf.reduce_sum([ tf.reduce_sum(gr * mv) for gr, mv in zip(gradient, pre_h)])
        #     * tf.sqrt(prev_bias_correction2) / prev_bias_correction1, false_fn=lambda: tf.cast(0, variable.dtype))

        self.learning_rate.assign_add(self.beta * h)

        lr = tf.cast(self.learning_rate, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)

        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2),
                    gradient.indices,
                )
            )
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        else:
            # Dense gradients.
            m.assign_add((gradient - m) * (1 - self.beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                "beta": self.beta
            }
        )
        return config


##
