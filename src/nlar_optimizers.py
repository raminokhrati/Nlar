# Editing and overwriting of Tensorflow and Keras codes are gratefully acknowledged.
# Author: Ramin OKhrati

import tensorflow as tf

import warnings

def custom_formatwarning(msg, *_args, **_kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

class Nlarc(tf.keras.optimizers.Optimizer):
    r"""
    Optimizer that implements the Nlarc/m algorithm.
    If momentum=0 or momentum=None, the optimizer becomes Nlarc. For a non-zero value of momentum, it is Nlarcm.
    """
    __name__ = 'Nlarc'
    def __init__(
        self,
        learning_rate=0.1,
        momentum=1,
        k0= 0.1,
        sigma= 1e-19,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=1,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="Nlarc",
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

        self._learning_rate = self._build_learning_rate(learning_rate) # learning_rate
        self.learning_rate_ = tf.Variable(learning_rate, dtype=tf.keras.backend.floatx()) # for internal usage

        self.momentum = momentum
        self.k0 = k0
        self.sigma = sigma
        self.amsgrad = amsgrad

        if self.momentum == 0:
            warnings.formatwarning = custom_formatwarning
            warnings.simplefilter("always")  # Ensure the warning is always shown
            warnings.warn(f"Warning.........."
                          f"The value of momentum is zero. This is interpreted as no momentum at all, "
                          f"which slows down the convergence. If this is intended, "
                          f"for a more efficient/stable convergence, please set momentum to be None.")
            warnings.simplefilter("default")  # Reset to the default behavior

        if self.global_clipnorm is None:
            warnings.formatwarning = custom_formatwarning
            warnings.simplefilter("always")  # Ensure the warning is always shown
            warnings.warn(f"Warning.........."
                          f"The global_clipnorm is set to None. In this case, Nlarc might not converge.")
            warnings.simplefilter("default")  # Reset to the default behavior

        self._current_precision = tf.keras.backend.floatx()

        if self._current_precision < 'float64':
            warnings.formatwarning = custom_formatwarning
            warnings.simplefilter("always")  # Ensure the warning is always shown
            warnings.warn(f"Warning.........."
                          f"Nlarc is optimized using float precision accuracy of 64. For a better performance, please "
                          f"change the float precision.")
            warnings.simplefilter("default")  # Reset to the default behavior

    def __max_precision(self):
        if self._current_precision == 'float16':
            return tf.float16, tf.float16.max
        elif self._current_precision == 'float32':
            return tf.float32, tf.float32.max
        elif self._current_precision == 'float64':
            return tf.float64, tf.float64.max
        else:
            raise ValueError(f"The floating precision {self._current_precision} is not recognized, "
                             "please use one of the following floating precessions: 16, 32, 64")

    def build(self, var_list):

        """Initialize optimizer variables.

        Nlarc/m optimizer has 6 types of variables: velocities, grad_accus, graddelta_accus
        lers, sigma_vars, and _velocity_hats (only if amsgrad is set True)

        Args:
          var_list: list of model variables to build Nlarc/m variables on.
        """

        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._velocities = []
        self._grad_accus = []
        self._graddelta_accus = []
        self._lers = []
        self._sigma_vars = []

        for var in var_list:

            initial_value_grad_accus = tf.zeros_like(var, dtype=var.dtype)
            initial_value_graddelta_accus = tf.zeros_like(var, dtype=var.dtype)
            initial_value_velocities = tf.zeros_like(var, dtype=var.dtype)
            initial_value_sigma_vars = tf.random.uniform(tf.shape(var), -tf.cast(tf.sqrt(3.), var.dtype),
                                                         tf.cast(tf.sqrt(3.), var.dtype),
                                                         dtype=var.dtype)
            # Other forms of initialization could be used such as truncated normal:
            # initial_value_sigma_vars = tf.random.truncated_normal(tf.shape(var), tf.cast(0, var.dtype),
            #                                              tf.cast(1, var.dtype),
            #                                              dtype=var.dtype)

            initial_value_lers = tf.ones_like(var, dtype=var.dtype) * self.learning_rate

            self._lers.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="ler",
                    initial_value=initial_value_lers
                )
            )

            self._grad_accus.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="grad_accu",
                    initial_value=initial_value_grad_accus
                )
            )

            self._graddelta_accus.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="graddelta_accu",
                    initial_value=initial_value_graddelta_accus
                )
            )

            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="vel",
                    initial_value=initial_value_velocities
                )
            )

            self._sigma_vars.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="sigma_var",
                    initial_value=initial_value_sigma_vars
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

        """Update step given gradient and the associated model variables."""

        sigma_tensor = tf.where(tf.greater(tf.abs(gradient),0),
                                tf.minimum(tf.cast(self.sigma, dtype=variable.dtype),
                                           tf.abs(gradient)), tf.cast(self.sigma, dtype=variable.dtype))

        local_step = tf.cast(self.iterations + 1, variable.dtype)
        var_key = self._var_key(variable)
        vel = self._velocities[self._index_dict[var_key]]
        ler = self._lers[self._index_dict[var_key]]

        grad_accu = self._grad_accus[self._index_dict[var_key]]
        graddelta_accu = self._graddelta_accus[self._index_dict[var_key]]
        sigma_var = self._sigma_vars[self._index_dict[var_key]]

        # This is to initiate learning rate schedule
        ler.assign(tf.cond(tf.greater(tf.cast(self.learning_rate_, variable.dtype),
                                      tf.cast(self.learning_rate, variable.dtype)),
                           true_fn=lambda: tf.multiply(tf.cast(self.learning_rate, variable.dtype), ler),
                           false_fn=lambda: ler))

        if isinstance(gradient, tf.IndexedSlices):
            warnings.formatwarning = custom_formatwarning
            warnings.simplefilter("always")  # Ensure the warning is always shown
            warnings.warn(f"Warning.........."
                            "Sparse gradients are not yet efficiently implemented in Nlarc. "
                            "This may cause the running process to be slow."
            )
            warnings.simplefilter("default")  # Reset to the default behavior

        pre_var = tf.identity(variable)

        if self.momentum is not None:

            # For float32; if sigma is too small, there could be an overflow. Hence, sigma needs to be capped
            # appropriately.
            positive_sigma2_inf_mask = tf.math.is_inf(tf.pow(tf.cast(self.sigma, dtype=variable.dtype), -2))
            def false_fn():
                return tf.divide(tf.multiply(tf.pow(tf.cast(self.sigma, dtype=variable.dtype), -2),
                                 tf.pow(sigma_tensor, 2)), local_step)
            def true_fn():
                _, max_precision = self.__max_precision()
                # to avoid overflow we have capped the max precision by a factor of 0.01
                return tf.divide(tf.multiply(
                    tf.clip_by_value(tf.pow(tf.cast(self.sigma, dtype=variable.dtype), -2), 0,
                                     max_precision * 0.01), tf.pow(sigma_tensor, 2)), local_step)

            # Note that if some of the gradients are very small, sigma_tensor^2 could be nearly
            # zero; which makes m_t=0. This especially affects (tf.abs(vel) + m_t) in rho_t. In particular,
            # if the precision is set to float32 and sigma very small; this could cause for elements of sigma_tensor
            # which depend on the square of sigma to become zero. This affects rho_t.
            m_t = tf.cond(positive_sigma2_inf_mask, true_fn, false_fn)
            any_zero_velmt = tf.reduce_any(tf.equal(tf.add(tf.abs(vel), m_t), 0))

            rho_t = tf.cond(any_zero_velmt,
                            lambda: tf.divide(tf.multiply(tf.divide(tf.cast(self.momentum, dtype=variable.dtype),
                            tf.add(tf.constant(1, dtype=variable.dtype), tf.abs(ler))), m_t),
                                              tf.add(tf.abs(vel),
                                                     tf.add(m_t, tf.constant(1e-10, dtype=variable.dtype)))),
                            lambda: tf.divide(tf.multiply(tf.divide(tf.cast(self.momentum, dtype=variable.dtype),
                                               tf.add(tf.constant(1, dtype=variable.dtype),
                                                      tf.abs(ler))), m_t),
                                              tf.add(tf.abs(vel), m_t))
                            )

            variable.assign_add(tf.multiply(rho_t, vel))
            variable.assign_sub(tf.multiply(ler, gradient))
            variable.assign_add(tf.multiply(sigma_tensor, sigma_var))

            vel.assign(tf.multiply(rho_t, vel))
            vel.assign_sub(tf.multiply(ler, gradient))
            # or similarly
            # vel.assign(rho_t * vel - ler * gradient)

        else:
            variable.assign_sub(tf.multiply(gradient, ler))
            variable.assign_add(tf.multiply(sigma_tensor, sigma_var))

        delta_params = tf.subtract(variable, pre_var)
        positive_inf_mask = tf.math.is_inf(tf.pow(sigma_tensor, -2))
        dtype, max_precision = self.__max_precision()
        sigma_tensor_square = tf.where(positive_inf_mask, tf.constant(max_precision * 0.01, dtype=dtype),
                                       tf.pow(sigma_tensor, -2))

        sigma_tensor1 = tf.multiply(sigma_tensor_square, tf.pow(gradient, 2))
        sigma_tensor2 = tf.multiply(sigma_tensor_square, tf.multiply(gradient, delta_params))

        grad_accu.assign_add(sigma_tensor1)
        graddelta_accu.assign_add(sigma_tensor2)
        grad_accu_temp = tf.add(tf.cast(self.k0, dtype=variable.dtype), grad_accu)
        graddelta_accu_temp = tf.subtract(tf.multiply(self.learning_rate, tf.cast(self.k0, dtype=variable.dtype)),
                                          graddelta_accu)

        ler.assign(tf.divide(graddelta_accu_temp, grad_accu_temp))

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "momentum": self.momentum,
                "sigma": self.sigma,
                "amsgrad": self.amsgrad,
                "k0": self.k0
            }
        )
        return config

##
class Nlars(tf.keras.optimizers.Optimizer):

    r"""
    Optimizer that implements the Nlars/m algorithm.
    If momentum=0 or momentum=None, the optimizer becomes Nlarc. For a non-zero value of momentum, it is Nlarcm.
    """

    __name__ = 'Nlars'
    def __init__(
        self,
        learning_rate=0.1,
        momentum=1,
        k0=0.1,
        sigma=1e-19,
        lower_clip=1e-150,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=1,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="Nlars",

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

        self._learning_rate = self._build_learning_rate(learning_rate)  # learning_rate
        self.learning_rate_ = tf.Variable(learning_rate, dtype=tf.keras.backend.floatx()) # for internal usage

        self.momentum = momentum
        self.k0 = k0
        self.sigma = sigma
        # self.seed = seed
        self.amsgrad = amsgrad
        self.lower_clip = lower_clip
        if self.global_clipnorm is None:
            warnings.formatwarning = custom_formatwarning
            warnings.warn(f"Warning..........The global_clipnorm is set to None. In this case, Nlars might not converge.")

        self._current_precision = tf.keras.backend.floatx()

        if self.momentum == 0:
            warnings.formatwarning = custom_formatwarning
            warnings.simplefilter("always")  # Ensure the warning is always shown
            warnings.warn(f"Warning.........."
                          f"The value of momentum is zero. This is interpreted as no momentum at all, "
                          f"which slows down the convergence. If this is intended, "
                          f"for a more efficient/stable convergence, please set momentum to be None.")
            warnings.simplefilter("default")  # Reset to the default behavior


    def __min_positive_float(self):
        if self._current_precision == 'float16':
            return tf.experimental.numpy.finfo(tf.float16).tiny
        elif self._current_precision == 'float32':
            return tf.experimental.numpy.finfo(tf.float32).tiny
        elif self._current_precision == 'float64':
            return tf.experimental.numpy.finfo(tf.float64).tiny
        else:
            raise ValueError(f"The floating precision {self._current_precision} is not recognized, "
                             "please use one of the following floating precessions: 16, 32, 64")

    def build(self, var_list):

        """Initialize optimizer variables.

        Nlars optimizer has 5 types of variables: velocities, grad_accus, graddelta_accus
        lers, and velocity_hat (only set when amsgrad is applied),

        Args:
          var_list: list of model variables to build Nlar variables on.
        """

        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._velocities = []
        self._rhos = []
        self._grad_accus = []
        self._graddelta_accus = []
        self._lers = []
        self._sigma_vars = []

        for var in var_list:
            initial_value_grad_accus = tf.zeros_like(var, dtype=var.dtype)
            initial_value_graddelta_accus = tf.zeros_like(var, dtype=var.dtype)
            initial_value_velocities = tf.zeros_like(var, dtype=var.dtype)
            initial_value_sigma_vars = tf.random.uniform(tf.shape(var), -tf.cast(tf.sqrt(3.), var.dtype),
                                                         tf.cast(tf.sqrt(3.), var.dtype),
                                                         dtype=var.dtype)

            initial_value_lers = tf.ones_like(var, dtype=var.dtype) * self._learning_rate
            self._lers.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="ler",
                    initial_value=initial_value_lers
                )
            )

            self._grad_accus.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="grad_accu",
                    initial_value=initial_value_grad_accus
                )
            )

            self._graddelta_accus.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="graddelta_accu",
                    initial_value=initial_value_graddelta_accus
                )
            )

            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="vel",
                    initial_value=initial_value_velocities
                )
            )

            self._sigma_vars.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="sigma_var",
                    initial_value=initial_value_sigma_vars
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

        smallest_normal_float = self.__min_positive_float()
        lower_clip_temp = tf.cond(tf.less(tf.cast(self.lower_clip, variable.dtype), smallest_normal_float),
                                  lambda: tf.cast(smallest_normal_float * 10, variable.dtype),
                                  lambda: tf.cast(self.lower_clip, variable.dtype))
        lower_clip_temp = tf.cast(lower_clip_temp, variable.dtype)

        gradient = tf.where(tf.greater(tf.abs(gradient), lower_clip_temp), gradient, tf.math.sign(gradient) * lower_clip_temp)

        local_step = tf.cast(self.iterations + 1, variable.dtype)
        var_key = self._var_key(variable)
        vel = self._velocities[self._index_dict[var_key]]

        ler = self._lers[self._index_dict[var_key]]

        grad_accu = self._grad_accus[self._index_dict[var_key]]
        graddelta_accu = self._graddelta_accus[self._index_dict[var_key]]
        sigma_var = self._sigma_vars[self._index_dict[var_key]]

        ler.assign(tf.cond(tf.greater(self.learning_rate_, self.learning_rate),
                           true_fn=lambda: self.learning_rate * ler,
                           false_fn=lambda: ler))

        if isinstance(gradient, tf.IndexedSlices):
            warnings.formatwarning = custom_formatwarning
            warnings.simplefilter("always")  # Ensure the warning is always shown
            warnings.warn(f"Warning.........."
                "Sparse gradients are not yet efficiently implemented in Nlarc. "
                "This may cause the running process to be slow."
            )
            warnings.simplefilter("default")  # Reset to the default behavior

        pre_var = tf.identity(variable)

        if self.momentum is not None:

            m_t = 1 / local_step
            rho_t = tf.divide(tf.multiply(tf.divide(tf.cast(self.momentum, dtype=variable.dtype),
                                                    (1 + tf.abs(ler))), m_t),
                              (tf.add(tf.abs(vel), m_t + tf.cast(self.sigma, dtype=variable.dtype))))

            variable.assign_add(tf.multiply(rho_t, vel))
            variable.assign_sub(tf.multiply(ler, gradient))
            variable.assign_add(tf.cast(self.sigma, dtype=variable.dtype) * sigma_var)

            vel.assign(tf.multiply(rho_t, vel))
            vel.assign_sub(tf.multiply(ler, gradient))
        else:
            variable.assign_sub(tf.multiply(gradient, ler))
            variable.assign_add(tf.cast(self.sigma, dtype=variable.dtype) * sigma_var)

        delta_params = variable - pre_var

        sigma_tensor1 = tf.pow(gradient, 2)
        sigma_tensor2 = tf.multiply(gradient, delta_params)

        grad_accu.assign_add(sigma_tensor1)
        graddelta_accu.assign_add(sigma_tensor2)

        grad_accu_temp = tf.cast(self.k0, dtype=variable.dtype) + grad_accu
        graddelta_accu_temp = tf.cast(self.k0, dtype=variable.dtype) * tf.cast(self.learning_rate, dtype=variable.dtype) - graddelta_accu

        ler.assign(tf.divide(graddelta_accu_temp, grad_accu_temp))

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "momentum": self.momentum,
                "sigma": self.sigma,
                "amsgrad": self.amsgrad,
                "k0": self.k0
            }
        )
        return config

#%%

class Nlarcc(tf.keras.optimizers.Optimizer):
    r"""
    Optimizer that implements the Nlarc/m algorithm.
    If momentum=0 or momentum=None, the optimizer becomes Nlarc. For a non-zero value of momentum, it is Nlarcm.
    """
    __name__ = 'Nlarc'
    def __init__(
        self,
        learning_rate=0.1,
        momentum=1,
        k0= 0.1,
        sigma= 1e-19,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=1,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="Nlarcc",
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

        self._learning_rate = self._build_learning_rate(learning_rate) # learning_rate
        self.learning_rate_ = tf.Variable(learning_rate, dtype=tf.keras.backend.floatx()) # for internal usage

        self.momentum = momentum
        self.k0 = k0
        self.sigma = sigma
        self.amsgrad = amsgrad

        if self.momentum == 0:
            warnings.formatwarning = custom_formatwarning
            warnings.simplefilter("always")  # Ensure the warning is always shown
            warnings.warn(f"Warning.........."
                          f"The value of momentum is zero. This is interpreted as no momentum at all, "
                          f"which slows down the convergence. If this is intended, "
                          f"for a more efficient/stable convergence, please set momentum to be None.")
            warnings.simplefilter("default")  # Reset to the default behavior

        if self.global_clipnorm is None:
            warnings.formatwarning = custom_formatwarning
            warnings.simplefilter("always")  # Ensure the warning is always shown
            warnings.warn(f"Warning.........."
                          f"The global_clipnorm is set to None. In this case, Nlarc might not converge.")
            warnings.simplefilter("default")  # Reset to the default behavior

        self._current_precision = tf.keras.backend.floatx()

        if self._current_precision < 'float64':
            warnings.formatwarning = custom_formatwarning
            warnings.simplefilter("always")  # Ensure the warning is always shown
            warnings.warn(f"Warning.........."
                          f"Nlarc is optimized using float precision accuracy of 64. For a better performance, please "
                          f"change the float precision.")
            warnings.simplefilter("default")  # Reset to the default behavior

    def __max_precision(self):
        if self._current_precision == 'float16':
            return tf.float16, tf.float16.max
        elif self._current_precision == 'float32':
            return tf.float32, tf.float32.max
        elif self._current_precision == 'float64':
            return tf.float64, tf.float64.max
        else:
            raise ValueError(f"The floating precision {self._current_precision} is not recognized, "
                             "please use one of the following floating precessions: 16, 32, 64")

    def build(self, var_list):

        """Initialize optimizer variables.

        Nlarc/m optimizer has 6 types of variables: velocities, grad_accus, graddelta_accus
        lers, sigma_vars, and _velocity_hats (only if amsgrad is set True)

        Args:
          var_list: list of model variables to build Nlarc/m variables on.
        """

        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._velocities = []
        self._grad_accus = []
        self._graddelta_accus = []
        self._lers = []
        self._sigma_vars = []

        for var in var_list:

            initial_value_grad_accus = tf.zeros_like(var, dtype=var.dtype)
            initial_value_graddelta_accus = tf.zeros_like(var, dtype=var.dtype)
            initial_value_velocities = tf.zeros_like(var, dtype=var.dtype)
            initial_value_sigma_vars = tf.random.uniform(tf.shape(var), -tf.cast(tf.sqrt(3.), var.dtype),
                                                         tf.cast(tf.sqrt(3.), var.dtype),
                                                         dtype=var.dtype)
            # Other forms of initialization could be used such as truncated normal:
            # initial_value_sigma_vars = tf.random.truncated_normal(tf.shape(var), tf.cast(0, var.dtype),
            #                                              tf.cast(1, var.dtype),
            #                                              dtype=var.dtype)

            initial_value_lers = tf.ones_like(var, dtype=var.dtype) * self.learning_rate

            self._lers.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="ler",
                    initial_value=initial_value_lers
                )
            )

            self._grad_accus.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="grad_accu",
                    initial_value=initial_value_grad_accus
                )
            )

            self._graddelta_accus.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="graddelta_accu",
                    initial_value=initial_value_graddelta_accus
                )
            )

            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="vel",
                    initial_value=initial_value_velocities
                )
            )

            self._sigma_vars.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="sigma_var",
                    initial_value=initial_value_sigma_vars
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

        """Update step given gradient and the associated model variables."""

        sigma_tensor = tf.where(tf.greater(tf.abs(gradient),0),
                                tf.minimum(tf.cast(self.sigma, dtype=variable.dtype),
                                           tf.abs(gradient)), tf.cast(self.sigma, dtype=variable.dtype))

        local_step = tf.cast(self.iterations + 1, variable.dtype)
        var_key = self._var_key(variable)
        vel = self._velocities[self._index_dict[var_key]]
        ler = self._lers[self._index_dict[var_key]]

        grad_accu = self._grad_accus[self._index_dict[var_key]]
        graddelta_accu = self._graddelta_accus[self._index_dict[var_key]]
        sigma_var = self._sigma_vars[self._index_dict[var_key]]

        # This is to initiate learning rate schedule
        ler.assign(tf.cond(tf.greater(tf.cast(self.learning_rate_, variable.dtype),
                                      tf.cast(self.learning_rate, variable.dtype)),
                           true_fn=lambda: tf.multiply(tf.cast(self.learning_rate, variable.dtype), ler),
                           false_fn=lambda: ler))

        if isinstance(gradient, tf.IndexedSlices):
            warnings.formatwarning = custom_formatwarning
            warnings.simplefilter("always")  # Ensure the warning is always shown
            warnings.warn(f"Warning.........."
                            "Sparse gradients are not yet efficiently implemented in Nlarc. "
                            "This may cause the running process to be slow."
            )
            warnings.simplefilter("default")  # Reset to the default behavior

        pre_var = tf.identity(variable)

        if self.momentum is not None:

            # For float32; if sigma is too small, there could be an overflow. Hence, sigma needs to be capped
            # appropriately.
            positive_sigma2_inf_mask = tf.math.is_inf(tf.pow(tf.cast(self.sigma, dtype=variable.dtype), -2))
            def false_fn():
                return tf.divide(tf.multiply(tf.pow(tf.cast(self.sigma, dtype=variable.dtype), -2),
                                 tf.pow(sigma_tensor, 2)), local_step)
            def true_fn():
                _, max_precision = self.__max_precision()
                # to avoid overflow we have capped the max precision by a factor of 0.01
                return tf.divide(tf.multiply(
                    tf.clip_by_value(tf.pow(tf.cast(self.sigma, dtype=variable.dtype), -2), 0,
                                     max_precision * 0.01), tf.pow(sigma_tensor, 2)), local_step)

            # Note that if some of the gradients are very small, sigma_tensor^2 could be nearly
            # zero; which makes m_t=0. This especially affects (tf.abs(vel) + m_t) in rho_t. In particular,
            # if the precision is set to float32 and sigma very small; this could cause for elements of sigma_tensor
            # which depend on the square of sigma to become zero. This affects rho_t.
            m_t = tf.cond(positive_sigma2_inf_mask, true_fn, false_fn)
            any_zero_velmt = tf.reduce_any(tf.equal(tf.add(tf.abs(vel), m_t), 0))

            rho_t = tf.cond(any_zero_velmt,
                            lambda: tf.divide(tf.multiply(tf.divide(tf.cast(self.momentum, dtype=variable.dtype),
                            tf.add(tf.constant(1, dtype=variable.dtype), tf.abs(ler))), m_t),
                                              tf.add(tf.abs(vel),
                                                     tf.add(m_t, tf.constant(1e-10, dtype=variable.dtype)))),
                            lambda: tf.divide(tf.multiply(tf.divide(tf.cast(self.momentum, dtype=variable.dtype),
                                               tf.add(tf.constant(1, dtype=variable.dtype),
                                                      tf.abs(ler))), m_t),
                                              tf.add(tf.abs(vel), m_t))
                            )



            variable.assign_add(tf.multiply(rho_t, vel))
            variable.assign_sub(tf.multiply(ler, gradient))
            variable.assign_add(tf.multiply(sigma_tensor, sigma_var))

            vel.assign(tf.multiply(rho_t, vel))
            vel.assign_sub(tf.multiply(ler, gradient))
            # or similarly
            # vel.assign(rho_t * vel - ler * gradient)

        else:
            variable.assign_sub(tf.multiply(gradient, ler))
            variable.assign_add(tf.multiply(sigma_tensor, sigma_var))

        delta_params = tf.subtract(variable, pre_var)
        positive_inf_mask = tf.math.is_inf(tf.pow(sigma_tensor, -2))
        dtype, max_precision = self.__max_precision()
        sigma_tensor_square = tf.where(positive_inf_mask, tf.constant(max_precision * 0.01, dtype=dtype),
                                       tf.pow(sigma_tensor, -2))

        sigma_tensor1 = tf.multiply(sigma_tensor_square, tf.pow(gradient, 2))
        sigma_tensor2 = tf.multiply(sigma_tensor_square, tf.multiply(gradient, delta_params))

        grad_accu.assign_add(sigma_tensor1)
        graddelta_accu.assign_add(sigma_tensor2)
        grad_accu_temp = tf.add(tf.cast(self.k0, dtype=variable.dtype), grad_accu)
        graddelta_accu_temp = tf.subtract(tf.multiply(self.learning_rate, tf.cast(self.k0, dtype=variable.dtype)),
                                          graddelta_accu)

        ler.assign(tf.divide(graddelta_accu_temp, grad_accu_temp) / local_step)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "momentum": self.momentum,
                "sigma": self.sigma,
                "amsgrad": self.amsgrad,
                "k0": self.k0
            }
        )
        return config
