import random
from collections import deque

import tensorflow  as tf
from experiments.src_epxeriments.models import model_generate

tf.keras.backend.set_floatx('float64')

##

class Memory:
    def __init__(self, capacity, seed):
        self.buffer = deque(maxlen=capacity)
        self.seed = seed

    def add(self, experience):
        self.buffer.append(experience)
    def sample(self, batch_size):
        buffer_batch = random.sample(self.buffer, batch_size)
        return buffer_batch


class DQNAgent:
    def __init__(self, action_size=None, optimizer=None, optimizer_args=None,
                 seed=None, buffer_size=None, model_conf=None,
                 gamma=None, epsilon=None, epsilon_min=None, epsilon_decay=None, loss=None):

        self.seed = seed
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.model_conf = model_conf
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = Memory(self.buffer_size, self.seed)
        self.loss = loss

        model_conf['seed'] = self.seed
        self.model =model_generate(**self.model_conf)

        self.target_model = model_generate(**self.model_conf)
        self.optimizer = optimizer(**optimizer_args)

        # Compile the main and target models
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss)

        self.target_model.compile(optimizer=self.optimizer,
                           loss=self.loss)
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if tf.random.uniform([1], 0, 1, dtype=tf.float64) <= self.epsilon:
            return tf.random.uniform((), minval=0, maxval=self.action_size, dtype=tf.int64)
        q_values = self.model(state)
        return tf.argmax(q_values[0])

    @tf.function
    def replay(self, states, actions, rewards, next_states, dones):

        with tf.GradientTape() as tape:

            q_online = tf.reshape(self.model(next_states), [-1, self.action_size])
            action_q_online = tf.math.argmax(q_online, axis=1)  # optimal actions from the q_online
            q_target = tf.reshape(self.target_model(next_states), [-1, self.action_size])
            one_hot_actions = tf.cast(tf.one_hot(action_q_online, self.action_size, 1, 0), tf.float64)
            ddqn = tf.reduce_sum(q_target * one_hot_actions, axis=1)
            expected_q = rewards + self.gamma * ddqn * (1.0 - tf.cast(dones, tf.float64))
            model_out_reshaped = tf.reshape(self.model(states), [-1, self.action_size])

            # Ensure all tensors are cast to appropriate dtypes
            common_float_dtype = tf.float64
            common_int_dtype = tf.int32

            model_out_reshaped = tf.cast(model_out_reshaped, dtype=common_float_dtype)
            actions = tf.cast(actions, dtype=common_int_dtype)  # tf.one_hot expects int64 for indices
            action_size = tf.cast(self.action_size, dtype=common_int_dtype)

            main_q = tf.reduce_sum(model_out_reshaped * tf.one_hot(actions, action_size, 1.0, 0.0, dtype=common_float_dtype), axis=1)
            # loss = tf.reduce_mean(tf.square(tf.stop_gradient(expected_q) - main_q))
            loss = self.loss(tf.stop_gradient(expected_q), main_q)

        #Compute and apply gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

