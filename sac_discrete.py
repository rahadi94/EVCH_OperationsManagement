import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.contrib.distributions import Categorical
import numpy as np

import gym
import random


def copy_to_target_graph(sess):
    # Get the parameters of our DQNNetwork
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "ValueNet")

    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "ValueNetTarget")

    op_holder = []

    # Update our target_network parameters with DQNNetwork parameters
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    sess.run(op_holder)


def update_target_graph(sess, tau=0.005):
    source_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "ValueNet")
    target_params = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, "ValueNetTarget"
    )
    target_update_op = [
        tf.assign(target, (1 - tau) * target + tau * source)
        for target, source in zip(target_params, source_params)
    ]
    sess.run(target_update_op)


def _params(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)


def gather_tensor(params, idx):
    idx = tf.stack([tf.range(tf.shape(idx)[0]), idx[:, 0]], axis=-1)
    params = tf.gather_nd(params, idx)
    return params


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class ValueNet:
    def __init__(self, sess, state_size, hidden_dim, name):
        self.sess = sess

        with tf.variable_scope(name):
            self.states = tf.placeholder(
                dtype=tf.float32, shape=[None, state_size], name="value_states"
            )
            self.targets = tf.placeholder(
                dtype=tf.float32, shape=[None, 1], name="value_targets"
            )
            x = Dense(units=hidden_dim, activation="relu")(self.states)
            x = Dense(units=hidden_dim, activation="relu")(x)
            self.values = Dense(units=1, activation=None)(x)

            optimizer = tf.train.AdamOptimizer(0.001)

            loss = 0.5 * tf.reduce_mean(
                (self.values - tf.stop_gradient(self.targets)) ** 2
            )
            self.train_op = optimizer.minimize(loss, var_list=_params(name))

    def get_value(self, s):
        return self.sess.run(self.values, feed_dict={self.states: s})

    def update(self, s, targets):
        self.sess.run(self.train_op, feed_dict={self.states: s, self.targets: targets})


class SoftQNetwork:
    def __init__(self, sess, state_size, action_size, hidden_dim, name):
        self.sess = sess

        with tf.variable_scope(name):
            self.states = tf.placeholder(
                dtype=tf.float32, shape=[None, state_size], name="q_states"
            )
            self.targets = tf.placeholder(
                dtype=tf.float32, shape=[None, 1], name="q_targets"
            )
            self.actions = tf.placeholder(
                dtype=tf.int32, shape=[None, 1], name="q_actions"
            )

            x = Dense(units=hidden_dim, activation="relu")(self.states)
            x = Dense(units=hidden_dim, activation="relu")(x)
            x = Dense(units=action_size, activation=None)(x)
            self.q = tf.reshape(gather_tensor(x, self.actions), shape=(-1, 1))

            optimizer = tf.train.AdamOptimizer(0.001)

            loss = 0.5 * tf.reduce_mean((self.q - tf.stop_gradient(self.targets)) ** 2)
            self.train_op = optimizer.minimize(loss, var_list=_params(name))

    def update(self, s, a, target):
        self.sess.run(
            self.train_op,
            feed_dict={self.states: s, self.actions: a, self.targets: target},
        )

    def get_q(self, s, a):
        return self.sess.run(self.q, feed_dict={self.states: s, self.actions: a})


class PolicyNet:
    def __init__(self, sess, state_size, action_size, hidden_dim):
        self.sess = sess

        with tf.variable_scope("policy_net"):
            self.states = tf.placeholder(
                dtype=tf.float32, shape=[None, state_size], name="policy_states"
            )
            self.targets = tf.placeholder(
                dtype=tf.float32, shape=[None, 1], name="policy_targets"
            )
            self.actions = tf.placeholder(
                dtype=tf.int32, shape=[None, 1], name="policy_actions"
            )

            x = Dense(units=hidden_dim, activation="relu")(self.states)
            x = Dense(units=hidden_dim, activation="relu")(x)
            self.logits = Dense(units=action_size, activation=None)(x)
            dist = Categorical(logits=self.logits)

            optimizer = tf.train.AdamOptimizer(0.001)

            # Get action
            self.new_action = dist.sample()
            self.new_log_prob = dist.log_prob(self.new_action)

            # Calc loss
            log_prob = dist.log_prob(tf.squeeze(self.actions))
            loss = tf.reduce_mean(tf.squeeze(self.targets) - 0.2 * log_prob)
            self.train_op = optimizer.minimize(loss, var_list=_params("policy_net"))

    def get_action(self, s):
        action = self.sess.run(
            self.new_action, feed_dict={self.states: s[np.newaxis, :]}
        )
        return action[0]

    def get_next_action(self, s):
        next_action, next_log_prob = self.sess.run(
            [self.new_action, self.new_log_prob], feed_dict={self.states: s}
        )
        return next_action.reshape((-1, 1)), next_log_prob.reshape((-1, 1))

    def update(self, s, a, target):
        self.sess.run(
            self.train_op,
            feed_dict={self.states: s, self.actions: a, self.targets: target},
        )


env = gym.make("CartPole-v1")
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
hidden_dim = 256

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

value_net = ValueNet(sess, state_space, hidden_dim, "ValueNet")
value_net_target = ValueNet(sess, state_space, hidden_dim, "ValueNetTarget")

soft_q_net_1 = SoftQNetwork(sess, state_space, action_space, hidden_dim, "soft_q_1")
soft_q_net_2 = SoftQNetwork(sess, state_space, action_space, hidden_dim, "soft_q_2")

policy_net = PolicyNet(sess, state_space, action_space, hidden_dim)
sess.run(tf.initializers.global_variables())
copy_to_target_graph(sess)

replay_buffer = ReplayBuffer(1000000)


def soft_q_update(batch_size, frame_idx):
    gamma = 0.99
    alpha = 0.2

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    action = action.reshape((-1, 1))
    reward = reward.reshape((-1, 1))
    done = done.reshape((-1, 1))

    # Q target
    v_ = value_net_target.get_value(next_state)
    q_target = reward + (1 - done) * gamma * v_

    # Update V
    next_action, next_log_prob = policy_net.get_next_action(state)
    q1 = soft_q_net_1.get_q(state, next_action)
    q2 = soft_q_net_2.get_q(state, next_action)
    q = np.minimum(q1, q2)
    v_target = q - alpha * next_log_prob

    # Update Pi
    q1 = soft_q_net_1.get_q(state, action)
    q2 = soft_q_net_2.get_q(state, action)
    policy_target = np.minimum(q1, q2)

    soft_q_net_1.update(state, action, q_target)
    soft_q_net_2.update(state, action, q_target)
    value_net.update(state, v_target)
    policy_net.update(state, action, policy_target)


max_frames = 10000
max_steps = 500
frame_idx = 0
rewards = []
batch_size = 128
updates = 0

while frame_idx < max_frames:
    s = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        action = policy_net.get_action(s)
        s_, r, d, _ = env.step(action)
        replay_buffer.push(s, action, r, s_, d)

        s = s_
        episode_reward += r
        frame_idx += 1

        if len(replay_buffer) > batch_size:
            soft_q_update(batch_size, frame_idx)
            updates += 1

            if updates % 10 == 0:
                update_target_graph(sess)

        if d:
            print("Reward: ", episode_reward)
            break
