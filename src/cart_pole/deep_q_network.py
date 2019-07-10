#! python
# -*- coding: utf-8 -*-

import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import gym
from gym import wrappers


def plot_total_rewards_n_running_avg(total_rewards):
    plt.plot(total_rewards)
    plt.title('Rewards')

    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = total_rewards[max(0, t - 100):(t + 1)].mean()
    plt.plot(running_avg)
    plt.title('RunningAverage')
    plt.show()

def record_video_env(env):
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = '../../model/video/cart_pole/' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)
    return env

# ---------------------------------------------------------
class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
        self.W = tf.Variable(tf.random.normal(shape=(M1, M2)))
        self.params = [self.W]
        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
            self.params.append(self.b)
        self.f = f

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.f(a)

# ---------------------------------------------------------
class DeepQNetwork:
    def __init__(self, D, K, hidden_layer_sizes, gamma, max_experiences=10000, min_experiences=100, batch_size=32):
        '''
        D - input size.
        K - output size. Actions.
        '''
        self.K = K

        # create the layers
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2
        # output layer
        layer = HiddenLayer(M1, K, lambda x: x)
        self.layers.append(layer)

        # params can be copied
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        # input, and target
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

        # calc
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = Z  # estimation, approximation
        self.predict_op = Y_hat

        # preparing for cost, then train
        selected_action_values = tf.reduce_sum(self.predict_op * tf.one_hot(self.actions, K),
                                               reduction_indices=[1])

        cost = tf.reduce_sum(tf.square(self.G - selected_action_values))
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)
        # self.train_op = tf.train.AdagradOptimizer(1e-2).minimize(cost)
        # self.train_op = tf.train.MomentumOptimizer(4e-2, momentum=0.9).minimize(cost)
        # self.train_op = tf.train.GradientDescentOptimizer(3e-6).minimize(cost)

        # replay buffer,
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': [] }
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_size = batch_size
        self.gamma = gamma

    def copy_from(self, other):
        ops = []
        my_params = self.params
        other_params = other.params
        for p, q in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        self.session.run(ops)

    def set_session(self, session):
        self.session = session

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def train(self, target_network):
        # random batch from buffer, do Gradient Descent
        if len(self.experience['s']) < self.min_experiences:
            return

        idx = np.random.choice(len(self.experience['s']), size=self.batch_size, replace=False)
        states = [self.experience['s'][i] for i in idx]
        actions = [self.experience['a'][i] for i in idx]
        rewards = [self.experience['r'][i] for i in idx]
        next_states = [self.experience['s2'][i] for i in idx]
        dones = [self.experience['done'][i] for i in idx]
        next_Qs = np.max(target_network.predict(next_states), axis=1)
        targets = [(r + self.gamma*next_q) if not done else r
                   for r, next_q, done in zip(rewards, next_Qs, dones)]

        # optimize
        self.session.run(self.train_op,
                         feed_dict={self.X: states,
                                    self.G: targets,
                                    self.actions: actions})

    def add_experience(self, s, a, r, s2, done):
        if len(self.experience['s']) >= self.max_experiences:
            self.experience['s'].pop(0)
            self.experience['a'].pop(0)
            self.experience['r'].pop(0)
            self.experience['s2'].pop(0)
            self.experience['done'].pop(0)
        self.experience['s'].append(s)
        self.experience['a'].append(a)
        self.experience['r'].append(r)
        self.experience['s2'].append(s2)
        self.experience['done'].append(done)

    def sample_action(self, x, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            X = np.atleast_2d(x)
            return np.argmax(self.predict(X)[0])

#end DeepQNetwork --------

def play_one_episode(env, model, tmodel, eps, gamma, copy_period):
    observation = env.reset()
    done = False
    total_reward = 0
    iters = 0
    while not done and iters < 2000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        total_reward += reward
        if done:
            reward = -200

        # update model, train
        model.add_experience(prev_observation, action, reward, observation, done)
        model.train(tmodel)

        iters += 1

        if iters % copy_period == 0:
            tmodel.copy_from(model)

    return total_reward


def play(env, model):
    observation = env.reset()
    done = False
    iters = 0
    while not done and iters < 2000:
        action = model.sample_action(observation, 0.0)
        prev_observation = observation
        observation, reward, done, info = env.step(action)
        iters += 1


def main():
    env = gym.make('CartPole-v0')
    env_video = record_video_env(env)
    if 'monitor' in sys.argv:
        env = env_video

    gamma = 0.99
    copy_period = 50

    D = len(env.observation_space.sample())
    K = env.action_space.n
    hidden_layers_sizes = [200, 200]
    model = DeepQNetwork(D, K, hidden_layers_sizes, gamma)
    tmodel = DeepQNetwork(D, K, hidden_layers_sizes, gamma)

    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    model.set_session(session)
    tmodel.set_session(session)

    # train/play
    N = 500
    total_rewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        eps = 1.0 / np.sqrt(n+1)
        total_reward = play_one_episode(env, model, tmodel, eps, gamma, copy_period)
        # total_reward =       play_one_c(env, model, tmodel, eps, gamma, copy_period)
        total_rewards[n] = total_reward
        if n % 100 == 0:
            print('episode:', n, 'total reward:', total_reward, 'eps:', eps,
                  'avg reward(last 100):', total_rewards[max(0, n-100):(n+1)].mean())
    print('avg reward for last 100 episodes:', total_rewards[-100:].mean())
    print('total steps:', total_rewards.sum())

    # # have a look for the training result
    model.env = env_video
    play(env_video, model)

    plot_total_rewards_n_running_avg(total_rewards)


if __name__ == '__main__':
    main()
