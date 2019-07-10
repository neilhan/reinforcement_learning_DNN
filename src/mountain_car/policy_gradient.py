#! python
# -*- coding: utf-8 -*-

import gym
from gym import wrappers
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor


def get_record_env(env):
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = '../../model/video/cart_pole/' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)
    return env


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


def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = total_rewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title('Running Avergae')
    plt.show()


def plot_cost_to_go(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0],
                    env.observation_space.high[0],
                    num=num_tiles)
    y = np.linspace(env.observation_space.low[0],
                    env.observation_space.high[0],
                    num=num_tiles)
    X, Y = np.meshgrid(x, y)
    # X, Y, Z all have shape: [num_tiles, num_tiles]
    Z = np.apply_along_axis(lambda _ : -np.max(estimator.predict(_)), 2, np.dstack([X, y]))
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z,
                           rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm,
                           vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-to-go == -V(s)')
    ax.set_title('Cost-to-go function')
    fig.colorbar(surf)
    plt.show()


# FeatureTransformer
class FeatureTransformer:
    def __init__(self, env, n_components=500):
        # observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        # sampling state has issues, velocities --> infinity,
        # so we created random the space,
        #  [(-1, 1), (-1, 1),..]
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=0.5, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=5.0, n_components=n_components))
        ])
        feature_examples = featurizer.fit_transform(scaler.transform(observation_examples))

        self.dimensions = feature_examples.shape[1] # [20000, 4000] -> 4000
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observation):
        '''
        from observation to int that represents a bin.
        :param observation: [a,b,c,d]
        :return: scaled -> transformed features [a',b',c',d']
        '''
        scaled = self.scaler.transform(observation)
        return self.featurizer.transform(scaled)

class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True, zeros=False):
        '''
        m1, m2: input / output size
        f: activation method
        zeros: init to zero
        '''
        if zeros:
            W = np.zeros((M1, M2)).astype(np.float32)
            self.W = tf.Variable(W)
        else:
            self.W = tf.random_normal(shape=( M1, M2 )) * np.sqrt(2.0 / M1, dtype=np.float32)

        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))

        self.f = f

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.f(a)


# model that does policy approximation pi(a|s)
class PolicyModel:
    def __init__(self, feature_transformer, D, hidden_layer_sizes=[]):
        '''
        feature_transformer: FeatureTransformer
        D: input size, the feature size
        hidden_layer_sizes_mean: hidden layer sizes for gaussian mean.
        hidden_layer_sizes_var: size for gaussian variance.
        '''
        self.feature_transformer = feature_transformer

        # inputs, targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.actions = tf.placeholder(tf.float32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

        # Hidden layers
        M1 = D
        self.hidden_layers = []
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.hidden_layers.append(layer)
            M1 = M2

        # final layer, mean
        self.mean_layer = HiddenLayer(M1, 1, lambda x: x, use_bias=False, zeros=True)
        # final layer, variance
        self.stdv_layer = HiddenLayer(M1, 1, tf.nn.softplus, use_bias=False, zeros=False)

        # connect all layers ---------
        Z = self.X
        for layer in self.hidden_layers:
            Z = layer.forward(Z)

        # connect to mean, stdv outputs
        mean = self.mean_layer.forward(Z)
        stdv = self.stdv_layer.forward(Z) + 1e-5  # smoothing
        # to 1_D
        mean = tf.reshape(mean, [-1])
        stdv = tf.reshape(stdv, [-1])

        norm = tf.contrib.distributions.Normal(mean, stdv)
        self.predict_op = tf.clip_by_value(norm.sample(), -1, 1)

        log_probs = norm.log_prob(self.actions)
        cost = -tf.reduce_sum(self.advantages * log_probs + 0.1 * norm.entropy())
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(cost)

    def set_session(self, session):
        self.session = session

    def init_vars(self):
        init_op = tf.variables_initializer(self.params)
        self.session.run(init_op)

    def partial_fit(self, X, actions, advantages):
        X = np.atleast_2d(X)
        X = self.feature_transformer.transform(X)

        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)
        self.session.run(
            self.train_op,
            feed_dict={
                self.X: X,
                self.actions: actions,
                self.advantages: advantages,
            }
        )

    def predict(self, X):
        X = np.atleast_2d(X)
        X = self.feature_transformer.transform(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def sample_action(self, X):
        p = self.predict(X)[0]
        return p


# V(s) approximation
class ValueModel:
    def __init__(self, feature_transformer, D, hidden_layer_sizes=[]):
        '''
        D: size of the input features
        '''
        self.feature_transformer = feature_transformer
        self.costs = []  # save the training cost for plot
        # layers --------------
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2
        # final layer
        layer = HiddenLayer(M1, 1, lambda x:x)
        self.layers.append(layer)

        # input, output
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = tf.reshape(Z, [-1])  # something_hat: approximation of something
        self.predict_op = Y_hat

        cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
        self.cost = cost
        self.train_op = tf.train.AdamOptimizer(1e-1).minimize(cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, Y):
        X = np.atleast_2d(X)
        X = self.feature_transformer.transform(X)
        Y = np.atleast_1d(Y)
        self.session.run(self.train_op, feed_dict={self.X:X, self.Y:Y})
        cost = self.session.run(self.cost, feed_dict={self.X:X, self.Y:Y})
        self.costs.append(cost)

    def predict(self, X):
        X = np.atleast_2d(X)
        X = self.feature_transformer.transform(X)
        return self.session.run(self.predict_op, feed_dict={self.X:X})


def play_one_td(env, pmodel, vmodel, gamma):
    observation = env.reset()
    done = False
    total_reward = 0
    iters = 0

    while not done and iters < 2000:
        # 2000, all good, quit. 200 is too early. Don't want to run forever
        action = pmodel.sample_action(observation)
        prev_observation = observation
        observation, reward, done, info = env.step([action])
        # mountain car requires object[0] as action

        # models updates
        V_next = vmodel.predict(observation)
        G = reward + gamma * V_next
        advantage = G - vmodel.predict(prev_observation)
        pmodel.partial_fit(prev_observation, action, advantage)
        vmodel.partial_fit(prev_observation, G)

        total_reward += reward
        iters += 1

    return total_reward, iters


def main():
    gamma = 0.99

    env = gym.make('MountainCarContinuous-v0')
    env_video = get_record_env(env)

    feature_transformer = FeatureTransformer(env, n_components=100)
    D = feature_transformer.dimensions
    pmodel = PolicyModel(feature_transformer, D, [])
    vmodel = ValueModel(feature_transformer, D, [])

    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    pmodel.set_session(session)
    vmodel.set_session(session)

    N = 50 # play 50 times
    total_rewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        total_reward, num_steps = play_one_td(env, pmodel, vmodel, gamma)
        total_rewards[n] = total_reward
        if n % 1 == 0:
            print('episode:', n, 'total reward: %.1f' % total_reward, 'num steps: %d' % num_steps)

    print('avg reward for last 100 episodes:', total_rewards[-100:].mean())

    # plot
    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(total_rewards)
    plot_cost_to_go(env, vmodel)

    # show one play
    play_one(env_video, pmodel, gamma)


if __name__ == '__main__':
    main()
