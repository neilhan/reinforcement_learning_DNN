#! python
# -*- coding: utf-8 -*-

import sys
import os
from datetime import datetime

import gym
from gym import wrappers
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt


class SGDRegressorMy:
    def __init__(self, D):
        '''
        :param D: a int value, the dimensions of the input.
        '''
        print('TensolFlow implementation of SGDRegressor')
        self.lr = 10e-2

        # inputs, targets, params
        # matmul doesn't work when w is 1-D
        # so make it 2-D, flatten the prediction
        self.w = tf.Variable(tf.random_normal(shape=(D, 1)), name='w')
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

        # make prediction and costs
        Y_hat = tf.reshape(tf.matmul(self.X, self.w), [-1])
        delta = self.Y - Y_hat
        cost = tf.reduce_sum(delta * delta)

        # ops we want to call later
        self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(cost)
        self.predict_op = Y_hat

        # init session, init params
        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)

    def partial_fit(self, X, Y):
        self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})
        # self.w += self.lr * (Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return self.session.run(self.predict_op, feed_dict={self.X: X})
        # return X.dot(self.w)


class FeatureTransformer:
    def __init__(self, env):
        # observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        # sampling state has issues, velocities --> infinity,
        # so we created random the space,
        #  [(-1, 1), (-1, 1),..]
        observation_examples = np.random.random((20000, 4)) * 2 - 1
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=0.05, n_components=1000)),
            ("rbf2", RBFSampler(gamma=0.1, n_components=1000)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=1000)),
            ("rbf4", RBFSampler(gamma=1.0, n_components=1000))
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


class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer
        self.models = []

        for i in range(env.action_space.n):
            model = SGDRegressorMy(feature_transformer.dimensions)
            self.models.append(model)

    def predict(self, s):
        x = self.feature_transformer.transform(np.atleast_2d(s))
        return np.array([m.predict(x)[0] for m in self.models])

    def update(self, s, a, G):
        # with a fixed learning rate
        x = self.feature_transformer.transform(np.atleast_2d(s))
        self.models[a].partial_fit(x, [G])

    def sample_action(self, s, eps):
        '''
        epsilon greedy, random action
        :param s: state, observation
        :param eps:
        :return:
        '''
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            p = self.predict(s)
            return np.argmax(p)


def play_one_episode(model, eps, gamma):
    '''
    play one episode
    :param model: Model
    :param eps: epsilon, how much to explore
    :param gamma: decay factor for this new state
    :return:
    '''
    env = model.env
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        totalreward += reward
        if done and iters < 199:
            reward = -300

        # update model. for q-learning, update return-G
        next = model.predict(observation)
        G = reward + gamma * np.max(next)  # New Return
        model.update(prev_observation, action, G)

        iters += 1
    return totalreward


def plot_total_rewards_n_running_avg(totalrewards):
    plt.plot(totalrewards)
    plt.title('Rewards')

    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t - 100):(t + 1)].mean()
    plt.plot(running_avg)
    plt.title('RunningAverage')
    plt.show()


def record_video(env):
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = '../../model/video/cart_pole/' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)
    return env


def main():
    env = gym.make('CartPole-v0')
    ftr = FeatureTransformer(env)
    model = Model(env, ftr)
    gamma = 0.99

    env_video = record_video(env)
    if 'monitor' in sys.argv:
        env = env_video

    N = 500
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        eps = 1.0 / np.sqrt(n + 1)
        totalreward = play_one_episode(model, eps, gamma)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print('episode:', n, 'total reward:', totalreward, 'eps:', eps)

    print('avg reward for last 100 episodes:', totalrewards[-100:].mean())
    print('total steps:', totalrewards.sum())

    model.env = env_video
    play_one_episode(model, 0, gamma)

    plot_total_rewards_n_running_avg(totalrewards)

if __name__ == '__main__':
    main()
