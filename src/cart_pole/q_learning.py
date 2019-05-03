#! python
# -*- coding: utf-8 -*-

import sys
import os
from datetime import datetime

import gym
from gym import wrappers
import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt


class SGDRegressorMy:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 10e-2

    def partial_fit(self, X, Y):
        self.w += self.lr * (Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)


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

        self.dimensions = feature_examples.shape[1] # [1, 4] -> 4
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


def build_state(features):
    '''
    take list of int, to a int, [1,2,3] -> 123
    :param features:
    :return: int
    '''
    return int(''.join(map(lambda feature: str(int(feature)), features)))


def to_bin(value, bins):
    '''
    return indexes of bins that each value belongs.
    value is in put array. bins is array of bin_size.
    :param value:
    :param bins:
    :return: array
    '''
    return np.digitize(x=[value], bins=bins)[0]


class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer

        num_states = 10 ** env.observation_space.shape[0]
        num_actions = env.action_space.n
        self.Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))

    def predict(self, s):
        x = self.feature_transformer.transform(s)
        return self.Q[x]  # of actions [q_0, q_1]

    def update(self, s, a, G):
        # with a fixed learning rate
        x = self.feature_transformer.transform(s)
        self.Q[x, a] += 10e-3 * (G - self.Q[x, a])

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

        # update model, q-learning, update return-G
        G = reward + gamma * np.max(model.predict(observation))  # New Return
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
    ft = FeatureTransformer()
    model = Model(env, ft)
    gamma = 0.9

    if 'monitor' in sys.argv:
        env = env_video

    N = 10000
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 1.0 / np.sqrt(n + 1)
        totalreward = play_one_episode(model, eps, gamma)
        totalrewards[n] = totalreward
        if n % 100 == 0:
            print('episode:', n, 'total reward:', totalreward, 'eps:', eps)
    print('avg reward for last 100 episodes:', totalrewards[-100:].mean())
    print('total steps:', totalrewards.sum())

    env_video = record_video(env)
    model.env = env_video
    play_one_episode(model, 0, gamma)

    plot_total_rewards_n_running_avg(totalrewards)

if __name__ == '__main__':
    main()
