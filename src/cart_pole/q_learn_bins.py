#! python
# -*- coding: utf-8 -*-

import sys
import os
import gym
from gym import wrappers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


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


class FeatureTransformer:
    def __init__(self):
        self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
        self.cart_velocity_bins = np.linspace(-2, 2, 9)
        self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
        self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9)

    def transform(self, observation):
        '''
        from observation to int that represents a bin.
        :param observation:
        :return:
        '''
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        return build_state([to_bin(cart_pos, self.cart_position_bins),
                            to_bin(cart_vel, self.cart_velocity_bins),
                            to_bin(pole_angle, self.pole_angle_bins),
                            to_bin(pole_vel, self.pole_velocity_bins), ])


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


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t - 100):(t + 1)].mean()
    plt.plot(running_avg)
    plt.title('RunningAverage')
    plt.show()


def record_video(env):
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = '../../model/video/' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)
    return env


if __name__ == '__main__':
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

    plt.plot(totalrewards)
    plt.title('Rewards')

    plot_running_avg(totalrewards)
