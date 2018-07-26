#! python
# -*- coding: utf-8 -*-

from __future__ import print_function, nested_scopes, generators, division
from builtins import range

import os
import sys
from datetime import datetime
import numpy as np

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
# from sklearn.linear_model import SGDRegressor

import gym
from gym import wrappers

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# SGDRegressor defaults:
# loss='squared_loss', penalty='l2', alpha=0.0001,
# l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True,
# verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling',
# eta0=0.01, power_t=0.25, warm_start=False, average=False

class SGDRegressor:
    def __init__(self, **kwargs):
        self.w = None
        self.lr = 0.01

    def partial_fit(self, x, y):
        if self.w is None:
            D = x.shape[1]
            self.w = np.random.randn(D) / np.sqrt(D)

        self.w += self.lr*(y - x.dot(self.w)).dot(x)

    def predict(self, x):
        return x.dot(self.w)


def plot_cost_to_go(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0],
                    env.observation_space.high[0],
                    num=num_tiles)
    y = np.linspace(env.observation_space.low[1],
                    env.observation_space.high[1],
                    num=num_tiles)
    X, Y = np.meshgrid(x, y)

    # X, Y, Z, all have shape (num_tiles, num_tiles)
    Z = np.apply_along_axis(lambda _ : -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface( X, Y, Z,
                            rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm,
                            vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title('Cost-To-Go Function')
    fig.colorbar(surf)
    plt.show()

def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = total_rewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title('Running Avergae')
    plt.show()

class FeatureTransformer:
    def __init__(self, env, n_components=500):
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        # to converte a state to a featurizes represetation.
        # RBF kernels with different variances to cover different parts 
        # of the space
        featurizer = FeatureUnion([('rbf1', RBFSampler(gamma=5.0, n_components=n_components)),
                                   ('rbf2', RBFSampler(gamma=2.0, n_components=n_components)),
                                   ('rbf3', RBFSampler(gamma=1.0, n_components=n_components)),
                                   ('rbf4', RBFSampler(gamma=0.5, n_components=n_components)),
                                  ])
        example_features = featurizer.fit_transform(scaler.transform(observation_examples))
        self.dimensions = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        # print('observations:', observations)
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)

# Holds one SGDRegressor for each action
class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            model.partial_fit(feature_transformer.transform([env.reset()]), [0])
            self.models.append(model)

    def predict(self, s):
        x = self.feature_transformer.transform([s])
        result = np.stack([m.predict(x) for m in self.models]).T
        assert(len(result.shape) == 2)
        return result

    def update(self, s, a, G):
        x = self.feature_transformer.transform([s])
        assert(len(x.shape) == 2)
        self.models[a].partial_fit(x, [G])

    def sample_action(self, s, eps):
        # for eps 0 
        # don't need epsilon-greedy, because SGDRegressor predicts 0 for all states
        # until they are updated. This works as the Optimistic Initial values method, 
        # since all the rewards, for Mountain Car are -1.
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

# calculate everything up to max[Q(s, a)]
# R(t) + gamma*R(t+1) + .. + (gamma^(n-1))*R(t+n-1) + (gamma^n)*max[Q(s(t+n), a(t+n))]

# returns a list of states_and_rewards, and the total reward
def play_one(model, eps, gamma, n=5):
    observation = env.reset()
    done = False
    total_reward = 0
    rewards = []
    states = []
    actions = []
    iters = 0
    # array of [gamma^0, gamma^1, .. gama^(n-1)]
    multiplier = np.array([gamma]*n)**np.arange(n)
    # while not done and iters < 200
    while not done and iters < 10000:
        states.append(observation)
        prev_observation = observation

        # next action?
        action = model.sample_action(observation, eps)
        actions.append(action)
        # act on it
        observation, reward, done, info = env.step(action)
        rewards.append(reward)

        # recording, update model etc
        if len(rewards) >= n:
            # return up to prediction
            return_up_to_prediction = multiplier.dot(rewards[-n:])
            G = return_up_to_prediction + (gamma**n) * np.max(model.predict(observation)[0])
            model.update(states[-n], actions[-n], G)
            total_reward += reward
            iters += 1
    # finish one play, do calculation
    # empty cache
    if n == 1:
        rewards = []
        states = []
        actions = []
    else:
        rewards = rewards[-n+1:]
        states = states[-n+1:]
        actions = actions[-n+1:]
    # gym cuts off at 200 steps even if you haven't reach goal
    # we are done if position >= 0.5
    if observation[0] >= 0.5:
        # made it
        while len(rewards) > 0:
            G = multiplier[:len(rewards)].dot(rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)
    else:
        # didn't make it
        while len(rewards) > 0:
            guess_rewards = rewards + [-1]*(n - len(rewards))
            G = multiplier.dot(guess_rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)
    return total_reward

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, 'constant')
    gamma = 0.99

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 5000
    total_rewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        eps = 0.05*(0.999**n)  # reduce epsilon greedy as system learn more
        total_reward = play_one(model, eps, gamma)
        total_rewards[n] = total_reward
        print('episode:', n, 'total reward:', total_reward)
    print('avg reward for last 100 episode:', total_rewards[-100:].mean())
    print('total steps:', -total_rewards.sum())

    plt.plot(total_rewards)
    plt.title('Rewards')
    plt.show()

    plot_running_avg(total_rewards)
    plot_cost_to_go(env, model)
