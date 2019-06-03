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

# FeatureTransformer
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
