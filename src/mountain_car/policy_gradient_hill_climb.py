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


# model that does policy approximation pi(a|s)
class PolicyModel:
    def __init__(self, feature_transformer, D, hidden_layer_sizes_mean=[], hidden_layer_sizes_var=[]):
        '''
        feature_transformer: FeatureTransformer
        D: input size, the feature size
        hidden_layer_sizes_mean: hidden layer sizes for gaussian mean.
        hidden_layer_sizes_var: size for gaussian variance.
        '''
        # save inputs
        self.feature_transformer = feature_transformer
        self.D = D
        self.hidden_layer_sizes_mean = hidden_layer_sizes_mean
        self.hidden_layer_sizes_var = hidden_layer_sizes_var

        # model the mean ----------
        self.mean_layers = []
        M1 = D
        for M2 in hidden_layer_sizes_mean:
            layer = HiddenLayer(M1, M2)
            self.mean_layers.append(layer)
            M1 = M2

        # final layer
        layer = HiddenLayer(M1, 1, lambda x: x, use_bias=False, zeros=True)
        self.mean_layers.append(layer)

        # model the variance ----------
        self.var_layers = []
        M1 = D
        for M2 in hidden_layer_sizes_var:
            layer = HiddenLayer(M1, M2)
            self.var_layers.append(layer)
            M1 = M2
        # add final layer
        layer = HiddenLayer(M1, 1, tf.nn.softplus, use_bias=False, zeros=False)
        self.var_layers.append(layer)

        # parameters ------------------
        self.params = []
        for layer in (self.mean_layers + self.var_layers):
            self.params += layer.params
            # self.params contains all parameters for all mean_layes and var_layers.
            # params are tf.Variable

        # input, target
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.actions = tf.placeholder(tf.float32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

        # local function
        def get_output(layers):
            Z = self.X
            for layer in layers:
                Z = layer.forward(Z)
            return tf.reshape(Z, [-1])

        # output and cost function
        mean = get_output(self.mean_layers)
        std = get_output(self.var_layers) + 1e-4 # smoothing. + 0.0001
        # variance is standard deviation, std
        norm = tf.contrib.distributions.Normal(mean, std)
        self.predict_op = tf.clip_by_value(norm.sample(), -1, 1)

        # todo What's train_op

    def set_session(self, session):
        self.session = session

    def init_vars(self):
        init_op = tf.variables_initializer(self.params)
        self.session.run(init_op)

    def partial_fit(self, X, actions, advantages):
        X = np.atleast_2d(X)
        X = self.feature_transformer.transforn(X)
        actions = np.atleast_1d(actions)
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

    def copy(self):
        clone = PolicyModel(self.feature_transformer, self.D, self.hidden_layer_sizes_mean, self.hidden_layer_sizes_var)
        clone.set_session(self.session)
        clone.init_vars()
        clone.copy_from(self)
        return clone

    def copy_from(self, other):
        ops = []
        my_params = other.params
        for p, q in zip(mp_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        self.session.run(ops)

    def perturb_params(self):
        ops = []
        for p in self.params:
            v = self.session.run(p)
            noise = np.random.randn(*v.shape) / np.sqrt(v.shape[0]) * 5.0
            if np.random.random() < 0.1:
                # probability 0.1 start from scratch
                op = p.assign(noise)
            else:
                op = p.assign(v + noise)
            ops.append(op)
        self.session.run(ops)
