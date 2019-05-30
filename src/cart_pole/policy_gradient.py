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

def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = total_rewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title('Running Avergae')
    plt.show()


class HiddenLayer:
    '''
    This is one hidden layer.
    '''

    def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
        '''
        M1 = input dimensions
        M2 = current layer # of neurons
        f is the activation function for the neurons.
        '''
        self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
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


# approximates pi(a | s)
class PolicyModel:
    def __init__(self, D, K, hidden_layer_sizes):
        '''
        D = feature size, number of features
        K = number of actions
        '''
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        # final layer, output
        # layer = HiddenLayer(M1, K, lambda x: x, use_bias=False)
        layer = HiddenLayer(M1, K, tf.nn.softmax, use_bias=False)
        self.layers.append(layer)

        # prepare: inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.actions = tf.placeholder(tf.int32, shape=(None, ), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None, ), name='advantages')

        # calculate output and cost
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        p_a_given_s = Z
        self.predict_op = p_a_given_s

        seleted_probs = tf.log(
            tf.reduce_sum(
                p_a_given_s * tf.one_hot(self.actions, K),
                reduction_indices=[1]
            )
        )

        cost = -tf.reduce_sum(self.advantages * seleted_probs)

        self.train_op = tf.train.AdagradOptimizer(10e-2).minimize(cost)
        # self.train_op = tf.train.AdamOptimizer(1e-1).minimize(cost)
        # self.train_op = tf.train.MomentumOptimizer(1e-4, momentum=0.9).minimize(cost)
        # self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

        def set_session(self, session):
            self.session = session

        def partial_fit(self, X, actions, advantages):
            X = np.atleast_2d(X)
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
            return self.session.run(self.predict_op, feed_dict={self.X: X})

        def sample_action(self, X):
            p = self.predict(X)[0]
            return np.random.choice(len(p), p=p)

class ValueModule:
    def __init__(self, D, hidden_layer_sizes):
        '''
        D = input feature size
        '''
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        # output layer
        layer = HiddenLayer(M1, 1, lambda x: x)
        self.layers.append(layer)

        # inputs, targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

        # calc output, cost
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = tf.reshape(Z, [-1])  # output
        self.predict_op = Y_hat

        cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
        self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)
        # self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)
        # self.train_op = tf.train.MomentumOptimizer(1e-2, momentum=0.9).minimize(cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, Y):
        X = np.atleast_2d(X)
        Y = np.atleast_1d(Y)
        self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.sesion.run(self.predict_op, feed_dict={self.X: X})
