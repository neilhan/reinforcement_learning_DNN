#! python
# -*- coding: utf-8 -*-

import os
import sys
import copy
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import gym
from gym import wrappers


## testing
# MAX_EXPERIENCES = 10000
# MIN_EXPERIENCES = 1000

MAX_EXPERIENCES = 500000
MIN_EXPERIENCES = 50000
TARGET_UPDATE_PERIOD = 10000
IM_SIZE = 84
K = 4

def get_model_npz_file_name():
    file_name = os.path.basename(__file__).split('.')[0]
    file_name = '../../model/atari/' + filename + '_weights.npz'
    return file_name

def get_record_video_env(env):
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = '../../model/video/atari/' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)
    return env

class ImageTransformer:
    '''
    Transform images to Input for neural network
    - to grayscale
    - resize
    - crop
    '''
    def __init__(self):
        with tf.variable_scope('image_transformer'):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(self.output,
                                                 [IM_SIZE, IM_SIZE],
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)  # remove size 1 dimension

    def transform(self, state, session= None):
        session = session or tf.get_default_session()
        return session.run(self.output,
                           feed_dict={self.input_state: state})

def update_state(state, obs_small):
    '''
    obs_small - the transformed image. What the env observation is.
    It's smaller size of the game-observation
    '''
    return np.append(state[:, :, 1:], np.expand_dims(obs_small, 2), axis=2)


class ReplayMemory:
    def __init__(self, size=MAX_EXPERIENCES, frame_height=IM_SIZE, frame_width=IM_SIZE,
                 agent_history_length=4, batch_size=32):
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0

        # allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

        self.states = np.empty((self.batch_size, self.agent_history_length, self.frame_height, self.frame_width),
                               dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length, self.frame_height, self.frame_width),
                               dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, frame, reward, terminal):
        '''
        action: int action
        frame: grayscale frame, 1 frame
        reward: the reward for this action
        terminal: bool, whether the episode terminated
        '''
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('frame shape wrong. expecting: (%s, %s)' % (self.frame_height, self.frame_width) )
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current]= terminal
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        if self.count is 0:
            raise ValueError('Replay memory is empty')
        if index < self.agent_history_length - 1:
            raise ValueError('index must be min 3')
        return self.frames[index-self.agent_history_length+1 : index+1, ...]

    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count-1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index-self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index-self.agent_history_length : index].any():
                    continue
                break
            #while
            self.indices[i] = index
        #for
        return self.indices

    def get_minibatch(self):
        '''
        return a batch of self.batch_size
        '''
        if self.count < self.agent_history_length:
            raise ValueError('not enough memory yet. needs more steps')
        indices = self._get_valid_indices()
        for i, idx in enumerate(indices):
            self.states[i] = self._get_state(idx -1)
            self.new_states[i] = self._get_state(idx)
        #for/
        return (np.transpose(self.states, axes=(0, 2, 3, 1)),
                self.actions[self.indices],
                self.rewards[self.indices],
                np.transpose(self.new_states, axes=(0, 2, 3, 1)),
                self.terminal_flags[self.indices])


# ------------
class DeepQNetwork:
    def __init__(self, K, conv_layer_sizes, hidden_layer_sizes, scope):
        '''
        K - action / output size
        scope - scope for tf variables.
        '''
        self.K = K
        self.scope = scope

        with tf.variable_scope(scope):
            # input. # tf convolution needs (#samples, h, w, c)
            # pixel / 255.0
            self.X = tf.placeholder(tf.float32, shape=(None, IM_SIZE, IM_SIZE, 4), name='X')

            self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
            self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

            # output
            Z = self.X / 255.0
            for num_output_filters, filtersz, poolsz in conv_layer_sizes:
                Z = tf.contrib.layers.conv2d(Z,
                                             num_output_filters,
                                             filtersz,
                                             poolsz,
                                             activation_fn=tf.nn.relu)

            # fully connected layers
            Z = tf.contrib.layers.flatten(Z)
            for M in hidden_layer_sizes:
                Z = tf.contrib.layers.fully_connected(Z, M)
            # output layer
            self.predict_op = tf.contrib.layers.fully_connected(Z, K)

            # preparing for cost, then train_op
            selected_action_values = tf.reduce_sum(self.predict_op * tf.one_hot(self.actions, K),
                                                   reduction_indices=[1])
            # cost = tf.reduce_mean(tf.square(self.G - selected_action_value))
            cost = tf.reduce_mean(tf.losses.huber_loss(self.G, selected_action_values))
            self.train_op = tf.train.AdamOptimizer(1e-5).minimize(cost)
            # self.train_op = tf.train.AdagradOptimizer(1e-2).minimize(cost)
            # self.train_op = tf.train.RMSPropOptimizer(2.5e-4, decay=0.99, epsilon=1e-3).minimize(cost)
            # self.train_op = tf.train.RMSPropOptimizer(2.5e-4, 0.99, 0.0, 1e-6).minimize(cost)
            # self.train_op = tf.train.MomentumOptimizer(1e-3, momentum=0.9).minimize(cost)
            # self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)
            self.cost = cost
        #/with
    #/__init__

    def copy_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)

        ops = []
        for p, q in zip(mine, theirs):
            op = p.assign(q)
            ops.append(op)
        #/for
        self.session.run(ops)
    #/copy_from

    def set_session(self, session):
        self.session = session

    def predict(self, states):
        return self.session.run(self.predict_op, feed_dict={self.X: states})

    def update(self, states, actions, targets):
        '''
        training
        '''
        c, _ = self.session.run([self.cost, self.train_op],
                               feed_dict={self.X: states,
                                          self.G: targets,
                                          self.actions: actions})
        return c

    def sample_action(self, x, eps):
        '''
        during game play, given current state, predict action
        '''
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            return np.argmax(self.predict([x])[0])

    def save(self):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        params = self.session.run(params)

        file_name = get_model_npz_file_name()
        np.savez(file_name, *params)

    def load(self):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        file_name = get_model_npz_file_name()
        npz = np.load(file_name)
        ops = []
        for p, (_, v) in zip(params, npz.iteritems()):
            ops.append(p.assign(v))
        self.session.run(ops)
#/ class DeepQNetwork


def learn(model, target_model, experience_replay_buffer, gamma, batch_size):
    # create minibatch
    states, actions, rewards, next_states, dones = experience_replay_buffer.get_minibatch()

    # target?
    next_Qs = target_model.predict(next_states)
    next_Q = np.amax(next_Qs, axis=1)
    targets = rewards + np.invert(dones).astype(np.float32) * gamma * next_Q

    # update model
    loss = model.update(states, actions, targets)
    return loss

def play_one(env, session, total_t, experience_replay_buffer,
             model, target_model, image_transformer,
             gamma, batch_size, epsilon, epsilon_change, epsilon_min):
    time_0 = datetime.now()
    # env init
    obs = env.reset()
    obs_small = image_transformer.transform(obs, session)

    state = np.stack([obs_small] * 4, axis=2)
    loss = None

    total_time_training = 0
    num_steps_in_episode = 0
    episode_reward = 0

    done = False
    while not done:
        # update target network once awhile
        if total_t % TARGET_UPDATE_PERIOD == 0:
            target_model.copy_from(model)
            print('Copied training params to target_model. total_t: %s, period: %s' % (total_t, TARGET_UPDATE_PERIOD))
        #/ IF

        # take step
        action = model.sample_action(state, epsilon)
        obs, reward, done, _ = env.step(action)
        obs_small = image_transformer.transform(obs, session)
        next_state = update_state(state, obs_small)

        # keep track of reward for reporting
        episode_reward += reward

        # save, update experience
        experience_replay_buffer.add_experience(action, obs_small, reward, done)

        # train
        time_1 = datetime.now()
        loss = learn(model, target_model, experience_replay_buffer, gamma, batch_size)
        dt = datetime.now() - time_1

        # for reporting
        total_time_training += dt.total_seconds()
        num_steps_in_episode += 1

        state = next_state
        total_t += 1

        epsilon = max(epsilon - epsilon_change, epsilon_min)
    #/ while

    return (total_t, episode_reward, (datetime.now() - time_0),
            num_steps_in_episode, total_time_training/num_steps_in_episode, epsilon)

def smooth(x):
    '''
    smooth last 100
    '''
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        y[i] = x[max(0, i-99):(i+1)].mean()
        # y[i] = float(x[max(0, i-99):(i+1)].sum()) / (i - start + 1)
    return y
#/

def main():
    conv_layer_sizes = [(32, 8, 4),
                        (64, 4, 2),
                        (64, 3, 1)]

    hidden_layer_sizes = [512]
    gamma = 0.99
    batch_size = 32
    num_episodes = 3500  # train for #episodes
    total_t = 0
    experience_replay_buffer = ReplayMemory()
    episode_rewards = np.zeros(num_episodes)

    # epsilon, linear decay
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) / 500000

    env = gym.envs.make('Breakout-v0')
    record_env = get_record_video_env(env)

    image_transformer = ImageTransformer()
    model = DeepQNetwork(K=K, conv_layer_sizes=conv_layer_sizes,
                         hidden_layer_sizes=hidden_layer_sizes, scope='model')
    target_model = DeepQNetwork(K=K, conv_layer_sizes=conv_layer_sizes,
                         hidden_layer_sizes=hidden_layer_sizes, scope='target_model')

    # game play: sample action, train
    with tf.Session() as session:
        model.set_session(session)
        target_model.set_session(session)
        session.run(tf.global_variables_initializer())

        print('populate experience buffer...')
        obs = env.reset()
        for i in range(MIN_EXPERIENCES):
            action = np.random.choice(K)
            obs, reward, done, _ = env.step(action)
            obs_small = image_transformer.transform(obs, session)
            experience_replay_buffer.add_experience(action, obs_small, reward, done)
            if done:
                obs = env.reset()
        #/ for. populate buffer.

        # play and train the model
        time_0 = datetime.now()
        for i in range(num_episodes):
            total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon = \
                play_one(env, session, total_t, experience_replay_buffer,
                         model, target_model, image_transformer,
                         gamma, batch_size, epsilon, epsilon_change, epsilon_min)

            episode_rewards[i] = episode_reward
            last_100_avg = episode_rewards[max(0, i-100): i+1].mean()
            print('Episode:', i,
                  "duration:", duration,
                  '#steps:', num_steps_in_episode,
                  'reward:', episode_reward,
                  'training time per step:', '%.3f'%time_per_step,
                  'avg reward:', '%.3f'%last_100_avg,
                  'epsilon:', '%.3f'%epsilon)
            sys.stdout.flush()
        #/ for # game play episodes
        print('Total duration:', datetime.now() - time_0)

        model.save()
    #/with session

    # plot returns
    y = smooth(episode_rewards)
    plt.plot(episode_rewards, label='original rewords')
    plt.plot(y, label='smoothed')
    plt.legend()
    plt.show()
#/ main()

if __name__ == '__main__':
    main()
