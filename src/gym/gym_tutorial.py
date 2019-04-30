#! python
# -*- coding: utf-8 -*-

import gym

import time

env = gym.make('CartPole-v0')
state = env.reset()

print('state:', state)

# what's observation_space
box = env.observation_space
print('env.observation_space:', box)

# a few random actions
done = False

i = 0
while not done:
    env.render()
    print('act:', i)
    observation, reward, done, info = env.step(env.action_space.sample())  # random action
    print('observation:', observation, 'reward:', reward,
          'done:', done, 'info:', info)
    i += 1
    time.sleep(0.1)

print('random done')

print('wait 10 sec')
time.sleep(10)
print('exit')
