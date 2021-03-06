#! python
# -*- coding: utf-8 -*-

import gym

env = gym.make('CartPole-v0')
state = env.reset()

print('state:', state)

done = False

i = 0
while not done:
    env.render()
    print('act:', i)
    observation, reward, done, info = env.step(env.action_space.sample())  # random action
    i += 1
print('random done')
