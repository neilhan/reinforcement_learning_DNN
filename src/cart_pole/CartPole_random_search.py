#! python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import matplotlib.pyplot as plt


def get_action(observ, weights):
    return 1 if observ.dot(weights) > 0 else 0


def play_one_episode(env, params, is_render=False):
    observation = env.reset()
    done = False
    t = 0

    while not done and t < 1000:
        if is_render:
          env.render() # debugging ------------

        t += 1
        action = get_action(observation, params)
        observation, reward, done, info = env.step(action)
        if done:
            break
    return t


def play_multiple_episodes(env, T, params, is_render=False):
    episode_lengths = np.empty(T)

    for i in range(T):
        episode_lengths[i] = play_one_episode(env, params, is_render)

    avg_length = episode_lengths.mean()
    print('avg length:', avg_length)
    return avg_length


def random_search(env):
    episode_lengths = []
    best = 0
    params = None
    for t in range(100):
        new_params = np.random.random(4) * 2 - 1
        avg_length = play_multiple_episodes(env, 100, new_params)
        episode_lengths.append(avg_length)

        if avg_length > best:
            params = new_params
            best = avg_length
    return episode_lengths, params


def main():
    env = gym.make('CartPole-v0')
    episode_lengths, params = random_search(env)
    plt.plot(episode_lengths)
    plt.show()

    print('final run with final weights')
    play_multiple_episodes(env, 10, params, True)


if __name__=='__main__':
    main()
