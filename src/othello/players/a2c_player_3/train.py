
from typing import Any, List, Sequence, Tuple
import numpy as np
import tensorflow as tf

import gym

from othello.tf_utils import set_global_seeds
from othello.players.a2c_player_3.A2CModel import A2CModel
from othello.players.a2c_player_3.A2CTrainer import A2CTrainer



def do_training():
    set_global_seeds(0)

    num_actions = 2
    num_hidden_units = 128

    model = A2CModel(num_actions, num_hidden_units)

    import gym
    # Create the environment
    env = gym.make("CartPole-v0")

    trainer = A2CTrainer(env)

    trainer.train(model)

    model.save('./__models__/player_3/')

    return model


if __name__ == '__main__':
    model = do_training()
