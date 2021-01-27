import multiprocessing
import random
import numpy as np
import tensorflow as tf
import logging

from reversi.players.a2c_player_2.GameWrapper import GameWrapperInpatient
from reversi.players.a2c_player_2.GameWrapperParallel import GameWrapperParallel
from reversi.players.a2c_player_2.A2CAgentCNN import A2CAgentCNN
from reversi.players.a2c_player_2.A2CAgentV2 import A2CAgentV2


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        # level=logging.DEBUG)
                        level=logging.INFO)

    # num_sub_processes = multiprocessing.cpu_count()
    num_sub_processes = 1
    board_size = 6

    # create the agent
    agent_nn = A2CAgentCNN(num_actions=board_size*board_size + 1,
                           vision_shape=(board_size, board_size, 1))
    agent = A2CAgentV2(
        agent_nn,
        model_save_path='./__models__/a2c_player_2/', model_save_interval=1000,
        learn_rate=0.007,
        ent_coef=0.5,
        value_coef=0.5,
        gamma=0.99,
    )

    game_wrapper = GameWrapperParallel(agent, num_sub_processes, board_size)

    # Train ----------
    agent.train(game_wrapper, 5, 10000)
