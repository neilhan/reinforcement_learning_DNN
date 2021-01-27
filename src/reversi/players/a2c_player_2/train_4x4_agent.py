import random
import numpy as np
import tensorflow as tf
import logging

from reversi.players.a2c_player_2.GameWrapper import GameWrapperInpatient as GameWrapper
from reversi.players.a2c_player_2.A2CAgentCNN import A2CAgentCNN
from reversi.players.a2c_player_2.A2CAgentV1 import A2CAgentV1
from reversi.game import GameBoard


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        # level=logging.DEBUG)
                        level=logging.INFO)

    # create the Game
    board_size = 4
    game = GameWrapper(1, board_size=board_size)

    # create the agent
    agent_nn = A2CAgentCNN(vision_shape=(board_size, board_size, 1),
                          num_actions=game.get_action_size())
    agent = A2CAgentV1(agent_nn)

    # Train
    agent.train(game, 5, 10_000_000)