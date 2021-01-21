import random
import numpy as np
import tensorflow as tf
import logging

from reversi.players.a2c_player_1.GameWrapper_simple import GameWrapper
from reversi.players.a2c_player_1.A2CAgentNN import A2CAgentNN
from reversi.players.a2c_player_1.A2CAgent import A2CAgent
from reversi.game import GameBoard


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        # level=logging.DEBUG)
                        level=logging.INFO)

    # create the Game
    board_size = 8
    game = GameWrapper(1, board_size=board_size)

    # create the agent
    agent_nn = A2CAgentNN(action_size=game.get_action_size(),
                          input_size=game.get_observation_size())
    agent = A2CAgent(agent_nn)

    # Train
    agent.train(game, 50, 100)