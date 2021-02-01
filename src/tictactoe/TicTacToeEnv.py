import random
import logging
import abc
import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment, tf_environment, tf_py_environment, utils, wrappers, suite_gym
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from tictactoe.GameBoard import GameBoard, Spot, PLAYER_1, PLAYER_2


class TicTacToeEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._game = GameBoard(board_size=3)
        self._episode_ended = False
        self._player_id = PLAYER_1
        self._log_on = False

        self._action_spec = array_spec.BoundedArraySpec(shape=(),
                                                        dtype=np.int32,
                                                        minimum=0,
                                                        maximum=8,
                                                        name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(9,),
                                                             dtype=np.float32,
                                                             minimum=-1.0,
                                                             maximum=1.0,
                                                             name='observation')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self) -> ts:
        self._game.reset()
        self._episode_ended = False
        return ts.restart(np.array(self._game.observe_board_1d(), dtype=np.float32))

    def _step(self, action):
        if self._log_on:
            logging.info('------------------------------------')
            logging.info(f'Player: {self._player_id} action: {Spot.from_action_code(action, self._game.board_size).to_friendly_format()}')
            logging.info(self._game)
            logging.info('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        # Not seems being used. In case of episode ended, needs reset game
        if self._episode_ended:
            return self.reset()

        if action >= self._game.board_size ** 2:
            raise ValueError(
                'action needs to be a valid spot on board. 3x3, from 0 to 8')

        result = self._game.play_a_piece(self._player_id,
                                         Spot.from_action_code(action, self._game.board_size))

        self._episode_ended = result.game_ended

        if result.game_ended:
            if result.player_won:
                reward = 10
            else:  # tie
                reward = 0
            # return termination time_step
            return ts.termination(np.array(self._game.observe_board_1d(), dtype=np.float32),
                                  reward)
        else:  # game not end
            if result.is_move_valid:
                reward = 1
                # player_2 move
                opponent_player_id = self._game.get_next_player(
                    self._player_id)
                opponent_move = random.choice(self._game.get_valid_spots())
                opponent_result = self._game.play_a_piece(opponent_player_id,
                                                          opponent_move)

                self._episode_ended = opponent_result.game_ended
                if opponent_result.game_ended and opponent_result.player_won:
                    reward = -10
                    self._episode_ended = True
                    return ts.termination(np.array(self._game.observe_board_1d(), dtype=np.float32),
                                          reward=reward)

                return ts.transition(np.asarray(self._game.observe_board_1d(), dtype=np.float32),
                                     reward=reward,
                                     discount=1.0)
            else:  # end game and invalid move, very bad reward
                reward = -20
                self._episode_ended = True
                return ts.termination(np.array(self._game.observe_board_1d(), dtype=np.float32),
                                      reward=reward)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        # level=logging.DEBUG)
                        level=logging.INFO)

    env = TicTacToeEnv()
    utils.validate_py_environment(env, episodes=5)
