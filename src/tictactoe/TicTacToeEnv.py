import random
import logging
import abc
import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment, tf_environment, tf_py_environment, utils, wrappers, suite_gym
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from tictactoe.GameBoard import GameBoard, Spot, ResultOfAMove, PLAYER_1, PLAYER_2


class TicTacToeEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._agent = None  # will need to set after Agent created
        self._game = GameBoard(board_size=3)
        self._episode_ended = False
        self._player_id = PLAYER_1
        self._log_on = False
        self._exploring_opponent = True
        self._random_start = True

        self._action_spec = array_spec.BoundedArraySpec(shape=(),
                                                        dtype=np.int32,
                                                        minimum=0,
                                                        maximum=8,
                                                        name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(9,),
                                                             # self._observation_spec = array_spec.BoundedArraySpec(shape=(3, 3, 1),
                                                             dtype=np.float32,
                                                             minimum=-1.0,
                                                             maximum=1.0,
                                                             name='observation')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _get_observation(self, player_id):
        obs = np.array(self._game.observe_board_2d(),
                       #    dtype=np.float32).reshape(3, 3, 1)
                       dtype=np.float32).reshape(9)
        # flip the -1 or 1 for the spot pieces
        obs = obs * player_id
        # obs = obs/3.0 + 0.5 # todo try without / 3 + 0.5
        return obs

    def _reset(self) -> ts:
        self._game.reset()
        self._episode_ended = False
        if self._random_start and bool(random.getrandbits(1)):
            # play a random start piece
            opponent_move = random.choice(self._game.get_valid_spots())
            opponent_result = self._game.play_a_piece(self._player_id,
                                                      opponent_move)
            self._player_id = self._game.get_next_player(self._player_id)

        return ts.restart(self._get_observation(self._player_id))

    def _step(self, action):
        # Not seems being used. In case of episode ended, needs reset game
        if self._episode_ended:
            return self.reset()

        if action >= self._game.board_size ** 2:
            raise ValueError(
                'action needs to be a valid spot on board. 3x3, from 0 to 8')

        move = Spot.from_action_code(action, self._game.board_size)
        move_result = self._game.play_a_piece(self._player_id,
                                              move)

        if self._log_on:  # logging ---------------------------
            print(
                f'Player {self._player_id}, move: {move.to_friendly_format()}')
            print(self._game)
            print('----------')

        self._episode_ended = move_result.game_ended

        return_ts = None
        if move_result.game_ended:
            return_ts = self._get_ts_game_ended(move_result)
        else:  # game not end
            if move_result.is_move_valid:
                return_ts = self._get_ts_opponent_turn(move_result)
            else:  # end game and invalid move, very bad reward
                return_ts = self._get_ts_invalid_move(move_result)

        return return_ts

    def _get_ts_invalid_move(self, move_result: ResultOfAMove):
        reward = -20
        self._episode_ended = True
        return_ts = \
            ts.termination(self._get_observation(self._player_id),
                           reward=reward)
        return return_ts

    def _get_ts_game_ended(self, move_result: ResultOfAMove):
        reward = 0  # default to tie
        if move_result.player_won:
            reward = 10
        # return termination time_step
        return_ts = \
            ts.termination(self._get_observation(self._player_id),
                           reward)
        return return_ts

    def _get_ts_opponent_game_ended(self, opponent_move_result: ResultOfAMove):
        if opponent_move_result.player_won:
            reward = -10
        else:  # tie ---
            reward = 0
        self._episode_ended = True
        return_ts = ts.termination(self._get_observation(self._player_id),
                                   reward=reward)
        return return_ts

    def _get_ts_opponent_turn(self, move_result: ResultOfAMove):
        return_ts = None
        reward = 1
        # player_2 move
        opponent_player_id = self._game.get_next_player(self._player_id)
        # prepare a time_step for player_2.
        # x 1 or x -1, so the observation will be: 1 for my piece, -1 as opponent's
        opponent_ts = \
            ts.transition(self._get_observation(opponent_player_id)[None, :],
                          reward=reward,
                          discount=1.0)

        if (self._agent == None
            or (self._exploring_opponent
                and bool(random.choice([True, False, False, False])))):
            if self._log_on:
                print('******* Opponent random exploring.')
            opponent_move = random.choice(self._game.get_valid_spots())
        else:
            action_step = self._agent.policy.action(opponent_ts)
            opponent_move = action_step.action.numpy().item()
            opponent_move = Spot.from_action_code(
                opponent_move, board_size=self._game.board_size)
            if not opponent_move in self._game.get_valid_spots():
                if self._log_on:
                    print('******* policy failed. random logic ******* ')
                opponent_move = random.choice(
                    self._game.get_valid_spots())
        opponent_result = self._game.play_a_piece(opponent_player_id,
                                                  opponent_move)

        self._episode_ended = opponent_result.game_ended
        if opponent_result.game_ended:
            return_ts = self._get_ts_opponent_game_ended(opponent_result)
        else:
            return_ts = ts.transition(self._get_observation(self._player_id),
                                      reward=reward,
                                      discount=1.0)
        if self._log_on:  # logging ---------------------------
            print(
                f'Player {opponent_player_id}, move: {opponent_move.to_friendly_format()}. dbg: game player: {self._player_id}')
            print(self._game)
            print('----------------------------------------')

        return return_ts


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        # level=logging.DEBUG)
                        level=logging.INFO)

    env = TicTacToeEnv()
    utils.validate_py_environment(env, episodes=5)
