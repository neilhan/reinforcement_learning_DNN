import random
import logging
import abc
import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment, tf_environment, tf_py_environment, utils, wrappers, suite_gym
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from othello.game.GameBoard import GameBoard, GameMove, ResultOfAMove, PLAYER_1, PLAYER_2


class OthelloEnv(py_environment.PyEnvironment):
    def __init__(self, board_size=8, random_start=True):
        self._agent = None  # will need to set after Agent created
        self._game: GameBoard = GameBoard(board_size=board_size,
                                          random_start=random_start)
        self._episode_ended = self._game.game_ended
        self._player_id = PLAYER_1
        self._log_on = False
        self._exploring_opponent = True
        self._random_start = random_start

        self._action_spec = array_spec.BoundedArraySpec(shape=(),
                                                        dtype=np.int32,
                                                        minimum=0,
                                                        # 0 - 8*8, including the last, for pass_turn
                                                        maximum=board_size*board_size,
                                                        name='action')
        # self._observation_spec = array_spec.BoundedArraySpec(shape=(9,),
        self._observation_spec = array_spec.BoundedArraySpec(shape=(board_size, board_size, 1),
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
                       dtype=np.float32).reshape(self._game.board_size, self._game.board_size, 1)
        #    dtype=np.float32).reshape(9)
        # flip the -1 or 1 for the spot pieces
        obs = obs * player_id
        # obs = obs/3.0 + 0.5 # todo try without / 3 + 0.5
        return obs

    def _reset(self) -> ts:
        self._game.reset(random_reset=(
            self._random_start and bool(random.getrandbits(1))))
        self._episode_ended = False
        # random player_id
        if self._random_start and bool(random.getrandbits(1)):
            self._player_id = PLAYER_2
            # opponent move...
            opponent_result = self._opponent_take_turn()
            self._game = opponent_result.new_game_board

        return ts.restart(self._get_observation(self._player_id))

    def _step(self, action):
        # action: 0 .. 64.
        #         0 .. 63 are placing a piece.
        #         64 - pass to opponent
        # Reward of this step:
        #     - legal move: 1 * how many pieces flipped
        #     - illegal move: -500
        #     - pass to opponent: -1
        #     - if game is ended, reward: +/- 100 + (my-pieces - opponent-pieces)*2

        # Not seems being used. In case of episode ended, needs reset game
        if self._episode_ended:
            return self.reset()

        if action > self._game.board_size ** 2:
            raise ValueError(
                'action needs to be a valid spot on board. 3x3, from 0 to 8')

        move = GameMove.from_action_code(action, self._game.board_size)
        move_result = self._game.make_a_move(self._player_id,
                                             move)
        self._game = move_result.new_game_board

        if self._log_on:  # logging ---------------------------
            print(
                f'Player {self._player_id}, move: {move.to_friendly_format()}')
            print(self._game)
            print('----------')

        self._episode_ended = move_result.game_ended

        return_ts = None
        # invalid move end game, -500
        if not move_result.is_move_valid:
            return_ts = self._build_ts_invalid_move(move_result)
        else:  # valid move
            if move_result.game_ended:
                return_ts = self._build_ts_game_ended(move_result)
            else:  # game not end
                if move_result.is_move_valid:
                    return_ts = self._build_ts_opponent_turn(move_result)
                else:  # end game and invalid move, very bad reward
                    return_ts = self._build_ts_invalid_move(move_result)

        return return_ts

    def _is_player_winning(self, player_id):
        p1_winning_count = self._game.player_1_count - self._game.player_2_count

        if (player_id * p1_winning_count) > 0:
            return True
        else:
            return False

    def _is_game_tie(self):
        p1_winning_count = self._game.player_1_count - self._game.player_2_count
        return (p1_winning_count == 0)

    def _build_ts_invalid_move(self, move_result: ResultOfAMove):
        reward = -500.0
        self._episode_ended = True
        return_ts = \
            ts.termination(self._get_observation(self._player_id),
                           reward=reward)
        return return_ts

    def _build_ts_game_ended(self, move_result: ResultOfAMove):
        reward = 0  # default to tie
        if self._is_player_winning(self._player_id):
            reward = 100 + abs(self._game.player_1_count -
                               self._game.player_2_count) * 2
        elif self._is_game_tie():
            reward = 0
        else:
            reward = -100 - abs(self._game.player_1_count -
                                self._game.player_2_count) * 2

        # return termination time_step
        return_ts = \
            ts.termination(self._get_observation(self._player_id),
                           reward)
        return return_ts

    def _build_ts_opponent_turn(self, move_result: ResultOfAMove):
        return_ts = None
        opponent_result = self._opponent_take_turn()
        # opponent turn ended ---
        self._episode_ended = opponent_result.game_ended
        if opponent_result.game_ended:
            return_ts = self._build_ts_game_ended(opponent_result)
        else:
            return_ts = ts.transition(self._get_observation(self._player_id),
                                      reward=1,  # alway reward _player_id with 1 if a move is valid
                                      discount=1.0)

        return return_ts

    def _opponent_take_turn(self) -> ResultOfAMove:
        # player_2 move
        opponent_player_id = self._game.get_next_player(self._player_id)
        if opponent_player_id == self._player_id:
            # opponent_has no moves. so return current state, reward and done.
            opponent_move = GameMove(pass_turn=True)
        else:  # otherwise, player_2, play a move.
            # prepare a time_step for player_2.
            # x 1 or x -1, so the observation will be: 1 for my piece, -1 as opponent's
            opponent_ts = \
                ts.transition(self._get_observation(opponent_player_id)[None, :],
                              reward=1,  # no use.
                              discount=1.0)
            if opponent_player_id == PLAYER_1:
                valid_spots = self._game.possible_moves_player_1
            else:
                valid_spots = self._game.possible_moves_player_2

            if (self._agent == None
                or (self._exploring_opponent
                    and bool(random.choice([True, False, False, False])))):
                if self._log_on:
                    print('******* Opponent randome exploring.')
                opponent_move = GameMove(random.choice(valid_spots))
            else:
                # Let agent pick a move
                action_step = self._agent.policy.action(opponent_ts)
                opponent_action_code = action_step.action.numpy().item()
                opponent_move = GameMove.from_action_code(
                    opponent_action_code, board_size=self._game.board_size)
                if not opponent_move.pass_turn and not opponent_move.spot in valid_spots:
                    if self._log_on:
                        print('******* policy failed. randome logic ******* ')
                    opponent_move = GameMove(random.choice(valid_spots))
        # opponent move...
        opponent_result = self._game.make_a_move(opponent_player_id,
                                                 opponent_move)
        self._game = opponent_result.new_game_board

        if self._log_on:  # logging ---------------------------
            print(
                f'Opponent {opponent_player_id}, move: {opponent_move.to_friendly_format()}. dbg: game player: {self._player_id}')
            print(self._game)
            print('----------------------------------------')

        return opponent_result


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        # level=logging.DEBUG)
                        level=logging.INFO)

    env = OthelloEnv()
    utils.validate_py_environment(env, episodes=5)
