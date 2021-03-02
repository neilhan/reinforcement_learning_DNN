import random
import logging
import abc
import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment, tf_environment, tf_py_environment, utils, wrappers, suite_gym
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from othello.game.GameBoard import GameBoard, GameMove, ResultOfAMove, PLAYER_1, PLAYER_2
from othello.service import tfagent_client as agent_client


def load_policy(policy_dir):
    try:
        saved_policy = tf.compat.v2.saved_model.load(policy_dir)
        return saved_policy
    except:
        print('load policy failed', policy_dir)
    return None


class OthelloEnv(py_environment.PyEnvironment):
    # existing_agent_policy_path=None):
    def __init__(self, board_size=8,
                 random_rate=0.0,
                 as_player_2_rate=0.0,
                 use_agent_service=False):
        self._agent = None  # will need to set after Agent created
        self._use_agent_service = use_agent_service
        self._game: GameBoard = GameBoard(board_size=board_size,
                                          random_start=random.random() < random_rate)
        self._episode_ended = self._game.game_ended
        self._player_id = PLAYER_1
        self._log_on = False
        self._random_rate = random_rate
        self._as_player_2_rate = as_player_2_rate

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

    def _get_observation(self, player_id, game: GameBoard = None):
        if game is None:
            game = self._game
        obs = np.array(game.observe_board_2d(),
                       dtype=np.float32).reshape(game.board_size, game.board_size, 1)
        #    dtype=np.float32).reshape(9)
        # flip the -1 or 1 for the spot pieces
        obs = obs * player_id
        # obs = obs/3.0 + 0.5 # todo try without / 3 + 0.5
        obs = tf.convert_to_tensor(obs, dtype=tf.float32, name='observation')
        return obs

    def _reset(self) -> ts:
        self._game = GameBoard(board_size=self._game.board_size,
                               random_start=random.random() < self._random_rate)
        self._episode_ended = False
        self._player_id = PLAYER_1
        # random player_id
        # if random.random() < self._random_rate:
        if random.random() < self._as_player_2_rate:
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
            print('--------')

        self._episode_ended = move_result.game_ended

        return_ts = None
        # invalid move end game, -500
        if not move_result.is_move_valid:
            if self._log_on:  # logging ---------------------------
                print(
                    f'Player {self._player_id}, INVALID move: {move.to_friendly_format()}. Game Ends.')
                print('^^^^^^^^^^^^^^^^^******^^^^^^^^^^^^^^^^^^^')
            return_ts = self._build_ts_invalid_move(move_result)
        else:  # valid move
            if move_result.game_ended:
                return_ts = self._build_ts_game_ended(move_result)
                if self._log_on:  # logging ---------------------------
                    print(
                        f'Player {self._player_id}, move: {move.to_friendly_format()}. Game Ends.')
                    print('^^^^^^^^^^^^^^^^^******^^^^^^^^^^^^^^^^^^^')
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
        game = move_result.new_game_board
        reward = -500.0
        self._episode_ended = True
        return_ts = \
            ts.termination(self._get_observation(self._player_id, game),
                           reward=reward)
        return return_ts

    def _build_ts_game_ended(self, move_result: ResultOfAMove):
        game = move_result.new_game_board
        reward = 0  # default to tie
        win_extra_reward = (abs(game.player_1_count - game.player_2_count) * 100
                            / (game.player_1_count + game.player_2_count))
        if self._is_player_winning(self._player_id):
            reward = 100 + win_extra_reward
        elif self._is_game_tie():
            reward = 0
        else:
            reward = -100 - win_extra_reward

        # return termination time_step
        return_ts = \
            ts.termination(self._get_observation(self._player_id, game),
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
            return_ts = ts.transition(self._get_observation(self._player_id,
                                                            opponent_result.new_game_board),
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
            # opponent_ts = self.current_time_step()
            # --
            # step_type = tf.convert_to_tensor(
            #     [0], dtype=tf.int32, name='step_type')
            # reward = tf.convert_to_tensor([ts.StepType.MID], dtype=tf.float32, name='reward')
            # discount = tf.convert_to_tensor(
            #     [1.0], dtype=tf.float32, name='discount')
            # # tf.convert_to_tensor( [state], dtype=tf.float32, name='observations')
            # observation = self._get_observation(opponent_player_id)[None, :]
            # opponent_ts = ts.TimeStep(
            #     step_type, reward, discount, observation)
            opponent_ts = \
                ts.transition(self._get_observation(opponent_player_id)[None, :],
                              reward=1,  # no use.
                              discount=1.0)
            # opponent_ts = ts.TimeStep(opponent_ts.step_type,
            #                           opponent_ts.reward,
            #                           opponent_ts.discount,
            #                           opponent_ts.observation * self._player_id)

            if opponent_player_id == PLAYER_1:
                valid_spots = self._game.possible_moves_player_1
            else:
                valid_spots = self._game.possible_moves_player_2

            if len(valid_spots) == 0:
                opponent_move = GameMove(pass_turn=True)
            elif self._agent == None and not self._use_agent_service:
                if self._log_on:
                    print('******* Opponent random move.')
                opponent_move = GameMove(random.choice(valid_spots))
            else:
                # Let agent pick a move
                if self._use_agent_service:
                    opponent_action_code = \
                        agent_client.agent_service_step(game_board=self._game.observe_board_2d(),
                                                        server_player_id=opponent_player_id,
                                                        client_player_id=self._player_id,
                                                        board_size=self._game.board_size)
                else:
                    action_step = self._agent.policy.action(opponent_ts)
                    opponent_action_code = action_step.action.numpy().item()
                opponent_move = GameMove.from_action_code(
                    opponent_action_code, board_size=self._game.board_size)
                if ((opponent_move.pass_turn and len(valid_spots) > 0) or
                        (not opponent_move.pass_turn and not opponent_move.spot in valid_spots)):
                    if self._log_on:
                        print('******* policy failed. random logic ******* ',
                              opponent_move.to_friendly_format())
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


# if __name__ == '__main__':
#     logging.basicConfig(format='%(levelname)s:%(message)s',
#                         # level=logging.DEBUG)
#                         level=logging.INFO)

#     env = OthelloEnv()
#     utils.validate_py_environment(env, episodes=5)
