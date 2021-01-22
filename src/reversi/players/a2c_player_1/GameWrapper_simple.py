from __future__ import annotations

import random
import logging
import numpy as np
import tensorflow as tf

from reversi.game import GameBoard


class GameWrapper:
    # Game play related state
    def __init__(self, id: int, board_size=8):
        self.id = id
        self.board_size = board_size
        self.PASS_TURN_ACTION = self.board_size * self.board_size
        self.reset()

    def reset(self) -> np.ndarray:
        # returns the observation of the board in one-d array. shape: (8*8,) float32
        self.game_board = [0.0] * self.board_size
        self.game_board[2] = GameBoard.PLAYER_1
        self.game_ended = False
        self.is_log_play = False
        self.current_player = GameBoard.PLAYER_1
        return self.observe(GameBoard.PLAYER_1)

    def get_action_size(self) -> int:
        return self.board_size

    def get_observation_size(self) -> int:
        return self.board_size

    def observe(self, as_player_id) -> np.ndarray:
        return np.asarray(self.game_board)

    def execute_move(self, action: int) -> Tuple[np.ndarray, float, int, GameWrapper, bool]:
        # action: place a piece, must be by another piece. When > 2 on board,
        #         the opposite of the new piece will be removed
        #         When the new piece is at [7] game ends. Reward 100
        # Returns the tuple: (observation, reward of this step,
        #                     done in int, new GameWrapper object,
        #                     move_valid)
        #         1. observation: flat observation of the game board
        #         2. reward of this step:
        #            - legal move: -1
        #            - illegal move: -2
        #            - if game is done, reward: 100
        #         3. done: game ended. 0-not, 1- ended.
        #         4. new state of the Game, as GameWrapper
        #         5. is valid move? True/False
        # The returned observation is for the next player. For the convenience during training.
        if action >= self.board_size:
            action = self.board_size - 1

        # for player 2, do nothing
        if self.current_player == GameBoard.PLAYER_2:
            reward_of_this_move = 0
            game_ended = self.game_ended
            self.current_player = -1 * self.current_player
            observation = self.observe(self.current_player)
            is_move_valid = True

            return (observation, reward_of_this_move, game_ended, self, is_move_valid)
        # find out is valid
        is_move_valid = False
        if not is_move_valid and (action >= 0 & action < 8):
            if action > 0 and self.game_board[action-1] == 1:
                is_move_valid = True
            if not is_move_valid and action < 7 and self.game_board[action+1] == 1:
                is_move_valid = True
        if self.game_board[action] != 0:
            is_move_valid = False

        if is_move_valid:
            self.game_board[action] = self.current_player

            if action == 7:
                game_ended = 1
            else:
                game_ended = 0

            # set left 3rd to 0
            if action - 2 >= 0:
                self.game_board[action-2] = 0.0
            # set right 3rd to 0
            if action + 2 <= 7:
                self.game_board[action+2] = 0.0

            # reward_of_this_move
            if game_ended:
                reward_of_this_move = 10.0
            else:
                reward_of_this_move = -0.1 * (8-action)

            # update to new state. Switch player etc
            self.current_player = -1 * self.current_player
            self.game_ended = game_ended
            # observation for next player
            observation = self.observe(self.current_player)
        else:
            # no valid move, so just observe.
            # the current_player, not next player, observation
            if self.game_board[action] != 0:
                reward_of_this_move = -4
            else:
                reward_of_this_move = -2
            game_ended = self.game_ended
            self.current_player = self.current_player
            observation = self.observe(self.current_player)
            is_move_valid = True

        if self.is_log_play:
            self.log_move(action, self)
        return (observation, reward_of_this_move, game_ended, self, is_move_valid)

    def get_all_valid_moves(self):
        return []

    def set_log_play(self, is_on):
        self.is_log_play = is_on

    def log_move(self, action: int, game: GameWrapper):
        logging.info('Previous move: ' + str(action))
        logging.info('Current player: %1d' % game.current_player)
        logging.info('game:' + str(self.game_board))

    def pick_a_random_valid_move(self) -> int:
        return 1

    def convert_action_to_spot(self, action: int):
        return action

    def convert_spot_to_action(self, spot: int) -> int:
        return spot
