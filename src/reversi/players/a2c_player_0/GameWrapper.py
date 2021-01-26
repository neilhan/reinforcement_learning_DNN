from __future__ import annotations

import random
import logging
from reversi.players.a2c_player_0.A2CAgentCNN import VisionShape
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
        self.game_board = GameBoard.GameBoard(self.board_size)
        self.current_player = GameBoard.PLAYER_1
        self.game_ended = False
        self.is_log_play = False
        return self.observe(GameBoard.PLAYER_1)
    
    def get_action_size(self) -> int:
        return self.board_size * self.board_size + 1

    def get_vision_shape(self)-> VisionShape:
        return VisionShape(self.board_size, self.board_size, 1)

    def observe(self, as_player_id) -> np.ndarray:
        # return np.ndarray of (8*8,) 64 1d array.
        # 0, is empty; 1 - is the player's piece; -1 - is opponent.
        obs_2d = np.asarray(self.game_board.observe_board_2d())
        obs_2d_1_color = obs_2d[None,:]
        # if as player1, 1 x 1, no impact
        # if as player2, 1 x -1, flips all the pieces, so the 1 is my color
        return obs_2d_1_color * as_player_id

    @tf.function
    def execute_move(self, action: int) -> Tuple[np.ndarray, float, int, GameWrapper, bool]:
        # action: 0 .. 64.
        #         0 .. 63 are placing a piece.
        #         64 - pass to opponent
        # Returns the tuple: (observation, reward of this step,
        #                     done in int, new GameWrapper object,
        #                     move_valid)
        #         1. observation: flat observation of the game board, shape (64,)
        #         2. reward of this step:
        #            - legal move: 1 * how many pieces flipped
        #            - illegal move: -1
        #            - pass to opponent: -2
        #            - if game is done, reward: 100 * (my-pieces - opponent-pieces)
        #         3. done: game ended. 0-not, 1- ended.
        #         4. new state of the Game, as GameWrapper
        #         5. is valid move? True/False
        # This method will flip pieces, then switch player, set current_player to next.
        # The returned observation is for the next player. For the convenience during training.
        if action < self.PASS_TURN_ACTION:
            # execute a move, place a piece
            spot = self.convert_action_to_spot(action)
            move_result = self.game_board.get_new_board_for_a_move(
                self.current_player, spot)
            self.game_board = move_result.new_game_board
            game_ended = self.game_board.game_ended

            # reward_of_this_move
            if game_ended:
                if self.current_player == GameBoard.PLAYER_1:
                    reward_of_this_move = (self.game_board.player_1_count -
                                           self.game_board.player_2_count)
                else:
                    reward_of_this_move = (self.game_board.player_2_count -
                                           self.game_board.player_1_count)
                if reward_of_this_move > 0:
                    reward_of_this_move = 10 * reward_of_this_move + 10
                else:
                    reward_of_this_move = 10 * reward_of_this_move - 10
            else:
                if move_result.is_move_valid:
                    reward_of_this_move = 1.5
                else:
                    reward_of_this_move = -0.1  # invalid move

            # update to new state. Switch player etc
            if move_result.is_move_valid:  # next player
                self.current_player = self.game_board.get_next_player(
                    self.current_player)
            self.game_ended = self.game_board.game_ended
            # observation for next player
            observation = self.observe(self.current_player)
            is_move_valid = move_result.is_move_valid
        else:  # pass, turn to opponent
            # no move, so just observe.
            # the current_player, not next player, observation
            num_possible_moves = len(
                self.game_board.get_valid_spots(self.current_player))
            reward_of_this_move = -2 * num_possible_moves
            game_ended = self.game_board.game_ended
            self.current_player = self.game_board.get_next_player(
                self.current_player)
            observation = self.observe(self.current_player)
            spot = None
            is_move_valid = True

        if self.is_log_play:
            self.log_move(spot, self)
        return (observation, reward_of_this_move, game_ended, self, is_move_valid)

    def get_all_valid_moves(self):
        valid_moves = self.game_board.get_valid_spots(self.current_player)
        return tf.convert_to_tensor(np.asarray(list(map(self.convert_spot_to_action, valid_moves))))

    def set_log_play(self, is_on):
        self.is_log_play = is_on

    def log_move(self, spot: GameBoard.Spot, game: GameWrapper):
        if spot is None:  # pass
            logging.info('Previous move: PASS')
        else:
            logging.info('Previous move: ' + spot.to_friendly_format())
        logging.info('Current player: %1d' % game.current_player)
        logging.info('game:' + str(self.game_board))

    def pick_a_random_valid_move(self) -> int:
        # return action
        choices = self.game_board.get_valid_spots(self.current_player)
        return self.convert_spot_to_action(random.choice(choices))

    def convert_action_to_spot(self, action: int):
        # if action < 0 or action >= self.get_action_size():
        #     raise ValueError('Action needs to be between [0, 64), received {0}' % action)

        row = int(action / self.board_size)
        col = int(action % self.board_size)  # remainder
        spot = GameBoard.Spot(row, col, self.board_size)
        return spot

    def convert_spot_to_action(self, spot: GameBoard.Spot) -> int:
        return spot.row * spot.board_size + spot.col
