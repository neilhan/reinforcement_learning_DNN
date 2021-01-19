from __future__ import annotations

import numpy as np

from reversi.game import GameBoard


class GameWrapper:
    # Game play related state
    def __init__(self, id: int):
        self.id = id
        self.reset()

    def reset(self) -> np.ndarray:
        # return (8*8,) float32
        self.game_board = GameBoard.GameBoard()
        self.current_player = GameBoard.PLAYER_1
        self.game_ended = False
        return self.observe(GameBoard.PLAYER_1)

    def observe(self, as_player_id) -> np.ndarray:
        # return np.ndarray of (8*8,) 64 1d array.
        # 0, is empty; 1 - is the player's piece; -1 - is opponent.
        flat = self.game_board.observe_board_1d()
        # if as player1, 1 x 1, no impact
        # if as player2, 1 x -1, flips all the pieces, so the 1 is my color
        return np.asarray(flat) * as_player_id

    def execute_move(self, action: int) -> Tuple[np.ndarray, float, bool, GameWrapper]:
        # action: 0 .. 64. 
        #         0 .. 63 are placing a piece.
        #         64 - pass to opponent
        # returns: 1. flat observation, shape (64,)
        #          2. reture of this step:
        #                 legal move: 0.01 * how many pieces flipped
        #                 illegal move: -1
        #                 if game is done, return the my-pieces - opponent-pieces
        #          3. Ture/False game ended
        #          4. new state of the Game, as GameWrapper
        # will flip pieces, then switch player, set current_player to next.
        if action < 64: # execute a move, place a piece
            spot = Spot(row, col)
            move_result = self.game_board.get_new_board_for_a_move(
                self.current_player, spot)

            observation = self.observe(self.current_player)
            game_ended = self.game_board.game_ended
            reture_of_this_move = 0.0

            if game_ended:
                if self.current_player == GameBoard.PLAYER_1:
                    return_of_this_move = self.game_board.player_1_count - \
                        self.game_board.player_2_count
                else:
                    return_of_this_move = self.game_board.player_2_count - \
                        self.game_board.player_1_count
            else:
                if not move_result.is_move_valid:
                    return_of_this_move = -1.0
                else:
                    return_of_this_move = len(move_result.flipped_spots) * 0.01

            # update to new state. Switch player etc
            self.game_board = move_result.new_game_board
            self.current_player = self.game_board.get_next_player(
                self.current_player)
            self.game_ended = self.game_board.game_ended

            return (observation, return_of_this_move, game_ended, self)
        else: # pass to opponent
            ????
