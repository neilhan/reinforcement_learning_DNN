import random

import reversi.GameBoard as GameBoard

class RandomPlayer:

    def __init__(self, player_id):
        self.player_id = player_id

    def pick_next_move(self, board: GameBoard.GameBoard) -> GameBoard.Spot:
        choices = board.get_valid_spots(self.player_id)
        return random.choice(choices)