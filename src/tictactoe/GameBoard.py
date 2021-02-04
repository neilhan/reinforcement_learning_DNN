# from __future__ import annotations
import random

import itertools
from itertools import product
import copy

PLAYER_1 = 1.0
PLAYER_2 = -PLAYER_1

# note, 6, 8 size board are supported. Not supporting > 8 board

COL_MAP = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
}

COL_MAP_TO_CHAR = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']


# player_id: 1 or -1
def get_opponent_player_id(player_id):
    return -player_id


class Spot:
    # note, 6, 8 size board are supported. Not supporting > 8 board
    def __init__(self, row, col, board_size=8):
        self.row = row
        self.col = col
        self.board_size = board_size
        # if row < board_size and row >= 0 and col < board_size and col >= 0:
        #     self.row = row
        #     self.col = col
        #     self.board_size = board_size
        # else:
        #     raise ValueError('row and col needs to be between [0, board_size)')

    @staticmethod
    def from_action_code(action, board_size=8):
        row = action // board_size
        col = action % board_size
        return Spot(row, col, board_size)

    def from_friendly_format(self, friendly_format: str) -> 'Spot':
        # returns Spot
        if len(friendly_format) < 2:
            raise ValueError('At lease 2 characters. 1a, 1b, or 7h')
        row = int(friendly_format[:1]) - 1
        col = COL_MAP[friendly_format[1:2]]
        return Spot(row, col, self.board_size)

    # return str, for example: 1A, 6B
    def to_friendly_format(self):
        return '{0}{1}'.format(self.row + 1, COL_MAP_TO_CHAR[self.col])

    def is_outside(self) -> bool:
        return (self.row < 0 or self.row >= self.board_size
                or self.col < 0 or self.col >= self.board_size)

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __repr__(self):
        return "Spot(%i, %i)" % (self.row, self.col)

    def __str__(self):
        return "Spot(%i, %i)" % (self.row, self.col)

    def step_left(self) -> 'Spot':
        new_spot = Spot(self.row, self.col, self.board_size)
        new_spot.col = new_spot.col - 1
        return new_spot

    def step_right(self) -> 'Spot':
        new_spot = Spot(self.row, self.col, self.board_size)
        new_spot.col = new_spot.col + 1
        return new_spot

    def step_up(self) -> 'Spot':
        new_spot = Spot(self.row, self.col, self.board_size)
        new_spot.row = new_spot.row - 1
        return new_spot

    def step_down(self) -> 'Spot':
        new_spot = Spot(self.row, self.col, self.board_size)
        new_spot.row = new_spot.row + 1
        return new_spot

    def step_up_left(self) -> 'Spot':
        new_spot = Spot(self.row, self.col, self.board_size)
        new_spot.row = new_spot.row - 1
        new_spot.col = new_spot.col - 1
        return new_spot

    def step_up_right(self) -> 'Spot':
        new_spot = Spot(self.row, self.col, self.board_size)
        new_spot.row = new_spot.row - 1
        new_spot.col = new_spot.col + 1
        return new_spot

    def step_down_left(self) -> 'Spot':
        new_spot = Spot(self.row, self.col, self.board_size)
        new_spot.row = new_spot.row + 1
        new_spot.col = new_spot.col - 1
        return new_spot

    def step_down_right(self) -> 'Spot':
        new_spot = Spot(self.row, self.col, self.board_size)
        new_spot.row = new_spot.row + 1
        new_spot.col = new_spot.col + 1
        return new_spot


class ResultOfAMove:
    def __init__(self,
                 new_game_board: 'GameBoard',
                 new_spot: Spot,
                 is_move_valid: bool,
                 game_ended: bool,
                 won: bool):
        self.new_game_board = new_game_board
        self.new_spot = new_spot
        self.is_move_valid = is_move_valid
        self.game_ended = game_ended
        self.player_won = won

    def _to_str(self):
        return f"""game board: 
    {self.new_game_board}
    move was: {self.new_spot.to_friendly_format()}
    valid move: {self.is_move_valid}
    game ended: {self.game_ended}
    player won: {self.player_won}
"""

    def __repr__(self):
        return self._to_str()

    def __str__(self):
        return self._to_str()


class GameBoard:
    # note, 6, 8 size board are supported. Not supporting > 8 board
    def __init__(self, board_size=8):
        self.board_size = board_size

        self.reset()

    def reset(self):
        self.board = [[0.0] * self.board_size for _ in range(self.board_size)]

        self.game_ended = False
        self.winner = 0
        self.player_1_count = 0
        self.player_2_count = 0
        self.possible_moves_player_1 = self.get_valid_spots()
        self.possible_moves_player_2 = self.get_valid_spots()

    def update_status(self):
        self.possible_moves_player_1 = self.get_valid_spots()
        self.possible_moves_player_2 = self.get_valid_spots()
        self.player_1_count = self.count_player_pieces(PLAYER_1)
        self.player_2_count = self.count_player_pieces(PLAYER_2)
        valid_spot = self.get_valid_spots()
        if len(valid_spot) == 0:
            self.game_ended = True
        for r, c in product(range(self.board_size), range(self.board_size)):
            current_spot = Spot(r, c, self.board_size)
            current_player = self.get_spot_state(current_spot)
            if current_player != 0:
                # player 1, or 2
                won = self.did_player_win(current_player, current_spot)
                if won:
                    self.game_ended = True
                    self.winner = current_player
                    break

    def deepcopy(self):
        new_game = GameBoard(self.board_size)
        new_game.board = copy.deepcopy(self.board)
        new_game.game_ended = self.game_ended
        new_game.winner = self.winner
        new_game.player_1_count = self.player_1_count
        new_game.player_2_count = self.player_2_count
        new_game.possible_moves_player_1 = self.possible_moves_player_1
        new_game.possible_moves_player_2 = self.possible_moves_player_2

        return new_game

    def __eq__(self, other):
        return self.board == other.board

    def _row_to_string(self, the_row):
        def __to_view_string(cell):
            if cell == 0:
                return '.'
            elif cell == PLAYER_1:
                return 'O'
            elif cell == PLAYER_2:
                return 'X'

        return ' '.join(map(__to_view_string, the_row))

    def _to_str(self):
        header = '  ' + \
            (' '.join(
                map(lambda i: COL_MAP_TO_CHAR[i], range(self.board_size))))
        return_str = 'Game board:\n' + header + '\n'\
            + '\n'.join([(str(r+1) + ' ' + self._row_to_string(self.board[r]) + ' ' + str(r+1))
                         for r in range(self.board_size)]) \
            + '\n' + header

        return_str = return_str + '\nPlayer 1: {0}'.format(self.player_1_count)
        return_str = return_str + '\nPlayer 2: {0}'.format(self.player_2_count)
        return_str = return_str + '\nNext possible moves for Player 1: {0}'.format(
            list(map(lambda m: m.to_friendly_format(), self.possible_moves_player_1)))
        return_str = return_str + '\nNext possible moves for Player 2: {0}'.format(
            list(map(lambda m: m.to_friendly_format(), self.possible_moves_player_2)))
        return_str = return_str + '\nGame Ended: {0}'.format(self.game_ended)
        return return_str

    def __repr__(self):
        return self._to_str()

    def __str__(self):
        return self._to_str()

    def observe_board_1d(self):
        flat = list(itertools.chain.from_iterable(self.board))
        return flat

    def observe_board_2d(self):
        # returns board as 2d array
        return self.board

    def get_spot_state(self, spot):
        if spot.is_outside():
            return 0
        else:
            return self.board[spot.row][spot.col]

    def count_player_pieces(self, player_id):
        all_spots = []
        for r in self.board:
            all_spots.extend(r)

        return len(list(filter(lambda s: s == player_id, all_spots)))

    def get_next_player(self, current_player_id):
        next_player_id = -current_player_id
        return next_player_id

    # return new Game.
    def _flip(self, flipping_spots):
        new_game = self.deepcopy()
        for spot in flipping_spots:
            new_game.board[spot.row][spot.col] = - \
                new_game.board[spot.row][spot.col]

        return new_game

    # this function look to a direction of the new piece,
    # will return any spots that will need to be flipped.
    # return array of spots. [(row, col),(row, col),..]
    # when return [] array of spots, it means the play_step is not valid
    # direction_fn: one of Spot.step_???()
    def _line_length(self, player_id, spot, direction_fn):
        num_my_pieces = 0
        check_spot = direction_fn(spot)
        while ((not check_spot.is_outside())
               and self.get_spot_state(check_spot) == player_id):
            num_my_pieces += 1
            check_spot = direction_fn(check_spot)
        return num_my_pieces

    # This function evaluate a move, find the spots that will be flipped by this move.
    # return array of positions that will be flipped.
    # positions will be identified by (row, col) index starting from 0
    def did_player_win(self, player_id, spot):
        line_length = 0
        # eval left # eval right # eval up # eval down # eval up left # eval up right # eval down left # eval down right
        line_length = (self._line_length(player_id, spot, Spot.step_left) +
                       self._line_length(player_id, spot, Spot.step_right)) + 1
        if line_length >= 3:
            return True
        line_length = (self._line_length(player_id, spot, Spot.step_up) +
                       self._line_length(player_id, spot, Spot.step_down)) + 1
        if line_length >= 3:
            return True
        line_length = (self._line_length(player_id, spot, Spot.step_up_left) +
                       self._line_length(player_id, spot, Spot.step_down_right)) + 1
        if line_length >= 3:
            return True
        line_length = (self._line_length(player_id, spot, Spot.step_up_right) +
                       self._line_length(player_id, spot, Spot.step_down_left)) + 1
        if line_length >= 3:
            return True

        return False

    # look all the spots, if it's a valid move for the player, include it in the return value
    # returns array of Spot().
    def get_valid_spots(self):
        valid_spots = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                current_spot = Spot(r, c, self.board_size)
                # at least adjacent to another piece
                if self.get_spot_state(current_spot) == 0:
                    valid_spots.append(current_spot)
        return valid_spots

    # returns new game state, it's a new deepcopy GameBoard,
    # {game, flipped, valid_move: true/false}
    def play_a_piece(self, player_id: int, spot: Spot) -> ResultOfAMove:
        if self.get_spot_state(spot) != 0:
            # invalid move
            return ResultOfAMove(new_game_board=self,
                                 new_spot=spot,
                                 is_move_valid=False,
                                 game_ended=self.game_ended,
                                 won=False)
        # valid, check for win? is it going to cause at lease one piece to flip?
        new_game = self
        new_game.board[spot.row][spot.col] = player_id
        new_game.update_status()
        return ResultOfAMove(new_game_board=self,
                             new_spot=spot,
                             is_move_valid=True,
                             game_ended=self.game_ended,
                             won=(self.game_ended and self.winner == player_id))


if __name__ == '__main__':
    # random play
    game = GameBoard(board_size=3)
    current_player = PLAYER_1
    while not game.game_ended:
        spot = random.choice(game.get_valid_spots())
        result = game.play_a_piece(current_player, spot)
        current_player = game.get_next_player(current_player)
    print(result)
    print(game)
