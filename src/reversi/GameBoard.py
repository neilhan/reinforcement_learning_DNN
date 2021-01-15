from __future__ import annotations

import copy

PLAYER_1 = 1
PLAYER_2 = 2

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


class Spot:
    def __init__(self, row, col):
        if row < 8 and row >= 0 and col < 8 and col >= 0:
            self.row = row
            self.col = col
        else:
            raise ValueError('row and col needs to be between [0, 8)')

    # return Spot
    @staticmethod
    def from_friendly_format(friendly_format: str) -> Spot:
        if len(friendly_format) < 2:
            raise ValueError('At lease 2 characters. 1a, 1b, or 7h')
        row = int(friendly_format[:1]) - 1
        col = COL_MAP[friendly_format[1:2]]
        return Spot(row, col)

    # return str, for example: 1A, 6B
    def to_friendly_format(self):
        return '{0}{1}'.format(self.row + 1, COL_MAP_TO_CHAR[self.col])

    def is_outside(self) -> bool:
        return self.row < 0 or self.row >= 8 or self.col < 0 or self.col >= 8

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __repr__(self):
        return "Spot(%i, %i)" % (self.row, self.col)

    def __str__(self):
        return "Spot(%i, %i)" % (self.row, self.col)

    def step_left(self) -> Spot:
        new_spot = Spot(self.row, self.col)
        new_spot.col = new_spot.col - 1
        return new_spot

    def step_right(self) -> Spot:
        new_spot = Spot(self.row, self.col)
        new_spot.col = new_spot.col + 1
        return new_spot

    def step_up(self) -> Spot:
        new_spot = Spot(self.row, self.col)
        new_spot.row = new_spot.row - 1
        return new_spot

    def step_down(self) -> Spot:
        new_spot = Spot(self.row, self.col)
        new_spot.row = new_spot.row + 1
        return new_spot

    def step_up_left(self) -> Spot:
        new_spot = Spot(self.row, self.col)
        new_spot.row = new_spot.row - 1
        new_spot.col = new_spot.col - 1
        return new_spot

    def step_up_right(self) -> Spot:
        new_spot = Spot(self.row, self.col)
        new_spot.row = new_spot.row - 1
        new_spot.col = new_spot.col + 1
        return new_spot

    def step_down_left(self) -> Spot:
        new_spot = Spot(self.row, self.col)
        new_spot.row = new_spot.row + 1
        new_spot.col = new_spot.col - 1
        return new_spot

    def step_down_right(self) -> Spot:
        new_spot = Spot(self.row, self.col)
        new_spot.row = new_spot.row + 1
        new_spot.col = new_spot.col + 1
        return new_spot


class GameBoard:
    def __init__(self):
        self.board = [[0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 2, 0, 0, 0],
                      [0, 0, 0, 2, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0]]
        self.game_ended = False
        self.winner = 0
        self.player_1_count = 2
        self.player_2_count = 2
        self.possible_moves_player_1 = self.get_valid_spots(PLAYER_1)
        self.possible_moves_player_2 = self.get_valid_spots(PLAYER_2)

    def update_status(self):
        self.possible_moves_player_1 = self.get_valid_spots(PLAYER_1)
        self.possible_moves_player_2 = self.get_valid_spots(PLAYER_2)
        self.player_1_count = self.count_player_pieces(PLAYER_1)
        self.player_2_count = self.count_player_pieces(PLAYER_2)
        if len(self.possible_moves_player_1) == 0 and len(self.possible_moves_player_2) == 0:
            self.game_ended = True
        if self.player_1_count > self.player_2_count:
            self.winner = PLAYER_1
        if self.player_2_count > self.player_1_count:
            self.winner = PLAYER_2
        if self.player_2_count == self.player_1_count:
            self.winner = 0

    def deepcopy(self):
        new_game = GameBoard()
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
            elif cell == 1:
                return 'O'
            elif cell == 2:
                return 'X'

        return ' '.join(map(__to_view_string, the_row))

    def _to_str(self):
        return_str = '''Game board:
                 a b c d e f g h
               1 {0} 1
               2 {1} 2
               3 {2} 3
               4 {3} 4
               5 {4} 5
               6 {5} 6
               7 {6} 7
               8 {7} 8
                 a b c d e f g h
               '''.format(*([self._row_to_string(r) for r in self.board]))
        return_str = return_str + '\nPlayer 1: {0}'.format(self.player_1_count)
        return_str = return_str + '\nPlayer 2: {0}'.format(self.player_2_count)
        return_str = return_str + '\nNext possible moves for Player 1: {0}'.format(
            list(map(lambda m: Spot.to_friendly_format(m), self.possible_moves_player_1)))
        return_str = return_str + '\nNext possible moves for Player 2: {0}'.format(
            list(map(lambda m: Spot.to_friendly_format(m), self.possible_moves_player_2)))
        return_str = return_str + '\nGame Ended: {0}'.format(self.game_ended)
        return return_str

    def __repr__(self):
        return self._to_str()

    def __str__(self):
        return self._to_str()

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

    def get_next_player(self, current_player):
        next_player = 1
        if current_player == 1:
            next_player = 2

        valid_moves = self.get_valid_spots(next_player)
        if len(valid_moves) > 0:
            return next_player
        else:
            return current_player  # opponent has no moves, so current_player continue

    # return new Game.
    def _flip(self, player_id, flipping_spots):
        new_game = self.deepcopy()
        for spot in flipping_spots:
            new_game.board[spot.row][spot.col] = player_id

        return new_game

    # this function look to a direction of the new piece,
    # will return any spots that will need to be flipped.
    # return array of spots. [(row, col),(row, col),..]
    # when return [] array of spots, it means the play_step is not valid
    # direction_fn: one of Spot.step_???()
    def _eval_step(self, player_id, spot, direction_fn):
        opponent_id = get_opponent_player_id(player_id)
        check_spot = direction_fn(spot)
        while ((not check_spot.is_outside())
               and self.get_spot_state(check_spot) == opponent_id):
            check_spot = direction_fn(check_spot)

        # if stopped not at board edge, and a piece of my player, return all the spots in between
        if (not check_spot.is_outside()) and self.get_spot_state(check_spot) == player_id:
            steps = max(abs(spot.row - check_spot.row), abs(spot.col - check_spot.col))
            flips = []
            current_spot = spot
            for i in range(0, steps - 1, 1):
                current_spot = direction_fn(current_spot)
                flips.append(current_spot)
            return flips
        else:
            return []

    # This function evaluate a move, find the spots that will be flipped by this move.
    # return array of positions that will be flipped.
    # positions will be identified by (row, col) index starting from 0
    def get_flipping_spots(self, player_id, spot):
        flipping_spots = []
        # eval left # eval right # eval up # eval down # eval up left # eval up right # eval down left # eval down right
        flipping_spots = (self._eval_step(player_id, spot, Spot.step_left) +
                          self._eval_step(player_id, spot, Spot.step_right) +
                          self._eval_step(player_id, spot, Spot.step_up) +
                          self._eval_step(player_id, spot, Spot.step_down) +
                          self._eval_step(player_id, spot, Spot.step_up_left) +
                          self._eval_step(player_id, spot, Spot.step_up_right) +
                          self._eval_step(player_id, spot, Spot.step_down_left) +
                          self._eval_step(player_id, spot, Spot.step_down_right))

        return flipping_spots

    # look all the spots, if it's a valid move for the player, include it in the return value
    # returns array of Spot().
    def get_valid_spots(self, player_id):
        valid_spots = []
        for r in range(0, 8, 1):
            for c in range(0, 8, 1):
                current_spot = Spot(r, c)
                # at least adjacent to another piece
                if self.get_spot_state (current_spot) == 0 and (self.get_spot_state(current_spot.step_up()) > 0
                        or self.get_spot_state(current_spot.step_down()) > 0
                        or self.get_spot_state(current_spot.step_left()) > 0
                        or self.get_spot_state(current_spot.step_right()) > 0
                        or self.get_spot_state(current_spot.step_up_left()) > 0
                        or self.get_spot_state(current_spot.step_up_right()) > 0
                        or self.get_spot_state(current_spot.step_down_left()) > 0
                        or self.get_spot_state(current_spot.step_down_right()) > 0
                ):
                    valid_spots.append(current_spot)
        valid_spots = list(filter(lambda s: len(self.get_flipping_spots(player_id, s)) > 0, valid_spots))
        return valid_spots

    # returns new game state, it's a new deepcopy GameBoard,
    # {game, flipped, valid_move: true/false}
    def get_new_board_for_a_move(self, player_id: int, spot: Spot) -> GameBoard:
        if self.get_spot_state(spot) > 0:
            # invalid move
            return {
                'game_board': self,
                'flipped': [],
                'is_move_valid': False,
            }

        # is it going to cause at lease one piece to flip?
        flipping_spots = self.get_flipping_spots(player_id, spot)
        if len(flipping_spots) > 0:
            # valid move, excute
            new_game = self._flip(player_id, flipping_spots + [spot, ])
            # update board status
            new_game.update_status()

            return {
                'game_board': new_game,
                'flipped': flipping_spots,
                'is_move_valid': True,
            }
        else:
            return {
                'game_board': self,
                'flipped': [],
                'is_move_valid': False,
            }


# player_id: 1 or 2
def get_opponent_player_id(player_id):
    if player_id == 1:
        return 2
    else:
        return 1
