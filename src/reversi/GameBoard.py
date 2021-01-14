import copy

PLAYER_1 = 1
PLAYER_2 = 2


class Spot:
    def __init__(self, row, col):
        if row < 8 and row >= 0 and col < 8 and col >= 0:
            self.row = row
            self.col = col
        else:
            raise ValueError('row and col needs to be between [0, 8)')

    def is_outside(self) -> bool:
        return self.row < 0 or self.row >= 8 or self.col < 0 or self.col >= 8

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __repr__(self):
        return "Spot(%i, %i)" % (self.row, self.col)

    def __str__(self):
        return "Spot(%i, %i)" % (self.row, self.col)

    def step_left(self):
        new_spot = Spot(self.row, self.col)
        new_spot.col = new_spot.col - 1
        return new_spot

    def step_right(self):
        new_spot = Spot(self.row, self.col)
        new_spot.col = new_spot.col + 1
        return new_spot

    def step_up(self):
        new_spot = Spot(self.row, self.col)
        new_spot.row = new_spot.row - 1
        return new_spot

    def step_down(self):
        new_spot = Spot(self.row, self.col)
        new_spot.row = new_spot.row + 1
        return new_spot

    def step_up_left(self):
        new_spot = Spot(self.row, self.col)
        new_spot.row = new_spot.row - 1
        new_spot.col = new_spot.col - 1
        return new_spot

    def step_up_right(self):
        new_spot = Spot(self.row, self.col)
        new_spot.row = new_spot.row - 1
        new_spot.col = new_spot.col + 1
        return new_spot

    def step_down_left(self):
        new_spot = Spot(self.row, self.col)
        new_spot.row = new_spot.row + 1
        new_spot.col = new_spot.col - 1
        return new_spot

    def step_down_right(self):
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

    def get_spot_state(self, spot):
        if spot.is_outside():
            return 0
        else:
            return self.board[spot.row][spot.col]

    def deepcopy(self):
        new_game = GameBoard()
        new_game.board = copy.deepcopy(self.board)
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

    def __repr__(self):
        return '''Game board:
                 a b c d e f g h
               1 {0}
               2 {1}
               3 {2}
               4 {3}
               5 {4}
               6 {5}
               7 {6}
               8 {7}'''.format(*([self._row_to_string(r) for r in self.board]))

    def __str__(self):
        return '''Game board:
                 a b c d e f g h
               1 {0}
               2 {1}
               3 {2}
               4 {3}
               5 {4}
               6 {5}
               7 {6}
               8 {7}'''.format(*([self._row_to_string(r) for r in self.board]))

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
            current_spot = spot
            return list(map(lambda x: direction_fn(current_spot), range(0, steps - 1, 1)))
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

    # return new Game.
    def _flip(self, player_id, flipping_spots):
        new_game = self.deepcopy()
        for spot in flipping_spots:
            print('1 spot:', spot)
            new_game.board[spot.row][spot.col] = player_id

        return new_game

    # returns new game state
    # {game, valid_move: true/false}
    def play_a_piece(self, player_id, row, col):
        # is it going to cause at lease one piece to flip?
        flipping_spots = self.get_flipping_spots(player_id, Spot(row, col))
        if len(flipping_spots) > 0:
            # valid move, excute
            new_game = self._flip(player_id, flipping_spots + [Spot(row, col)])
            return {
                'game': new_game,
                'flipped': flipping_spots,
                'is_move_valid': True,
            }
        else:
            # invalid move
            return {
                'game': self,
                'flipped': flipping_spots,
                'is_move_valid': False,
            }

    # look all the spots, if it's a valid move for the player, include it in the return value
    # returns array of Spot().
    def get_valid_spots(self, player_id):
        valid_spots = []
        for r in range(0, 8, 1):
            for c in range(0, 8, 1):
                current_spot = Spot(r, c)
                # at least adjacent to another piece
                if (self.get_spot_state(current_spot.step_up()) > 0
                        or self.get_spot_state(current_spot.step_down()) > 0
                        or self.get_spot_state(current_spot.step_left()) > 0
                        or self.get_spot_state(current_spot.step_right()) > 0
                        or self.get_spot_state(current_spot.step_up_left()) > 0
                        or self.get_spot_state(current_spot.step_up_right()) > 0
                        or self.get_spot_state(current_spot.step_down_left()) > 0
                        or self.get_spot_state(current_spot.step_down_right()) > 0
                ):
                    valid_spots.append(current_spot)
        valid_spots = list(filter(lambda s: len(self.get_flipping_spots(player_id,s))>0, valid_spots))
        return valid_spots


# player_id: 1 or 2
def get_opponent_player_id(player_id):
    if player_id == 1:
        return 2
    else:
        return 1
