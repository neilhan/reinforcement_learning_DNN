from __future__ import annotations
import random

import itertools
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

    def to_action_code(self):
        return self.row * self.board_size + self.col

    def from_friendly_format(self, friendly_format: str) -> Spot:
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

    def step_left(self) -> Spot:
        new_spot = Spot(self.row, self.col, self.board_size)
        new_spot.col = new_spot.col - 1
        return new_spot

    def step_right(self) -> Spot:
        new_spot = Spot(self.row, self.col, self.board_size)
        new_spot.col = new_spot.col + 1
        return new_spot

    def step_up(self) -> Spot:
        new_spot = Spot(self.row, self.col, self.board_size)
        new_spot.row = new_spot.row - 1
        return new_spot

    def step_down(self) -> Spot:
        new_spot = Spot(self.row, self.col, self.board_size)
        new_spot.row = new_spot.row + 1
        return new_spot

    def step_up_left(self) -> Spot:
        new_spot = Spot(self.row, self.col, self.board_size)
        new_spot.row = new_spot.row - 1
        new_spot.col = new_spot.col - 1
        return new_spot

    def step_up_right(self) -> Spot:
        new_spot = Spot(self.row, self.col, self.board_size)
        new_spot.row = new_spot.row - 1
        new_spot.col = new_spot.col + 1
        return new_spot

    def step_down_left(self) -> Spot:
        new_spot = Spot(self.row, self.col, self.board_size)
        new_spot.row = new_spot.row + 1
        new_spot.col = new_spot.col - 1
        return new_spot

    def step_down_right(self) -> Spot:
        new_spot = Spot(self.row, self.col, self.board_size)
        new_spot.row = new_spot.row + 1
        new_spot.col = new_spot.col + 1
        return new_spot


class GameMove:
    def __init__(self, spot: Spot = None, pass_turn: bool = False):
        self.spot = spot
        self.pass_turn = pass_turn

    @staticmethod
    def from_action_code(action_code, board_size=8):
        if action_code >= board_size*board_size:
            return GameMove(pass_turn=True)
        else:
            return GameMove(spot=Spot.from_action_code(action_code, board_size))
    def to_friendly_format(self):
        if self.pass_turn:
            return 'PASS'
        else:
            return self.spot.to_friendly_format()


class ResultOfAMove:
    def __init__(self,
                 new_game_board: GameBoard,
                 game_ended: bool,
                 the_move: GameMove,
                 is_move_valid: bool,
                 flipped_spots: list[Spot],
                 ):
        self.new_game_board = new_game_board
        self.game_ended = game_ended
        self.the_move = the_move
        self.is_move_valid = is_move_valid
        self.flipped_spots = flipped_spots


class GameBoard:
    # note, 6, 8 size board are supported. Not supporting > 8 board
    def __init__(self, board_size=8, random_start=False):
        self.board_size = board_size
        if board_size % 2 > 0:
            raise ValueError('board_size needs to be > 4, and even number.')
        self.reset(random_reset=random_start)

    def reset(self, random_reset):
        self.board = [[0.0]*self.board_size for _ in range(self.board_size)]
        board_size = self.board_size
        self.current_player = PLAYER_1
        self.game_ended = False
        self.winner = 0
        self.player_1_count = 0
        self.player_2_count = 0
        self.possible_moves_player_1 = 0
        self.possible_moves_player_2 = 0

        # this is to help seed the agent. Otherwise agen will overfit
        if random_reset:
            for r in range(self.board_size):
                for c in range(self.board_size):
                    self.board[r][c] = random.choice(
                        [PLAYER_1, 0, 0, PLAYER_2])
            # self.current_player = random.choice([PLAYER_1, PLAYER_2])

        self.board[int(board_size/2-1)][int(board_size/2-1)] = PLAYER_1
        self.board[int(board_size/2)][int(board_size/2)] = PLAYER_1
        self.board[int(board_size/2-1)][int(board_size/2)] = PLAYER_2
        self.board[int(board_size/2)][int(board_size/2-1)] = PLAYER_2

        self.update_status()

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
        new_game = GameBoard(self.board_size)
        new_game.board = copy.deepcopy(self.board)
        new_game.current_player = self.current_player
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

    def get_opponent_of(self, player_id):
        opponent_player_id = player_id * -1
        return opponent_player_id

    def get_next_player(self, player_id):
        """Return who is the next valid player. If opponent has no valid move, the same player continue."""
        opponent_id = self.get_opponent_of(player_id)

        valid_moves = self.get_valid_spots(opponent_id)
        if len(valid_moves) > 0:
            return opponent_id
        else:
            return player_id  # opponent has no moves, so current_player continue

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
    def _eval_step(self, player_id, spot, direction_fn):
        opponent_id = get_opponent_player_id(player_id)
        check_spot = direction_fn(spot)
        while ((not check_spot.is_outside())
               and self.get_spot_state(check_spot) == opponent_id):
            check_spot = direction_fn(check_spot)

        # if stopped not at board edge, and a piece of my player, return all the spots in between
        if ((not check_spot.is_outside())
                and self.get_spot_state(check_spot) == player_id):
            steps = max(abs(spot.row - check_spot.row),
                        abs(spot.col - check_spot.col))
            flips = []
            current_spot = spot
            for _ in range(0, steps - 1, 1):
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
        for r in range(self.board_size):
            for c in range(self.board_size):
                current_spot = Spot(r, c, self.board_size)
                # at least adjacent to another piece
                if (self.get_spot_state(current_spot) == 0
                    and (self.get_spot_state(current_spot.step_up()) != 0
                         or self.get_spot_state(current_spot.step_down()) != 0
                         or self.get_spot_state(current_spot.step_left()) != 0
                         or self.get_spot_state(current_spot.step_right()) != 0
                         or self.get_spot_state(current_spot.step_up_left()) != 0
                         or self.get_spot_state(current_spot.step_up_right()) != 0
                         or self.get_spot_state(current_spot.step_down_left()) != 0
                         or self.get_spot_state(current_spot.step_down_right()) != 0
                         )):
                    valid_spots.append(current_spot)
        valid_spots = \
            list(filter(lambda s: len(self.get_flipping_spots(player_id, s)) > 0,
                        valid_spots))
        return valid_spots

    # returns new game state, it's a new deepcopy GameBoard,
    # {game, flipped, valid_move: true/false}
    def make_a_move(self, player_id: int, move: GameMove) -> ResultOfAMove:
        if move.pass_turn:
            # player pass turn. May not be valid if there are moves for him
            valid = True
            if player_id == PLAYER_1 and len(self.possible_moves_player_1) > 0:
                valid = False
            if player_id == PLAYER_2 and len(self.possible_moves_player_2) > 0:
                valid = False
            if valid:  # valid to pass
                self.current_player = self.get_opponent_of(player_id)
                self.update_status()
                return ResultOfAMove(new_game_board=self,
                                     game_ended=self.game_ended,
                                     the_move=move,
                                     is_move_valid=True,
                                     flipped_spots=[])

            else:  # not valid to do pass
                return ResultOfAMove(new_game_board=self,
                                     game_ended=self.game_ended,
                                     the_move=move,
                                     is_move_valid=False,
                                     flipped_spots=[])

        else:  # Placing a piece
            spot = move.spot
            if self.get_spot_state(spot) != 0:
                # invalid move. No need to do self.update_status()
                return ResultOfAMove(new_game_board=self,
                                     game_ended=self.game_ended,
                                     the_move=move,
                                     is_move_valid=False,
                                     flipped_spots=[])

            # is it going to cause at lease one piece to flip?
            flipping_spots = self.get_flipping_spots(player_id, spot)
            if len(flipping_spots) > 0:  # valid move, excute
                new_game = self._flip(flipping_spots)
                new_game.board[spot.row][spot.col] = player_id

                # update board status
                new_game.update_status()

                return ResultOfAMove(new_game_board=new_game,
                                     game_ended=new_game.game_ended,
                                     the_move=move,
                                     is_move_valid=True,
                                     flipped_spots=flipping_spots)
            else:  # no flipping pieces, invalid move
                return ResultOfAMove(new_game_board=self,
                                     game_ended=self.game_ended,
                                     the_move=move,
                                     is_move_valid=False,
                                     flipped_spots=[])
