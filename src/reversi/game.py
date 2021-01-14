
PLAYER_1 = 1
PLAYER_2 = 2

class Spot:
    def __init__(self, row, col):
        if row <8 and row >=0 and col<8 and col >=0:
            self.row = row
            self.col = col
        else:
            raise ValueError('row and col needs to be between [0, 8)')
    def __eq__(self, other):
        return self.row == other.row and self.col == other.col
    
    def __repr__(self):
        return "Spot(%i, %i)" % (self.row, self.col)

    def __str__(self):
        return "Spot(%i, %i)" % (self.row, self.col)


class Game:
    def __init__(self):
        self.board = [  [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,1,2,0,0,0],
                        [0,0,0,2,1,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0] ] 

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

    # this function look to the left of the new piece, will return any spots 
    # that will need to be flipped
    # return array of spots. [(row, col),(row, col),..]
    def _eval_step_left(self, player_id, spot):
        col = spot.col 
        row = spot.row
        the_row = self.board[row]
        check_idx = col -1
        opponent_id = get_opponent_player_id(player_id)
        while (check_idx >= 0 and the_row[check_idx] == opponent_id):
            check_idx = check_idx - 1
        
        # if stopped not at board edge, and a piece of my player, return all the spots in between
        if the_row[check_idx] == player_id:
            return [Spot(row, i) for i in range(check_idx+1, col, 1)] 
        else:
            return []
        
    # This function evaluate a move, find the spots that will be flipped by this move.
    # return array of positions that will be flipped.
    # positions will be identified by (row, col) index starting from 0
    def eval_step(self, player_id, row, col):
        flipping_spots = []
        # is valid? spot open
        if self.board[row][col] == 0: 
            # eval left
            # eval right
            # eval up
            # eval down
            # eval up left
            # eval up right
            # eval down left
            # eval down right
            pass

        return flipping_spots

    # returns new game state
    def play_a_piece(game, player_id, row, col):
        # is it going to cause at lease one piece to flip?
        pass

# player_id: 1 or 2
def get_opponent_player_id(player_id):
    if player_id == 1: 
        return 2
    else:
        return 1

