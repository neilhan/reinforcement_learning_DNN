from reversi.game import GameBoard


class GameWrapper:
    # Game play related state
    def __init__(self):
        self.game_board = GameBoard.GameBoard()
        self.current_player = GameBoard.PLAYER_1
        self.game_ended = False

    def execute_move(self, spot):
        move_result = self.game_board.get_new_board_for_a_move(self.current_player, spot)

        if not move_result.is_move_valid:
            raise ValueError('Invalid move.')

        self.game_board = move_result.new_game_board
        self.current_player = self.game_board.get_next_player(self.current_player)
        self.game_ended = self.game_board.game_ended

        return self

