from othello.game import GameBoard


class GameWrapper:
    # Game play related state
    def __init__(self, id: int):
        self.id = id
        self.game_board = GameBoard.GameBoard()
        self.current_player = GameBoard.PLAYER_1
        self.game_ended = False

    def execute_move(self, spot):
        move_result = self.game_board.make_a_move(self.current_player, GameBoard.GameMove(spot))

        if not move_result.is_move_valid:
            raise ValueError('Invalid move.')

        self.game_board = move_result.new_game_board
        self.current_player = self.game_board.get_next_player(self.current_player)
        self.game_ended = self.game_board.game_ended

        return self

