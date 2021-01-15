from reversi import GameBoard
from .players import RandomPlayer


class Game:
    # Game play related state
    def __init__(self):
        self.game_board = GameBoard.GameBoard()
        self.current_player = GameBoard.PLAYER_1
        self.game_ended = False

    def execute_move(self, spot):
        new_game_state = self.game_board.play_a_piece(self.current_player, spot)

        if not new_game_state['is_move_valid']:
            raise ValueError('Invalid move.')

        self.game_board = new_game_state['game_board']
        self.current_player = self.game_board.get_next_player(self.current_player)
        self.game_ended = self.game_board.game_ended

        return self


def play_human():
    the_game = Game()
    print('Input your move by using <row><collum>. for example: 1a. To quite: ctl-c, or enter q')

    while True:
        print(the_game.game_board)
        if the_game.current_player == GameBoard.PLAYER_1:
            user_input = input('Player O: ')
        else:
            user_input = input('Player X: ')

        if user_input == 'q' or user_input == 'Q':
            break

        try:
            spot = GameBoard.Spot.from_friendly_format(user_input)

            # check spot and next player # {game, flipped, valid_move: true/false}
            the_game = the_game.execute_move(spot)
            # is game done?
            if the_game.game_ended:
                print('Game ended')
                print('Player O:', the_game.player_1_count)
                print('Player X:', the_game.player_2_count)
                if the_game.winner == GameBoard.PLAYER_1:
                    print('Player O won')
                if the_game.winner == GameBoard.PLAYER_2:
                    print('Player X won')
                if the_game.winner == 0:
                    print('Tie')
                break

        except Exception as e:
            print(e)
            print('Invalid move. ctl-c or q to quite. Or try again.')


def play_random():
    the_game = Game()

    player_1 = RandomPlayer.RandomPlayer(GameBoard.PLAYER_1)
    player_2 = RandomPlayer.RandomPlayer(GameBoard.PLAYER_2)

    print('Game starts.')

    while True:
        print(the_game.game_board)
        if the_game.current_player == GameBoard.PLAYER_1:
            next_move = player_1.pick_next_move(the_game.game_board)
            print('Player O:', next_move.to_friendly_format())
        else:
            next_move = player_2.pick_next_move(the_game.game_board)
            print('Player X:', next_move.to_friendly_format())
        print('---------------------------------')

        try:
            # check spot and next player # {game, flipped, valid_move: true/false}
            the_game = the_game.execute_move(next_move)
        except Exception as e:
            print(e)
            print('Invalid move. ctl-c or q to quite. Or try again.')

        # is game done?
        if the_game.game_ended:
            print('*** Game ended ***')
            print(the_game.game_board)
            print('*** Game ended ***')
            if the_game.game_board.winner == GameBoard.PLAYER_1:
                print('Player O won')
            if the_game.game_board.winner == GameBoard.PLAYER_2:
                print('Player X won')
            if the_game.game_board.winner == 0:
                print('Tie')
            break


if __name__ == '__main__':
    # play_human()
    play_random()
