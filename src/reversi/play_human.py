from reversi import GameBoard
from reversi.GameAdapter import Game 


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
                print('Player O:', the_game.game_board.player_1_count)
                print('Player X:', the_game.game_board.player_2_count)
                if the_game.game_board.winner == GameBoard.PLAYER_1:
                    print('Player O won')
                if the_game.game_board.winner == GameBoard.PLAYER_2:
                    print('Player X won')
                if the_game.game_board.winner == 0:
                    print('Tie')
                break

        except Exception as e:
            print(e)
            print('Invalid move. ctl-c or q to quite. Or try again.')


if __name__ == '__main__':
    play_human()
