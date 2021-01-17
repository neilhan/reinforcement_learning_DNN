from reversi import GameBoard
from reversi.GameAdapter import Game 
from reversi.players import RandomPlayer


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
