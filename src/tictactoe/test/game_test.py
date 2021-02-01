import unittest

from tictactoe.GameBoard import GameBoard, PLAYER_1, PLAYER_2, Spot


class TestGame(unittest.TestCase):
    def test_observe_1d(self):
        the_game = GameBoard(board_size=3)
        flat_array = the_game.observe_board_1d()
        self.assertEqual(flat_array, [0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, ])
        the_game.board[1][2] = 1.0
        flat_array = the_game.observe_board_1d()
        self.assertEqual(flat_array, [0.0, 0.0, 0.0,
                                      0.0, 0.0, 1.0,
                                      0.0, 0.0, 0.0, ])

    def test_deepcopy(self):
        the_game = GameBoard(board_size=3)
        new_game = the_game.deepcopy()
        new_game.board[0][0] = PLAYER_1
        self.assertFalse(the_game == new_game)

    def test_win_1(self):
        the_game = GameBoard(board_size=3)
        the_game.board = [[1.0, 1.0, 1.0, ],
                          [0.0, 0.0, 1.0, ],
                          [0.0, 0.0, 0.0, ]]

        the_game.update_status()
        self.assertTrue(the_game.game_ended)
        self.assertEqual(the_game.winner, PLAYER_1)

    def test_win_2(self):
        the_game = GameBoard(board_size=3)
        the_game.board = [[1.0, 1.0, -1.0, ],
                          [0.0, 0.0, 1.0, ],
                          [-1.0, 0.0, 0.0, ]]

        result = the_game.play_a_piece(PLAYER_2, Spot(1,1))
        self.assertTrue(the_game.game_ended)
        self.assertEqual(the_game.winner, PLAYER_2)
        self.assertTrue(result.game_ended)
        self.assertTrue(result.player_won)

    def test_win_3(self):
        the_game = GameBoard(board_size=3)
        the_game.board = [[1.0, -1.0, 1.0, ],
                          [-1.0, 1.0, -1.0, ],
                          [0.0, 1.0, -1.0, ]]

        result = the_game.play_a_piece(PLAYER_1, Spot(2,0))
        self.assertTrue(the_game.game_ended)
        self.assertEqual(the_game.winner, PLAYER_1)
        self.assertTrue(result.game_ended)
        self.assertTrue(result.player_won)


if __name__ == '__main__':
    unittest.main()
