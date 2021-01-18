import unittest

from reversi.game import GameBoard


class TestGame(unittest.TestCase):
    def test_deepcopy(self):
        the_game = GameBoard.GameBoard()
        new_game = the_game.deepcopy()
        new_game.board[0][0] = GameBoard.PLAYER_1
        self.assertFalse(the_game == new_game)

    def test_eval_left(self):
        the_game = GameBoard.GameBoard()
        # P1 place at (3, 5), going to flip (3,4)
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_1, GameBoard.Spot(3, 5), GameBoard.Spot.step_left)
        self.assertEqual(flipping_spots, [GameBoard.Spot(3, 4)])

        # P1 at (3,3), not valid. has existing piece
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_1, GameBoard.Spot(3, 3), GameBoard.Spot.step_left)
        self.assertEqual(flipping_spots, [])

        # P1 at (3,2), not valid. not going to cause flip
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_1, GameBoard.Spot(3, 2), GameBoard.Spot.step_left)
        self.assertEqual(flipping_spots, [])

        # P1 at (3,0), not valid. not going to cause flip
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_1, GameBoard.Spot(3, 0), GameBoard.Spot.step_left)
        self.assertEqual(flipping_spots, [])

        # opponent to the left edge, place (0, 2)
        # set board[0]: [GameBoard.PLAYER_2, GameBoard.PLAYER_2,0,0,0,0,0,0], place at (0,2), edge not going to cause flip
        the_game.board[0] = [GameBoard.PLAYER_2,
                             GameBoard.PLAYER_2, 0, 0, 0, 0, 0, 0]
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_1, GameBoard.Spot(0, 2), GameBoard.Spot.step_left)
        self.assertEqual(flipping_spots, [])

    def test_eval_right(self):
        the_game = GameBoard.GameBoard()
        # P2 place at (3, 2), going to flip (3, 3)
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_2, GameBoard.Spot(3, 2), GameBoard.Spot.step_right)
        self.assertEqual(flipping_spots, [GameBoard.Spot(3, 3)])

        # P2 at (3,3), not valid. has existing piece
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_2, GameBoard.Spot(3, 3), GameBoard.Spot.step_right)
        self.assertEqual(flipping_spots, [])

        # P1 at (3,5), not valid. No flip
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_2, GameBoard.Spot(3, 5), GameBoard.Spot.step_right)
        self.assertEqual(flipping_spots, [])

        # P1 at (3,0), not valid. No flip
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_2, GameBoard.Spot(3, 0), GameBoard.Spot.step_right)
        self.assertEqual(flipping_spots, [])

        # opponent to the left edge, place (0, 2)
        # board[0]: [P1, P1,0,0,0,0,0,0]
        the_game.board[0] = [GameBoard.PLAYER_1,
                             GameBoard.PLAYER_1, 0, 0, 0, 0, 0, 0]
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_2, GameBoard.Spot(0, 2), GameBoard.Spot.step_right)
        self.assertEqual(flipping_spots, [])

        # opponent to the left edge, place (0, 2)
        # board[0]: [0, P1, P1, P2, 0, 0, 0, 0]
        the_game.board[0] = [0, GameBoard.PLAYER_1, GameBoard.PLAYER_1, GameBoard.PLAYER_2,
                             0, 0, 0, 0]
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_2, GameBoard.Spot(0, 0), GameBoard.Spot.step_right)
        self.assertEqual(flipping_spots, [
                         GameBoard.Spot(0, 1), GameBoard.Spot(0, 2)])

    def test_eval(self):
        the_game = GameBoard.GameBoard()
        # P1 place at (5, 3), going to flip (4, 3)
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_1, GameBoard.Spot(5, 3), GameBoard.Spot.step_up)
        self.assertEqual(flipping_spots, [GameBoard.Spot(4, 3)])

        # P1 at (4, 3), not valid. has existing piece
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_1, GameBoard.Spot(4, 3), GameBoard.Spot.step_up)
        self.assertEqual(flipping_spots, [])

        # P1 at (5, 4), not valid. not going to cause flip
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_1, GameBoard.Spot(5, 4), GameBoard.Spot.step_up)
        self.assertEqual(flipping_spots, [])

        # P1 at (5, 0), not valid. not going to cause flip
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_1, GameBoard.Spot(5, 0), GameBoard.Spot.step_up)
        self.assertEqual(flipping_spots, [])

        # opponent to up edge, place (0, 2), not causing flip
        # set board[0]: [2,0,0,0,0,0,0,0], place at (1,0), edge not going to cause flip
        the_game.board[0] = [GameBoard.PLAYER_2, 0, 0, 0, 0, 0, 0, 0]
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_1, GameBoard.Spot(1, 0), GameBoard.Spot.step_up)
        self.assertEqual(flipping_spots, [])

    def test_eval_down(self):
        the_game = GameBoard.GameBoard()
        # P1 place at (2, 4), going to flip (3, 4)
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_1, GameBoard.Spot(2, 4), GameBoard.Spot.step_down)
        self.assertEqual(flipping_spots, [GameBoard.Spot(3, 4)])

        # P1 at (3, 4), not valid. has existing piece
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_1, GameBoard.Spot(3, 4), GameBoard.Spot.step_down)
        self.assertEqual(flipping_spots, [])

        # P1 at (3, 3), not valid. not going to cause flip
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_1, GameBoard.Spot(3, 3), GameBoard.Spot.step_down)
        self.assertEqual(flipping_spots, [])

        # P1 at (5, 0), not valid. not going to cause flip
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_1, GameBoard.Spot(5, 0), GameBoard.Spot.step_down)
        self.assertEqual(flipping_spots, [])

        # opponent to down edge, place (7, 0), not causing flip
        # set board[7]: [2,0,0,0,0,0,0,0], place at (6, 0), edge not going to cause flip
        the_game.board[7] = [GameBoard.PLAYER_2, 0, 0, 0, 0, 0, 0, 0]
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_1, GameBoard.Spot(6, 0), GameBoard.Spot.step_down)
        self.assertEqual(flipping_spots, [])

    def test_eval_up_left(self):
        the_game = GameBoard.GameBoard()
        the_game.board[4][5] = GameBoard.PLAYER_1
        # P2 place at (5, 6), going to flip (4, 5)
        flipping_spots = the_game._eval_step(
            GameBoard.PLAYER_2, GameBoard.Spot(5, 6), GameBoard.Spot.step_up_left)
        self.assertEqual(flipping_spots, [GameBoard.Spot(4, 5)])

    def test_play_a_piece(self):
        the_game = GameBoard.GameBoard()
        result = the_game.get_new_board_for_a_move(
            GameBoard.PLAYER_1, GameBoard.Spot(3, 5))
        self.assertTrue(result.is_move_valid)
        self.assertEqual(result.flipped_spots, [GameBoard.Spot(3, 4)])

    def test_get_valid_spots(self):
        the_game = GameBoard.GameBoard()
        valid_spots = the_game.get_valid_spots(GameBoard.PLAYER_1)
        self.assertEqual(valid_spots,
                         [GameBoard.Spot(2, 4), GameBoard.Spot(3, 5), GameBoard.Spot(4, 2), GameBoard.Spot(5, 3)])

        valid_spots = the_game.get_valid_spots(GameBoard.PLAYER_2)
        self.assertEqual(valid_spots,
                         [GameBoard.Spot(2, 3), GameBoard.Spot(3, 2), GameBoard.Spot(4, 5), GameBoard.Spot(5, 4)])
        print(valid_spots)


if __name__ == '__main__':
    unittest.main()
