import unittest

from othello.game import GameBoard


class TestRot90(unittest.TestCase):
    def test_observe_1d(self):
        action_code = 0
        action_code_rot = GameBoard.GameMove.rot90_action_code(action_code, board_size=6)
        self.assertEqual(action_code_rot, 30)

        action_code = 5
        action_code_rot = GameBoard.GameMove.rot90_action_code(action_code, board_size=6)
        self.assertEqual(action_code_rot, 0)

        action_code = 6 # row 2, column 1
        action_code_rot = GameBoard.GameMove.rot90_action_code(action_code, board_size=6)
        self.assertEqual(action_code_rot, 31) # after rot90, row 6, column 2

        action_code = 11 # row 2, column 6
        action_code_rot = GameBoard.GameMove.rot90_action_code(action_code, board_size=6)
        self.assertEqual(action_code_rot, 1) # after rot90, row 1, column 2


if __name__ == '__main__':
    unittest.main()
