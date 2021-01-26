import unittest

import numpy as np

from reversi.game import GameBoard
from reversi.players.a2c_player_1.GameWrapper import GameWrapper


class TestGameWrapper(unittest.TestCase):
    def test_reset(self):
        env = GameWrapper(1)
        observation = env.reset()
        np.testing.assert_array_equal(
            observation,
            np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def test_spot_to_action(self):
        self.assertEqual(GameWrapper.convert_spot_to_action(GameBoard.Spot(0, 0)), 0)
        self.assertEqual(GameWrapper.convert_spot_to_action(GameBoard.Spot(1, 0)), 8)
        self.assertEqual(GameWrapper.convert_spot_to_action(GameBoard.Spot(1, 2)), 10)

    def test_action_to_spot(self):
        self.assertEqual(GameWrapper.convert_action_to_spot(0), GameBoard.Spot(0, 0))
        self.assertEqual(GameWrapper.convert_action_to_spot(8), GameBoard.Spot(1, 0))
        self.assertEqual(GameWrapper.convert_action_to_spot(10), GameBoard.Spot(1, 2))

        with self.assertRaises(Exception) as context:
            GameWrapper.convert_action_to_spot(64)

        with self.assertRaises(Exception) as context:
            GameWrapper.convert_action_to_spot(-1)

    def test_execute_move_pass(self):
        game = GameWrapper(1)
        old_observation = game.reset()

        env_observation, reward, done, game, is_valid = game.execute_move(game.PASS_TURN_ACTION)
        self.assertTrue(is_valid)
        np.testing.assert_array_equal(env_observation, old_observation*-1)
        self.assertEqual(reward, -0.01)
        self.assertFalse(done)

    def test_execute_move_illegal(self):
        game = GameWrapper(1)
        old_player = game.current_player
        # place at the corner. Not a valid move.
        env_observation, reward, done, game, is_valid = game.execute_move(0)
        # expecting: player not switched, same board observation, reward -0.01
        self.assertFalse(is_valid)
        self.assertEqual(old_player, game.current_player)
        self.assertEqual(reward, -0.01)
        self.assertFalse(done)
        np.testing.assert_array_equal(
            env_observation,
            np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def test_execute_move_legal(self):
        # current player is p1
        game = GameWrapper(1)

        # place a piece, to flip (3,5)
        env_observation, reward, done, game, is_valid = game.execute_move(8*3+5)
        # expection:
        #   player switched,
        #   not done game,
        #   new board observation,
        #   reward 0.01, +1 new pieces
        self.assertTrue(is_valid)
        self.assertEqual(game.current_player, GameBoard.PLAYER_2)
        self.assertFalse(done)
        self.assertEqual(reward, 0.01)
        np.testing.assert_array_equal(
            env_observation,
            np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def test_execute_move_legal_p2(self):
        # current player is p2
        game_env = GameWrapper(1)
        game_env.current_player = GameBoard.PLAYER_2

        # place a piece, to flip (4,5)
        env_observation, reward, done, game_env, is_valid = game_env.execute_move(8*4+5)
        # expection:
        #   player switched to p1,
        #   not done game,
        #   new board observation,
        #   reward 0.01, +1 new pieces
        self.assertTrue(is_valid)
        self.assertEqual(game_env.current_player, GameBoard.PLAYER_1)
        self.assertFalse(done)
        self.assertEqual(reward, 0.01)
        np.testing.assert_array_equal(
            env_observation,
            np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def test_execute_move_win(self):
        game_env = GameWrapper(1)
        game_env.game_board.board = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
                                     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
                                     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
                                     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
                                     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
                                     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
                                     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
                                     [1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]]
        # place at the last spot to win
        env_observation, reward, done, game_env, is_valid = game_env.execute_move(63)
        # expection:
        #   player switched to p1,
        #   game is done
        #   new board observation,
        #   reward 0.01, +1 new pieces
        self.assertTrue(is_valid)
        self.assertEqual(game_env.current_player, GameBoard.PLAYER_1)
        self.assertTrue(done)
        self.assertEqual(reward, 64)
        np.testing.assert_array_equal(
            env_observation,
            np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ]))

    def test_execute_move_tie(self):
        game_env = GameWrapper(1)
        game_env.current_player = GameBoard.PLAYER_1
        game_env.game_board.board = \
            [[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, ],
             [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, ],
             [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, ],
             [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, ],
             [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
             [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
             [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
             [1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]]
        # place at the last spot to win
        env_observation, reward, done, game_env, is_valid = game_env.execute_move(63)
        # expection:
        #   player switched to p1,
        #   game is done
        #   new board observation,
        #   reward 0.01, +1 new pieces
        self.assertTrue(is_valid)
        self.assertEqual(game_env.current_player, GameBoard.PLAYER_1)
        self.assertTrue(done)
        self.assertEqual(reward, 0)
        np.testing.assert_array_equal(
            env_observation,
            np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ]))

    def test_execute_move_win_p2(self):
        game_env = GameWrapper(1)
        game_env.current_player = GameBoard.PLAYER_2
        game_env.game_board.board = \
            list(map(lambda r: list(map(lambda i: i * -1.0, r)),
                     [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
                      [1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]]))
        # place at the last spot to win
        env_observation, reward, done, game_env, is_valid = game_env.execute_move(63)
        # expection:
        #   player switched to p1,
        #   game is done
        #   new board observation,
        #   reward 0.01, +1 new pieces
        self.assertTrue(is_valid)
        self.assertEqual(game_env.current_player, GameBoard.PLAYER_2)
        self.assertTrue(done)
        self.assertEqual(reward, 64)
        np.testing.assert_array_equal(
            env_observation,
            np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ]))

    def test_execute_move_tie_p2(self):
        game_env = GameWrapper(1)
        game_env.current_player = GameBoard.PLAYER_2
        game_env.game_board.board = \
            list(map(lambda r: list(map(lambda i: i * -1.0, r)),
                     [[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, ],
                      [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, ],
                      [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, ],
                      [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, ],
                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ],
                      [1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]]))
        # place at the last spot to win
        env_observation, reward, done, game_env, is_valid = game_env.execute_move(63)
        # expection:
        #   player switched to p1,
        #   game is done
        #   new board observation,
        #   reward 0.01, +1 new pieces
        self.assertTrue(is_valid)
        self.assertEqual(game_env.current_player, GameBoard.PLAYER_2)
        self.assertTrue(done)
        self.assertEqual(reward, 0)
        np.testing.assert_array_equal(
            env_observation,
            np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ]))
