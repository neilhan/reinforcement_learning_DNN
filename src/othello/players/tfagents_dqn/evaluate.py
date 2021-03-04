# c51 - different from DQN, C51 is based on DQN. C51 predicts
# a histogram model for the probability distribution of the Q-Value

import logging
import sys
import random
import traceback
import time
import argparse

import numpy as np
import tensorflow as tf
from tf_agents.trajectories import time_step as ts
from othello.service import tfagent_client as agent_client

from othello.game.GameBoard import GameBoard, GameMove, ResultOfAMove, PLAYER_1, PLAYER_2

# to load tf_agents properly, needs v1.enable_v2
tf.compat.v1.enable_v2_behavior()

policy_dirs = {
    '6x6': './__tf_agents__/othello_6x6_dqn_lr_e4/policy',
    '8x8': './__tf_agents__/othello_8x8_dqn_lr_e4/policy',
    '8x8_33M': './__tf_agents__/othello_8x8_dqn_lr_e4_33m/policy',
    '8x8_30M': './__tf_agents__/othello_8x8_dqn_lr_e4_30m/policy',
    '8x8_20M': './__tf_agents__/othello_8x8_dqn_lr_e4_20m/policy',
    '8x8_10M': './__tf_agents__/othello_8x8_dqn_lr_e4_10m/policy',
    '8x8_6M': './__tf_agents__/othello_8x8_dqn_lr_e4_6m/policy',
    '8x8_3M': './__tf_agents__/othello_8x8_dqn_lr_e4_3m/policy',
}

agent_policies = {
    '6x6': None,
    '8x8': None,
    '8x8_33M': None,
    '8x8_30M': None,
    '8x8_20M': None,
    '8x8_10M': None,
    '8x8_6M': None,
    '8x8_3M': None,
    '6x6_timestamp': 0,
    '8x8_timestamp': 0,
    '8x8_33M_timestamp': 0,
    '8x8_30M_timestamp': 0,
    '8x8_20M_timestamp': 0,
    '8x8_10M_timestamp': 0,
    '8x8_6M_timestamp': 0,
    '8x8_3M_timestamp': 0,
}


def load_policy(policy_dir, policy_key):
    tf.compat.v1.enable_v2_behavior()
    global agent_policies
    try:
        saved_policy = tf.compat.v2.saved_model.load(policy_dir)
        agent_policies[policy_key] = saved_policy
        return saved_policy
    except Exception as e:
        print('load policy failed', policy_dir)
        print(traceback.format_exc())
    return None


def get_policy(policy_key):
    lastupdate_timestamp = agent_policies[policy_key+'_timestamp']
    policy = agent_policies[policy_key]

    if lastupdate_timestamp + (5 * 60) < time.time():
        # too old, load policy
        print('policy is too old. Reloading '+policy_key)
        policy = None
        agent_policies[policy_key+'_timestamp'] = time.time()

    if policy == None:
        load_policy(policy_dirs[policy_key], policy_key)
    return agent_policies[policy_key]


def _to_tensor_observation(game_board, player_id, board_size):
    obs = np.array(game_board, dtype=np.float32).reshape(
        board_size, board_size, 1)
    #    dtype=np.float32).reshape(9)
    # flip the -1 or 1 for the spot pieces
    obs = obs * player_id
    obs = tf.convert_to_tensor(obs, dtype=tf.float32, name='observation')
    return obs


def get_next_action(policy, game_board, player_id, board_size) -> ResultOfAMove:
    # player_2 move
    # time_step had to be constructed in this way, to work with loaded policy
    step_type = tf.convert_to_tensor([0], dtype=tf.int32, name='step_type')
    reward = tf.convert_to_tensor([ts.StepType.MID],
                                  dtype=tf.float32,
                                  name='reward')
    discount = tf.convert_to_tensor([1.0],
                                    dtype=tf.float32,
                                    name='discount')
    observation = _to_tensor_observation(game_board,
                                         player_id,
                                         board_size)[None, :]
    opponent_ts = ts.TimeStep(step_type,
                              reward,
                              discount,
                              observation)
    action_step = policy.action(opponent_ts)
    action_code = action_step.action.numpy().item()
    # print('action:', action_code)
    return action_code


def get_next_move(board: GameBoard, player_id, policy):
    if player_id == PLAYER_1:
        valid_spots = board.possible_moves_player_1
    else:
        valid_spots = board.possible_moves_player_2

    if len(valid_spots) == 0:
        return GameMove(pass_turn=True)

    action_code = get_next_action(
        policy, board.board, player_id, board.board_size)
    move = GameMove.from_action_code(action_code, board_size=board.board_size)
    if ((move.pass_turn and len(valid_spots) > 0) or
            (not move.pass_turn and not move.spot in valid_spots)):
        print('******* random move ******* ', move.to_friendly_format())
        move = GameMove(random.choice(valid_spots))
    return move


def server_step(board: GameBoard, server_player_id, client_player_id) -> GameBoard:
    if server_player_id == PLAYER_1:
        valid_spots = board.possible_moves_player_1
    else:
        valid_spots = board.possible_moves_player_2

    if len(valid_spots) == 0:
        opponent_move = GameMove(pass_turn=True)
    else:
        # Let server pick a move
        opponent_action_code = \
            agent_client.agent_service_step(game_board=board.observe_board_2d(),
                                            server_player_id=server_player_id,
                                            client_player_id=client_player_id,
                                            board_size=board.board_size)
        opponent_move = GameMove.from_action_code(
            opponent_action_code, board_size=board.board_size)
        if ((opponent_move.pass_turn and len(valid_spots) > 0) or
                (not opponent_move.pass_turn and not opponent_move.spot in valid_spots)):
            print('******* server take random logic ******* ',
                opponent_move.to_friendly_format())
            opponent_move = GameMove(random.choice(valid_spots))
    move = opponent_move
    move_result = board.make_a_move(server_player_id, move)
    board = move_result.new_game_board

    if server_player_id == PLAYER_1:
        print(f'Player X, move: {move.to_friendly_format()}')
    else:
        print(f'Player O, move: {move.to_friendly_format()}')
    print(board)
    print('--------------------')
    return board


def agent_step(board: GameBoard, player_id, policy) -> GameBoard:
    move = get_next_move(board, player_id, policy)

    move_result = board.make_a_move(player_id, move)
    board = move_result.new_game_board

    if player_id == PLAYER_1:
        print(f'Player X, move: {move.to_friendly_format()}')
    else:
        print(f'Player O, move: {move.to_friendly_format()}')
    print(board)
    print('--------------------')
    return board


def fight(policy, agent_as_player_id) -> GameBoard:
    board_size = 8
    board = GameBoard(board_size=board_size, random_start=False)
    if agent_as_player_id == PLAYER_1:
        while not board.game_ended:
            # player 1
            board = agent_step(board, PLAYER_1, policy)
            if board.game_ended:
                break
            # player 2
            board = server_step(board, PLAYER_2, PLAYER_1)
    else:
        while not board.game_ended:
            # player 1
            board = server_step(board, PLAYER_1, PLAYER_2)
            if board.game_ended:
                break
            # player 2
            board = agent_step(board, PLAYER_2, policy)

    return board


def get_args():
    # Get some basic command line arguements
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--policy', help='Which policy to be loaded as the agent', type=str, default='8x8_20M')
    parser.add_argument('--num_games', help='No. of games',
                        type=int, default=10)
    return parser.parse_args()


def run_some_games(policy, as_player_id, num_games):
    wins = {'agent': 0,
            'server': 0,
            'ties': 0,
            }
    for _ in range(num_games):
        if as_player_id == PLAYER_1:
            board = fight(policy, PLAYER_1)
            if board.player_1_count > board.player_2_count:
                wins['agent'] = wins['agent'] + 1
                print('Player X won')
            elif board.player_2_count > board.player_1_count:
                wins['server'] = wins['server'] + 1
                print('Player O won')
            else:
                wins['ties'] = wins['ties'] + 1
            print('==================================')
        else:
            board = fight(policy, PLAYER_2)
            if board.player_1_count < board.player_2_count:
                wins['agent'] = wins['agent'] + 1
                print('Player X won')
            elif board.player_2_count < board.player_1_count:
                wins['server'] = wins['server'] + 1
                print('Player O won')
            else:
                wins['ties'] = wins['ties'] + 1
            print('==================================')
    return wins


def main():
    args = get_args()

    policy = get_policy(args.policy)

    wins_1 = run_some_games(policy, PLAYER_1, args.num_games)
    wins_2 = run_some_games(policy, PLAYER_2, args.num_games)

    print('Agent as Player X:', wins_1)
    print('Agent as Player O:', wins_2)

    # demo_main(board_size=8, random_start=False)


if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    main()
