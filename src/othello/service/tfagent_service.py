import traceback
import time

from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from flask_jsonpify import jsonify

import numpy as np
import tensorflow as tf
import tf_agents
from tf_agents.trajectories import time_step as ts

from othello.game.GameBoard import GameBoard, GameMove, ResultOfAMove, PLAYER_1, PLAYER_2


app = Flask(__name__)
api = Api(app)

policy_dirs = {
    '6x6': './__tf_agents__/othello_6x6_dqn_lr_e4/policy',
    '8x8': './__tf_agents__/othello_8x8_dqn_lr_e4/policy',
}
agent_policies = {
    '8x8': None,
    '6x6': None,
    '8x8_timestamp': 0,
    '6x6_timestamp': 0,
}


# to load tf_agents properly, needs v1.enable_v2
tf.compat.v1.enable_v2_behavior()


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


# -------------------------------------------------------------------------------
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

# -------------------------------------------------------------------------------


class Agent6x6(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self):
        policy = get_policy('6x6')
        return {'employees': '6x6: hello there.' + str(get_policy('6x6'))}

    def post(self):
        # request.method 'POST'
        # print('flask request:', request.json)
        policy = get_policy('6x6')
        action_code = get_next_action(policy,
                                      request.json['game_board'],
                                      request.json['server_player_id'],
                                      board_size=6)
        return {'action_code': action_code}


class Agent8x8(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self):
        return {'employees': '8x8: hello there.' + str(get_policy('8x8'))}
        # query = conn.execute("select * from employees where EmployeeId =%d "  %int(employee_id))
        # result = {'data': [dict(zip(tuple (query.keys()) ,i)) for i in query.cursor]}
        # return jsonify(result)
    def post(self):
        # request.method 'POST'
        # print('flask request:', request.json)
        policy = get_policy('8x8')
        action_code = get_next_action(policy,
                                      request.json['game_board'],
                                      request.json['server_player_id'],
                                      board_size=8)
        return {'action_code': action_code}


api.add_resource(Agent6x6, '/6x6_step')  # Route for 6x6 agent policy
api.add_resource(Agent8x8, '/8x8_step')  # Route for 8x8 agent policy


def main(*args, **kwargs):
    app.run(port='5000')


if __name__ == '__main__':
    tf_agents.system.multiprocessing.handle_main(main)
