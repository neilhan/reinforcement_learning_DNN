import requests


def agent_service_step(game_board, server_player_id, client_player_id, board_size,
                       evaluate=False):
    request_json = {
        "client_player_id": client_player_id,
        "server_player_id": server_player_id,
        "board_size": board_size,
        "game_board": game_board,
        "evaluate": evaluate,
    }

    resp = \
        requests.post(f'http://localhost:5000/{int(board_size)}x{int(board_size)}_step',
                      json=request_json)
    # print(resp)
    if resp.status_code != 200:
        # This means something went wrong.
        raise Exception('GET /tasks/ {}'.format(resp.status_code))
    action = resp.json()
    # print('action from post:', action)
    return action['action_code']


if __name__ == '__main__':
    action_code = agent_service_step(game_board=[[0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0],
                                                 [0, 0, 1, -1, 0, 0],
                                                 [0, 0, -1, 1, 0, 0],
                                                 [0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0],
                                                 ],
                                     server_player_id=-1,
                                     client_player_id=1,
                                     board_size=6)
    print('received action_code:', action_code)
