import logging
import numpy as np
import multiprocessing
from multiprocessing import Process, Pipe

from othello.players.a2c_player_2.GameWrapper import GameWrapperInpatient
from othello.players.a2c_player_2.A2CAgentCNN import A2CAgentCNN


class CloudpickleWrapper():
    """
    cloudpickle to serialize content
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


# This is part of the sub_process.
def _get_a_train_batch(agent,
                       game: GameWrapperInpatient,
                       n_step: int = 5,
                       log_action_n_board: bool = False):
    """
    observations, actions, rewards, dones, values are the batch the sub_process 
    is preparing.
    """
    # agent is A2CAgentV2
    def _opponent_move(obs) -> float:
        # We need to let the next player make one move.
        # Opponent_reward should be 0, or if won/loose, the end game reward.
        # Discard next_observation.
        # From the current player's view, as part of the Env,
        # the opponent needs to make a move if game is not done.
        # the observation after the opponent's turn is the consiquence
        # of player's action.
        opponent_action, opponent_value = \
            agent.model.get_action_value(obs[None, :],
                                         game.get_all_valid_moves(),
                                         force_valid=True)
        next_observation, opponent_reward, dones[i], game, is_move_valid = \
            game.execute_move(opponent_action)
        if not is_move_valid:  # pike random move since the agent isn't smart enough
            opponent_action = game.pick_a_random_valid_move()
            next_observation, opponent_reward, done, game, is_move_valid = \
                game.execute_move(opponent_action)
        return opponent_reward

    def _reset_game():
        # return game observation
        next_observation = game.reset()  # reset game <--------------
        # randomly switch to player_2 for the new game
        if bool(random.getrandbits(1)):
            first_action, _ = \
                agent.model.get_action_value(next_observation[None, :],
                                             game.get_all_valid_moves(),
                                             force_valid=True)
            # move can be illegal move
            next_observation, _, _, game, is_move_valid = \
                game.execute_move(first_action)
        return next_observation

    def _returns_advantages(rewards, dones, values, next_value):
        # var dones is int array. 0-not done, 1 - game is done.
        # next_value is the estimation of the future state. so we can calc critic
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # calc returns: discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            # if game is done, done count in future rewards. *(1-dones[t])
            returns[t] = \
                rewards[t] + agent.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        # advantages: returns - baseline (estimate)
        advantages = returns - values
        return returns, advantages

    observations = np.empty((n_step,) + game.get_vision_shape())
    actions = np.empty((n_step,), dtype=np.int32)
    rewards, dones, values = np.empty((3, n_step))

    # for each step in the batch:
    # A. as a player, play a step, collecting reward, and observe.
    # B. Then switch to next player.
    # (the observe is alway return the board as mine vs opponent)
    next_observation = np.array(game.observe(game.current_player))
    for i in range(n_step):
        # 1. find which should be the action, estimate value
        #    by send the model current observation
        observations[i] = np.copy(next_observation)
        actions[i], values[i] = \
            agent.model.get_action_value(observations[i][None, :],
                                         game.get_all_valid_moves(),
                                         force_valid=False)
        if log_action_n_board:
            logging.info(str(game.game_board) + '\nNext move:' +
                         str(game.convert_action_to_spot(actions[i]).to_friendly_format()))
        # 2. play the action on the game
        next_observation, rewards[i], dones[i], game, is_move_valid = \
            game.execute_move(actions[i])

        if not is_move_valid:  # if move invalid, reset game, and start again
            next_observation = game.reset()
            dones[i] = 1
        # if the move is valid, and game is not done, we let opponent play one piece
        if is_move_valid and not dones[i]:
            opponent_reward = _opponent_move(next_observation)
        else:
            opponent_reward = 0

        # if done, reset game
        if dones[i]:
            rewards[i] = rewards[i] - opponent_reward
            next_observation = _reset_game()
            logging.info('End game Reward: %03d ******' % (rewards[i]))
        # reduce reset_game counter
    # //// for loop of this batch
    # we have: observations, actions, rewards, dones, values,
    # now we can prepare the action advantage for training
    _, next_value = agent.model.get_action_value(next_observation[None, :],
                                                 game.get_all_valid_moves(),
                                                 force_valid=False)
    returns, advs = _returns_advantages(rewards, dones, values, next_value)
    # send actions, advantages. zip them
    acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
    return observations, acts_and_advs, returns


def worker_process(conn, agent, game):
    n_step = 5

    while True:
        cmd, data = conn.recv()
        print('*** received command:', cmd)
        if cmd == 'get_a_train_batch':
            result = _get_a_train_batch(agent, game, n_step=n_step)
            print('After _get_batch. before conn.sent(result)')
            conn.send(result)
        elif cmd == 'log':
            logging.info(game.game_board)
        elif cmd == 'reset':
            result = game.reset()
            conn.send(result)
        elif cmd == 'close':
            logging.info('worker_process close()')
            break  # // while loop
        else:
            raise NotImplementedError
    conn.close()


def get_agent_game_fn(id, board_size):
    # create game wrapper
    def _new_agent_n_game_fn():
        # return agent, new game
        return agent, GameWrapperInpatient(id, board_size=board_size)
    return _new_agent_n_game_fn


class GameWrapperParallel():
    def __init__(self, agent, num_sub_processes, board_size):
        """
        envs: list of gym environments to run in subprocesses
        """
        # tensolflow hangs if not set this
        multiprocessing.set_start_method('spawn')

        self.closed = False

        games = [GameWrapperInpatient(i, board_size)
                 for i in range(num_sub_processes)]

        self.parent_conn, self.child_conn = zip(
            *[Pipe() for _ in range(num_sub_processes)])
        self.sub_processes = [Process(target=worker_process,
                                      args=(child_conn, agent, game))
                              for (child_conn, game) in zip(self.child_conn, games)]
        for p in self.sub_processes:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        # for child_conn in self.child_conn:
        #     child_conn.close()

    def get_a_train_batch(self):
        for conn in self.parent_conn:
            conn.send(('get_a_train_batch', 0))
        print('before parent_conn recv()')
        results = [conn.recv() for conn in self.parent_conn]
        observations, acts_and_advs, returns = zip(*results)
        return np.stack(observations), np.stack(acts_and_advs), np.stack(returns)

    def reset(self):
        for remote in self.parent_conn:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.parent_conn])

    def close(self):
        if self.closed:
            return

        for remote in self.parent_conn:
            remote.send(('close', None))
        for p in self.sub_processes:
            p.join()
        self.closed = True

    def log(self):
        for conn in self.parent_conn:
            conn.send(('log', 0))

    @property
    def num_envs(self):
        return len(self.parent_conn)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        # level=logging.DEBUG)
                        level=logging.INFO)

    def fn():
        print('fn()')
        return 1, 2
    games = GameWrapperParallel([fn for _ in range(16)])
    # games.step([i for i in range(16)])
    games.close()
