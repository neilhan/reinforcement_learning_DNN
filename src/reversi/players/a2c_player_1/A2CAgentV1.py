import random
import numpy as np
import tensorflow as tf
import logging

from reversi.players.a2c_player_1.GameWrapper import GameWrapper
from reversi.players.a2c_player_1.A2CAgentNN import A2CAgentNN
from reversi.game import GameBoard


class A2CAgentV1:
    def __init__(self, model: A2CAgentNN,
                 learn_rate=5e-3,
                 ent_coef=0.0001,
                 value_coef=0.5,
                 gamma=0.99):
        self.value_coef = value_coef
        self.ent_coef = ent_coef
        self.gamma = gamma

        self.model = model
        self.model._model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learn_rate),
                                  loss=[self._logits_loss, self._value_loss])

    def train(self, game: GameWrapper, batch_size=10, train_for_num_batches=40):
        logging.info('training starts: batch size: %04d for batchs: %05d' %
                     (batch_size, train_for_num_batches))
        # prepare batch arrayes
        actions = np.empty((batch_size,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_size))
        observations = np.empty((batch_size, game.get_observation_size()))

        game_done_rewards = []
        next_observation = game.reset()
        # training loop. sample, train
        for b in range(train_for_num_batches):
            if b % 10 == 0:
                logging.info(game.game_board)
            if b % 3:  # reset game, after 3 batchs.
                next_observation = game.reset()
            if b >= (train_for_num_batches - 3):
                game.set_log_play(True)

            counter_invalid_moves = 0
            # for each step in the batch:
            # A. as a player, play a step, collecting reward, and observe.
            # B. Then switch to next player.
            # (the observe is alway return the board as mine vs opponent)
            for i in range(batch_size):
                # 1. find which should be the action, estimate value. by send the model current observation
                observations[i] = next_observation.copy()
                actions[i], values[i] = self.model.get_action_value(next_observation[None, :],
                                                                    game.get_all_valid_moves(),
                                                                    force_valid=False)
                if b % 10 == 0:
                    logging.info(str(game.game_board) +
                                 ' action:' + str(actions[i]))
                # 2. play the action on the game
                next_observation, rewards[i], dones[i], game, is_move_valid = \
                    game.execute_move(actions[i])

                if not is_move_valid:
                    counter_invalid_moves = counter_invalid_moves + 1
                # if the move is a valid, and game is not done, we let opponent play one piece
                if is_move_valid and not dones[i]:
                    # We need to let the next player make one move.
                    # Discard next_observation.
                    # From the current player's view, as part of the Env,
                    # the opponent needs to make a move if game is not done.
                    # the observation after the opponent's turn is the consiquence
                    # of player's action.
                    opponent_action, opponent_value = \
                        self.model.get_action_value(next_observation[None, :],
                                                    game.get_all_valid_moves(),
                                                    force_valid=True)
                    next_observation, opponent_reward, dones[i], game, is_move_valid = \
                        game.execute_move(opponent_action)
                    if not is_move_valid:
                        opponent_action = game.pick_a_random_valid_move()
                        next_observation, opponent_reward, dones[i], game, is_move_valid = \
                            game.execute_move(opponent_action)

                # if done, reset game
                if dones[i]:
                    rewards[i] = rewards[i] - opponent_reward
                    game_done_rewards.append(rewards[i])
                    logging.debug(game.game_board)
                    next_observation = game.reset()
                    # randomly switch to player_2 for the new game
                    # TODO donot switch to player2
                    if False and bool(random.getrandbits(1)):
                        first_action, _ = \
                            self.model.get_action_value(next_observation[None, :],
                                                        game.get_all_valid_moves(),
                                                        force_valid=True)
                        # move can be illegal move
                        next_observation, _, _, game, is_move_valid = \
                            game.execute_move(first_action)
                    logging.info('Episode: %04d, End game Reward: %03d' %
                                 (len(game_done_rewards), rewards[i]))
            # //// for loop of this batch
            # now we have the batch for training
            _, next_value = self.model.get_action_value(next_observation[None, :],
                                                        game.get_all_valid_moves(),
                                                        force_valid=False)
            returns, advs = self._returns_advantages(
                rewards, dones, values, next_value)
            # send actions, advantages. zip them
            acts_and_advs = np.concatenate(
                [actions[:, None], advs[:, None]], axis=-1)
            # Train
            losses = self.model._model.train_on_batch(x=observations,
                                                      y=[acts_and_advs, returns])
            logging.info('[%d/%d] Losses: %s, %d' %
                         (b+1, train_for_num_batches, losses, counter_invalid_moves))

    def _returns_advantages(self, rewards, dones, values, next_value):
        # convert done to int array
        # next_value is the estimate of the future state. critic
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # calc returns: discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            # if game is done, done count in future rewards. *(1-dones[t])
            returns[t] = \
                rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        # advantages: returns - baseline (estimate)
        advantages = returns - values
        return returns, advantages

    def _value_loss(self, returns, value):
        # build the loss_fn
        return self.value_coef * tf.keras.losses.mean_squared_error(returns, value)

    def _logits_loss(self, actions_and_advantages, logits):
        # split the advantage and actions
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        # policy gradients, weighted by advantages. only calc on actions we take.
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(
            actions, logits, sample_weight=advantages)
        # entropy loss, by cross-entropy over itself
        probs = tf.nn.softmax(logits)
        entropy_loss = tf.keras.losses.categorical_crossentropy(probs, probs)
        # loss function: to minimize policy and maximize entropy losses.
        # flip signs, optimizer minimizes
        return policy_loss - self.ent_coef * entropy_loss


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        # level=logging.DEBUG)
                        level=logging.INFO)

    # create the Game
    board_size = 6
    game = GameWrapper(1, board_size=board_size)

    # create the agent
    agent_nn = A2CAgentNN(action_size=game.get_action_size(),
                          input_size=game.get_observation_size())
    agent = A2CAgentV1(agent_nn)

    # Train
    agent.train(game, 100, 1000)
