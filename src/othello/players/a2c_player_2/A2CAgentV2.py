import random
import numpy as np
import tensorflow as tf
import logging

from othello.players.a2c_player_2.A2CAgentCNN import A2CAgentCNN
from othello.players.a2c_player_2.A2CAgentNN import A2CAgentNN
from othello.players.a2c_player_2.GameWrapper import GameWrapperInpatient as GameWrapper
from othello.players.a2c_player_2.GameWrapperParallel import GameWrapperParallel


class A2CAgentV2:
    """
    will make this Agent work with GameWrapperParallel
    """

    def __init__(self,
                 model: A2CAgentCNN,
                 model_save_path,
                 learn_rate=0.007,
                 ent_coef=0.500,
                 value_coef=0.5,
                 gamma=0.99,
                 model_save_interval=1000,):
        self.model_save_path = model_save_path
        self.model_save_interval = model_save_interval

        self.value_coef = value_coef
        self.ent_coef = ent_coef
        self.gamma = gamma

        self.model = model
        self.model._model.compile(
            optimizer=tf.keras.optimizers.RMSprop(lr=learn_rate,
                                                  momentum=0.01),
            loss=[self._logits_loss_fn, self._value_loss_fn])

    def train(self, game: GameWrapperParallel, batch_size=10, train_for_num_batches=40):
        logging.info('training starts: batch size: %04d for batchs: %05d' %
                     (batch_size, train_for_num_batches))
        # prepare batch arrayes
        # training loop. sample, train
        for b in range(train_for_num_batches):
            if b % 50 == 0:
                game.log()

            if b % self.model_save_interval == 0:
                self.model.save_model(self.model_save_path)

            # get the training batch
            observations, acts_and_advs, returns = game.get_a_train_batch()
            # Train with the new batch
            losses = self.model._model.train_on_batch(x=observations,
                                                      y=[acts_and_advs, returns])
            logging.info(
                '[%d/%d] Invalide moves: Losses: %s' % (b+1, train_for_num_batches, losses))
        # ///// batch loop

    def _value_loss_fn(self, returns, value):
        # build the loss_fn
        return self.value_coef * tf.keras.losses.mean_squared_error(returns, value)

    def _logits_loss_fn(self, actions_and_advantages, logits):
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
                          input_size=game.get_vision_shape())
    agent = A2CAgentV2(agent_nn)

    # Train
    agent.train(game, 5, 10000)
