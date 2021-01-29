from typing import Any, List, Sequence, Tuple
import logging
import random
import numpy as np
import tensorflow as tf
# import tqdm

from reversi.tf_utils import set_global_seeds
from reversi.players.a2c_player_4.A2CModel import A2CModel
from reversi.players.a2c_player_4.GameWrapper import GameWrapperInpatient

import gym


class A2CTrainer:
    # huber_loss: L_d(G, V) - less sensitive to outliers in data than squared-error loss
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    # Small epsilon value for stabilizing division operations
    eps = np.finfo(np.float32).eps.item()

    def __init__(self,
                 env: GameWrapperInpatient,
                 model: A2CModel,
                 model_save_path='./__models__/a2c_player_4/',
                 optimizer_learn_rate=0.001):
        # the action 0, to 63 are the moves to take. The action 64 is pass this turn.
        # the last one is pass to other player
        self.PASS_TURN_ACTION = model.num_actions

        self.env = env  # gym.make("CartPole-v0")
        self.is_log_env = False
        self.model_save_path = model_save_path

        # self.optimizer = tf.keras.optimizers.RMSprop(lr=optimizer_learn_rate,
        #                                              momentum=0.05)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=optimizer_learn_rate)
        self.dist = lambda lgt: tf.squeeze(
            tf.random.categorical(tf.math.log(lgt), 1), axis=-1)

        self.model = model

    def get_action_value(self, observation, all_valid_moves, force_valid=False):
        action_logits, output_values = self.model.call(observation)
        sampled_action = self.dist(action_logits)
        # # This following logic is when forcing the action to a valid move
        if force_valid:
            if len(all_valid_moves) > 0 and not sampled_action in all_valid_moves:
                logging.debug(
                    'get_action_value() pick random action')
                sampled_action = random.choice(all_valid_moves)
            elif len(all_valid_moves) == 0 and sampled_action != self.PASS_TURN_ACTION:
                logging.debug(
                    'get_action_value() forced correction pass')
                sampled_action = self.PASS_TURN_ACTION

        return (sampled_action, output_values[:, 0])

    def _opponent_step(self, state):
        # let opponent's move ------
        valid_moves = self.env.get_all_valid_moves()
        # op_action, op_action_probs_t, op_value = self.model A2CTrainer._sample_action_n_value(state)
        opponent_action, opponent_value = self.get_action_value(state[None, :], valid_moves,
                                                                force_valid=True)
        opponent_observation, opponent_reward, done, _env, is_move_valid = \
            self.env.execute_move(opponent_action)
        return opponent_observation, opponent_reward, done

    def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        """
        execute the move, and let component do their move
        """
        def _env_step_fn(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Returns state, reward and done flag given an action."""
            # Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
            # This would allow it to be included in a callable TensorFlow graph.
            if self.is_log_env:
                logging.info('Taking: {0}'.format(
                    self.env.convert_action_to_spot(action).to_friendly_format()))

            state, reward, done, _env, is_move_valid = \
                self.env.execute_move(action)

            if not done:
                opponent_observation, opponent_reward, done = self._opponent_step(state)
                # update state
                state = opponent_observation
                if done:
                    reward = reward - opponent_reward

            return (state.astype(np.float32),
                    np.array(reward, np.int32),
                    np.array(done, np.int32))

        return tf.numpy_function(_env_step_fn,
                                 [action],
                                 [tf.float32, tf.int32, tf.int32])

    @staticmethod
    def _sample_action_n_value(state: tf.Tensor, model) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Returns action, action_probabilities, value in Tensor. 
        shape: int, (1, num_actions), float
        """
        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)

        # Model pick an action ---------------------------
        # Run the model and to get action probabilities and critic value
        action_logits_t, value = model.call(state)

        # Sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)
        return action, action_probs_t, value

    def run_episode(self,  # model: tf.keras.Model,
                    initial_state: tf.Tensor,
                    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, ]:
        """Runs a single episode to collect training data."""

        action_probs = tf.TensorArray(
            dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps):
            action, action_probs_t, value = A2CTrainer._sample_action_n_value(
                state, self.model)

            # Store critic values
            values = values.write(t, tf.squeeze(value))

            # Store log probability of the action chosen
            action_probs = action_probs.write(t, action_probs_t[0, action])

            # Game Step ---------------------------
            # Apply action to the environment to get next state and reward
            state, reward, done = self.tf_env_step(action)
            state.set_shape(initial_state_shape)

            # Store reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

    @tf.function
    def train_step(self,  # model: tf.keras.Model,
                   initial_state: tf.Tensor,
                   optimizer: tf.keras.optimizers.Optimizer,
                   gamma: float,
                   max_steps_per_episode: int) -> tf.Tensor:
        """Runs a model training step."""

        # 1. run game for one episode. Collect training data, -> loss
        with tf.GradientTape() as tape:
            # Run the model for one episode to collect training data
            action_probs, values, rewards = self.run_episode(
                initial_state, max_steps_per_episode)

            # Calculate expected returns
            returns = A2CTrainer.get_expected_return(rewards, gamma)

            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

            # Calculating loss values to update our network
            loss = A2CTrainer.compute_loss(action_probs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.model._model.trainable_variables)

        # Apply the gradients to the model's parameters
        optimizer.apply_gradients(
            zip(grads, self.model._model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward

    def train(self,  # model,  # the model to train
              max_episodes=10_000,
              max_steps_per_episode=1_000,
              gamma=0.99):
        """
        gamma - Discount factor for future rewards
        """

        # Cartpole-v0 is considered solved if average reward is >= 195 over 100
        # consecutive trials
        reward_threshold = 195
        running_reward = 0
        log_reward_interval = 500
        log_game_interval = 5_000
        save_model_interval = 15_000
        best_reward = -99999999
        best_reward_batch = -99999999

        # with tqdm.trange(max_episodes) as t:
        #   for i in t:
        for i in range(max_episodes):
            initial_state_np = self.env.reset()
            # random switch to player 2
            if bool(random.getrandbits(1)):
                initial_state_np, _, _ = self._opponent_step(initial_state_np)

            if i % log_game_interval == 0:
                self.is_log_env = True
                self.env.set_log_play(True)

            initial_state = tf.constant(initial_state_np, dtype=tf.float32)
            episode_reward = int(self.train_step(initial_state,  # model,
                                                 self.optimizer, gamma,
                                                 max_steps_per_episode))

            running_reward = episode_reward * 0.01 + running_reward * 0.99
            if episode_reward > best_reward:
                best_reward = episode_reward
            if episode_reward > best_reward_batch:
                best_reward_batch = episode_reward

            # Show average episode reward every 10 episodes
            if i % log_reward_interval == 0:
                logging.info(
                    f'Episode {i}: Over all Best: {best_reward}, Batch Best: {best_reward_batch}, Average reward: {running_reward}')
                best_reward_batch = -999999999

            if i % log_game_interval == 0:
                self.is_log_env = False
                self.env.set_log_play(False)

            if i % save_model_interval == 0:
                self.model._model.save(self.model_save_path)
            if running_reward > reward_threshold:
                break

        print(
            f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')

    @staticmethod
    def get_expected_return(rewards: tf.Tensor,
                            gamma: float,
                            standardize: bool = True) -> tf.Tensor:
        """Compute expected returns per timestep."""

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array. [::-1] reverts an array.
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) /
                       (tf.math.reduce_std(returns) + A2CTrainer.eps))  # eps to avoid /0 error

        return returns

    @staticmethod
    def compute_loss(action_probs: tf.Tensor,
                     values: tf.Tensor,
                     returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = A2CTrainer.huber_loss(values, returns)

        return actor_loss + critic_loss
