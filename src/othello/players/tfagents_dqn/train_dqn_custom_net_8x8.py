# c51 - different from DQN, C51 is based on DQN. C51 predicts
# a histogram model for the probability distribution of the Q-Value

import logging
import sys
from datetime import datetime

import tensorflow as tf
import tf_agents
from tf_agents.networks import q_network, categorical_q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import random_tf_policy, policy_saver, py_tf_eager_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from othello.env.TFAgentsOthelloEnv import OthelloEnv
from othello.game import GameBoard
from othello.players.tfagents_dqn.CustomNN import CustomNN8x8
import othello.tf_utils as tf_utils


def create_envs(board_size=8,
                random_rate: float = 0.0,
                as_player_2_rate: float = 0.0,
                use_agent_service=False):
    def _create_train_env():
        return OthelloEnv(board_size=board_size,
                          random_rate=random_rate,
                          as_player_2_rate=as_player_2_rate,
                          use_agent_service=use_agent_service)
        #   existing_agent_policy_path=old_policy_path)
    # Environment
    train_py_env = OthelloEnv(board_size=board_size,
                              random_rate=random_rate,
                              as_player_2_rate=as_player_2_rate,
                              use_agent_service=use_agent_service)
    eval_py_env = OthelloEnv(board_size=board_size,
                             random_rate=random_rate,
                             as_player_2_rate=as_player_2_rate,
                             use_agent_service=use_agent_service)

    # train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    train_env = \
        tf_py_environment.TFPyEnvironment(
            ParallelPyEnvironment(
                env_constructors=[lambda: _create_train_env() for _ in range(8)]))
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    return train_env, eval_env, train_py_env, eval_py_env


def update_envs_with_agent(train_py_env, eval_py_env, agent):
    # put agent to TicTaxToeEnv
    train_py_env._agent = agent
    eval_py_env._agent = agent


def create_agent(train_env, global_step_counter):
    # q network
    q_net = CustomNN8x8(train_env.observation_spec(),
                        train_env.action_spec())

    # optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    agent = dqn_agent.DqnAgent(train_env.time_step_spec(),
                               train_env.action_spec(),
                               q_network=q_net,
                               optimizer=optimizer,
                               n_step_update=n_step_update,
                               td_errors_loss_fn=common.element_wise_squared_loss,
                               gamma=gamma,
                               train_step_counter=global_step_counter)
    agent.initialize()

    # Training
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)
    # Reset the train step
    agent.train_step_counter.assign(0)

    return agent


def compute_avg_return(environment, policy, num_episodes=10):
    # average return to evaluate the training and agent
    best_episode_return = -1_000_000.0
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward

        best_episode_return = max(best_episode_return, episode_return)
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0], best_episode_return.numpy()[0]


def collect_step(environment, policy, replay_buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    # batch = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), traj)
    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)

# This collect loop is so common in RL, that we provide standard implementations of
# these. For more details see the drivers module.


def rot90_trajectories(trajs, board_size=8):
    observation = tf_utils.rot90_5d_batch(trajs.observation)
    action = trajs.action
    action_shape = action.shape
    action = tf.reshape(action, [-1])
    action = tf.map_fn(
        fn=lambda a: GameBoard.GameMove.rot90_action_code(
            a, board_size=board_size),
        elems=action)
    action = tf.reshape(action, action_shape)

    trajs.replace(observation=observation, action=action)
    return trajs


def create_replay_buffer(train_env, agent):
    # replay_buffer - for data collection. so that training can use the collected Trajectory
    replay_buffer = \
        tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                       batch_size=train_env.batch_size,
                                                       max_length=replay_buffer_capacity)
    # Dataset generates trajectories with shape [BxTx...] where
    # T = n_step_update + 1.
    dataset = replay_buffer.as_dataset(num_parallel_calls=8,
                                       sample_batch_size=batch_size,
                                       num_steps=n_step_update + 1)  # .prefetch(3)

    replay_buffer_itr = iter(dataset)  # <---- iterator being used in training

    return replay_buffer, replay_buffer_itr


def _train_agent(num_batches, agent,
                 train_env, eval_env,
                 replay_buffer_itr, replay_buffer,
                 board_size=8):
    # Evaluate the agent's policy once before training.
    avg_return, best_episode_return = compute_avg_return(eval_env,
                                                         agent.policy,
                                                         num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_batches):
        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_batch):
            collect_step(train_env, agent.collect_policy, replay_buffer)

        # Sample a batch of data from the buffer and update the agent's network.
        # trajectories = replay_buffer.gather_all()
        trajectories_1, unused_info = next(replay_buffer_itr)
        train_loss_1 = agent.train(trajectories_1)

        # rotate 90 1st time
        trajectories_2 = rot90_trajectories(
            trajectories_1, board_size=board_size)
        train_loss_2 = agent.train(trajectories_2)

        # rotate 90 2rd time
        trajectories_3 = rot90_trajectories(
            trajectories_2, board_size=board_size)
        train_loss_3 = agent.train(trajectories_3)

        # rotate 90 3rd time
        trajectories_4 = rot90_trajectories(
            trajectories_3, board_size=board_size)
        train_loss_4 = agent.train(trajectories_4)

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print(
                f'step = {step}: loss = {train_loss_1.loss}, {train_loss_2.loss}, {train_loss_3.loss}, {train_loss_4.loss}')

        if step % eval_interval == 0:
            avg_return, best_episode_return = compute_avg_return(eval_env,
                                                                 agent.policy,
                                                                 num_eval_episodes)
            print('step = {0}: Average Return = {1:.2f} Best = {2:.2f}'.format(
                step, avg_return, best_episode_return))
            returns.append(avg_return)
        # flush stdout
        sys.stdout.flush()


def train_agent_and_save(board_size=8, random_rate=0.0, as_player_2_rate=0.5):
    """Load training checkpoint, then do training. Save at the end. """
    global_step_counter = tf.compat.v1.train.get_or_create_global_step()

    train_env, eval_env, train_py_env, eval_py_env = create_envs(
        board_size=board_size,
        random_rate=random_rate,
        as_player_2_rate=as_player_2_rate,
        use_agent_service=True)
    agent = create_agent(train_env, global_step_counter)
    # update_envs_with_agent(train_py_env, eval_py_env, agent)

    replay_buffer, replay_buffer_itr = create_replay_buffer(train_env, agent)

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())
    # compute_avg_return(eval_env, random_policy, num_eval_episodes)
    for _ in range(num_random_collect_steps):  # collect a few records.
        collect_step(train_env, random_policy, replay_buffer)

    train_checkpointer = common.Checkpointer(ckpt_dir=checkpoint_dir,
                                             max_to_keep=1,
                                             agent=agent,
                                             policy=agent.policy,
                                             replay_buffer=replay_buffer,
                                             global_step=global_step_counter)
    # restory checkpoint
    train_checkpointer.initialize_or_restore()
    # Setup work is done new. ////////

    for i in range(500):
        print('*****************************')
        # training ------------------------
        _train_agent(num_batches, agent, train_env,
                     eval_env, replay_buffer_itr, replay_buffer,
                     board_size=board_size)
        print(f'============{i}==============')

        # Save agent checkpointe ------------------------
        # save checkpoint
        train_checkpointer.save(global_step_counter)

        # save policy
        tf_policy_saver = policy_saver.PolicySaver(agent.policy)
        tf_policy_saver.save(policy_dir)

        # demo -----------
        demo_game_play(agent.policy, eval_env, eval_py_env,
                       random_rate=random_rate,
                       as_player_2_rate=as_player_2_rate)
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f'Saved policy and training checkpoint. {dt_string}')
        print(f'============{i}==============')

    return agent, eval_env, eval_py_env


def load_policy(policy_dir):
    try:
        saved_policy = tf.compat.v2.saved_model.load(policy_dir)
        return saved_policy
    except:
        print('load policy failed', policy_dir)
    return None


def demo_game_play(agent_policy, eval_env, eval_py_env,
                   random_rate=0.0, as_player_2_rate=0.0):
                   # as_player_2_rate is not used. eval_py_env has player2 rate set.
    # log game play
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        # level=logging.DEBUG)
                        level=logging.INFO)

    old_log_on = eval_py_env._log_on
    old_random_rate = eval_py_env._random_rate

    for _ in range(3):
        time_step = eval_env.reset()
        eval_py_env._log_on = True
        eval_py_env._random_rate = random_rate

        while not time_step.is_last():
            action_step = agent_policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            # print('--------', time_step)

        print('==============================')
    # reset eval_env
    eval_py_env._log_on = old_log_on
    eval_py_env._random_rate = old_random_rate


# ------------------------------------------------------------
checkpoint_dir = './__tf_agents__/othello_8x8_dqn_lr_e4/checkpoint'
policy_dir = './__tf_agents__/othello_8x8_dqn_lr_e4/policy'

# num_iterations = 105_000  # @param {type:"integer"}

num_random_collect_steps = 10  # @param {type:"integer"}
num_batches = 5_000  # @param {type:"integer"}
collect_steps_per_batch = 1  # @param {type:"integer"}
replay_buffer_capacity = 100000  # @param {type:"integer"}

# conv_layer_params = [(32, 2, 1), ]
# fc_layer_params = (128, 64,)

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-4  # @param {type:"number"}
gamma = 0.98
log_interval = 200  # @param {type:"integer"}

num_atoms = 51  # @param {type:"integer"}
min_q_value = -20  # @param {type:"integer"}
max_q_value = 20  # @param {type:"integer"}
n_step_update = 2  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}


def train_main(board_size=8, random_rate=0.0,
               as_player_2_rate: float = 0.0,):
    agent, eval_env, eval_py_env = train_agent_and_save(board_size=board_size,
                                                        random_rate=random_rate,
                                                        as_player_2_rate=as_player_2_rate)


def demo_main(board_size=8, random_rate=0.0, as_player_2_rate=0.0):
    policy = load_policy(policy_dir)
    train_env, eval_env, train_py_env, eval_py_env = create_envs(
        board_size=board_size, random_rate=random_rate)
    demo_game_play(policy, eval_env, eval_py_env,
                   random_rate=random_rate,
                   as_player_2_rate=as_player_2_rate)


def main(*args, **kwargs):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f'starting {dt_string}')
    print('-----------------------------------------------')
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        # level=logging.DEBUG)
                        level=logging.INFO)
    train_main(board_size=8,
               random_rate=0.0,
               as_player_2_rate=1.7)
               # set to 0.7, see whether it can find way out of quick-death plays.


if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    tf_agents.system.multiprocessing.handle_main(main)
