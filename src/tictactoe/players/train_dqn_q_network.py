# c51 - different from DQN, C51 is based on DQN. C51 predicts
# a histogram model for the probability distribution of the Q-Value

import logging
import tensorflow as tf

from tf_agents.networks import q_network, categorical_q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import random_tf_policy, policy_saver, py_tf_eager_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from tictactoe.TicTacToeEnv import TicTacToeEnv


tf.compat.v1.enable_v2_behavior()

# ------------------------------------------------------------
checkpoint_dir = './__tf_agents__/dqn_tictactoe_lr_0.001/checkpoint'
policy_dir = './__tf_agents__/dqn_tictactoe_lr_0.001/policy'

# num_iterations = 105_000  # @param {type:"integer"}
num_iterations = 5_000  # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_capacity = 100000  # @param {type:"integer"}

# conv_layer_params = [(32, 2, 1), ]
fc_layer_params = (60, 40,)

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
gamma = 0.99
log_interval = 200  # @param {type:"integer"}

num_atoms = 51  # @param {type:"integer"}
min_q_value = -20  # @param {type:"integer"}
max_q_value = 20  # @param {type:"integer"}
n_step_update = 2  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}


def create_envs():
    # Environment
    train_py_env = TicTacToeEnv()
    eval_py_env = TicTacToeEnv()

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    return train_env, eval_env, train_py_env, eval_py_env


def update_envs_with_agent(train_py_env, eval_py_env, agent):
    # put agent to TicTaxToeEnv
    train_py_env._agent = agent
    eval_py_env._agent = agent


def create_agent(train_env, global_step_counter):
    # q network
    # categorical_q_net = \
    #     categorical_q_network.CategoricalQNetwork(train_env.observation_spec(),
    #                                               train_env.action_spec(),
    #                                               num_atoms=num_atoms,
    #                                               conv_layer_params=conv_layer_params,
    #                                               fc_layer_params=fc_layer_params)

    q_net = q_network.QNetwork(train_env.observation_spec(),
                               train_env.action_spec(),
                               #    conv_layer_params=conv_layer_params,
                               fc_layer_params=fc_layer_params)

    # optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    # Agent
    # train_step_counter = tf.compat.v2.Variable(0)

    # agent = categorical_dqn_agent.CategoricalDqnAgent(train_env.time_step_spec(),
    #                                train_env.action_spec(),
    #                                categorical_q_network=categorical_q_net,
    #                                optimizer=optimizer,
    #                                min_q_value=min_q_value,
    #                                max_q_value=max_q_value,
    #                                n_step_update=n_step_update,
    #                                td_errors_loss_fn=common.element_wise_squared_loss,
    #                                gamma=gamma,
    #                                train_step_counter=train_step_counter)

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
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


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


def create_replay_buffer(train_env, agent):
    # replay_buffer - for data collection. so that training can use the collected Trajectory
    replay_buffer = \
        tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                       batch_size=train_env.batch_size,
                                                       max_length=replay_buffer_capacity)
    # Dataset generates trajectories with shape [BxTx...] where
    # T = n_step_update + 1.
    dataset = replay_buffer.as_dataset(num_parallel_calls=3,
                                       sample_batch_size=batch_size,
                                       num_steps=n_step_update + 1)  # .prefetch(3)

    replay_buffer_itr = iter(dataset)  # <---- iterator being used in training

    return replay_buffer, replay_buffer_itr


def _train_agent(num_iterations, agent, train_env, eval_env, replay_buffer_itr, replay_buffer):
    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):
        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy, replay_buffer)

        # Sample a batch of data from the buffer and update the agent's network.
        # trajectories = replay_buffer.gather_all()
        trajectories, unused_info = next(replay_buffer_itr)
        train_loss = agent.train(trajectories)

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(
                eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1:.2f}'.format(
                step, avg_return))
            returns.append(avg_return)


def train_agent_and_save():
    """Load training checkpoint, then do training. Save at the end. """
    global_step_counter = tf.compat.v1.train.get_or_create_global_step()

    train_env, eval_env, train_py_env, eval_py_env = create_envs()
    agent = create_agent(train_env, global_step_counter)
    update_envs_with_agent(train_py_env, eval_py_env, agent)

    replay_buffer, replay_buffer_itr = create_replay_buffer(train_env, agent)

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())
    # compute_avg_return(eval_env, random_policy, num_eval_episodes)
    for _ in range(10):
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

    # training ------------------------
    _train_agent(num_iterations, agent, train_env,
                 eval_env, replay_buffer_itr, replay_buffer)

    # Save agent checkpointe ------------------------
    # save checkpoint
    train_checkpointer.save(global_step_counter)

    # save policy
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    tf_policy_saver.save(policy_dir)

    return agent, eval_env, eval_py_env


def load_policy(policy_dir):
    saved_policy = tf.compat.v2.saved_model.load(policy_dir)
    return saved_policy


def demo_game_play(agent_policy, eval_env, eval_py_env):
    # log game play
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        # level=logging.DEBUG)
                        level=logging.INFO)
    num_episodes = 5
    for _ in range(num_episodes):
        time_step = eval_env.reset()
        eval_py_env._log_on = True
        eval_py_env._exploring_opponent = False
        eval_py_env._random_start = False

        while not time_step.is_last():
            action_step = agent_policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            # print('--------', time_step)

        print('==============================')


def train_main():
    for i in range(50):
        agent, eval_env, eval_py_env = train_agent_and_save()
        demo_game_play(agent.policy, eval_env, eval_py_env)
        print(f'============{i}==============')


def demo_main():
    policy = load_policy('./__tf_agents__/dqn_tictactoe/policy')
    train_env, eval_env, train_py_env, eval_py_env = create_envs()
    demo_game_play(policy, eval_env, eval_py_env)


if __name__ == '__main__':
    train_main()
