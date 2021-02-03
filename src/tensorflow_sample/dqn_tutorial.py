import base64
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay

import tensorflow as tf
import tf_agents as tfa
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


tf.compat.v1.enable_v2_behavior()

display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

# --------------------------------
num_iterations = 20_000

initial_collection_steps = 100
collect_steps_per_iteration = 1
replay_buffer_max_length = 100_000

batch_size = 64
learning_rate = 1e-3
log_interval = 200

num_eval_episodes = 10
eval_interval = 1_000

# ----------------
env_name = 'CartPole-v0'
env = suite_gym.load(env_name)


time_step = env.reset()
PIL.Image.fromarray(env.render())

print('-----------------------------------')
print('Observation Spec:', env.time_step_spec().observation)

print('Reward Spec:', env.time_step_spec().reward)

print('Action Spec:', env.action_spec())

print('-----------------------------------')
time_step = env.reset()
print('T 0, time step:', time_step)
action = np.array(0, dtype=np.int32)

next_time_step = env.step(action)
print('T 1, time step:', next_time_step)

next_time_step = env.step(action)
print('T 2, time step:', next_time_step)

print('-----------------------------------')
# 2 env, one for train, one for evaluation
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

# to tensor
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# Q value: Q(s, a) -> q-value
# Q Network - dnn gave s -> q-value[a]

fc_layer_params = (100,)
q_net = q_network.QNetwork(train_env.observation_spec(),
                           train_env.action_spec(),
                           fc_layer_params=fc_layer_params)


optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(train_env.time_step_spec(),
                           train_env.action_spec(),
                           q_network=q_net,
                           optimizer=optimizer,
                           td_errors_loss_fn=common.element_wise_squared_loss,
                           train_step_counter=train_step_counter)
agent.initialize()

print('q_net:', q_net.summary())

# ---------------------------------------------------------
# agent has 2 policy
# main policy - evaluation and deployment
# second policy - used for data collection
eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

example_environment = tf_py_environment.TFPyEnvironment(
    suite_gym.load('CartPole-v0'))

time_step = example_environment.reset()

random_policy.action(time_step)

# ---------------------------------------------------------
# Evaluation of Policy
# comput average return


def compute_avg_return(environment, policy, num_episodes=500):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return = episode_return + time_step.reward
        total_return = total_return + episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


print('-----------------------------------')
random_policy_avg_return = \
    compute_avg_return(eval_env, random_policy, num_eval_episodes)
print('random policy avg_return:', random_policy_avg_return)

# --------------------------------------------------
# Replay buffer
# data collected from the environment.
replay_buffer = \
    tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                   batch_size=train_env.batch_size,
                                                   max_length=replay_buffer_max_length)
print('-----------------------------------')
print('agent collect_data_spec:', agent.collect_data_spec)
print('agent collect_data_spec._fields:', agent.collect_data_spec._fields)

# --------------------------------------------------
# collect game play data


def collect_step(env, policy, buffer):
    time_step = env.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = env.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


collect_data(train_env, random_policy, replay_buffer, initial_collection_steps)

# --------------------------------------------------
# dataset for training
dataset = replay_buffer.as_dataset(num_parallel_calls=3,
                                   sample_batch_size=batch_size,
                                   num_steps=2).prefetch(3)
print('----------------------------')
print('dataset collected for training:', dataset)

# --------------------------------------------------
iterator = iter(dataset)
print('----------------------------')
print('dataset iterator:', iterator)

# --------------------------------------------------
# Training the Agent
# --------------------------------------------------

# wrapping code in graph using TF function
agent.train = common.function(agent.train)

# reset step
agent.train_step_counter.assign(0)
# evaluate the policy once before training
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):
    # -> replay_buffer. -> iterator(replay_buffer.as_dataset)
    collect_data(train_env,
                 agent.collect_policy,
                 replay_buffer,
                 collect_steps_per_iteration)
    experience, unused_info = next(iterator)
    # experience? trajectory is an episode. info?
    # print('-------------------------')
    # print('In training experience:', experience)
    # print('In training buffer_info:', unused_info)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    # log
    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))
    if step % eval_interval == 0:
        avg_return = compute_avg_return(
            eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

# ---------------------------------------
# see the training
iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('avg return')
plt.xlabel('iterations')
plt.ylim(top=250)
plt.show()
