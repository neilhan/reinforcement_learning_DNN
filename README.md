# Reinforcement learning with Deep Neural Network

Course: deep reinforcement learning in python by LazyProgrammer
sample code: git@github.com:lazyprogrammer/machine_learning_examples.git rl2


## OpenAI Gym tutorial

- gym
  - gym_tutorial.py - new cart
  -
- cart_pole
  - CartPole_random_search.py  - random search for policy
  - CartPole_random_search_save_video.py - create a video.
  - q_learn_bins.py - q-learning using the tabular method.
    Continuous states to discrete state, boxes.
  - q_learning.py - RBF + Q-learning on cart pole problem.
  - q_learning_tf.py - using tensorflow to implement SGDRegressor
  - policy_gradient.py - policy gradient. TD(0), actor-critic when doing TD
  - policy_learning.py - Policy gradient, and value function learning.
  - deep_q_network.py - q-learning network,
  -
- mountain_car
  Radial Basis Function, RBF activation neural network
  SGDRegressor, Stochastic Gradient Descent
  - q_learning.py - RBF network to solve mountain car
  - n_step.py - n-step q learning method. q with n steps.
  - td_lambda.py - td_lambda, with q-learning.
  - policy_gradient_hill_climb.py - continuous policy solution with policy gradient, learning is done with hill-climbing.
  - policy_gradient.py - policy gradient. TD(0), actor-critic when doing TD

- atari
  - deep_q_network.py - q-learning on game Break. CNN

# notes

```
# install python, venv on Debian
sudo apt install gcc libpq-dev -y
sudo apt install python-dev  python-pip -y
sudo apt install python3-dev python3-pip python3-venv python3-wheel -y

# opengl
sudo apt install python-opengl

# venv. ubuntu needs the without-pip
#    > sudo apt install python3-venv
# or > pip3 install virtualenv 
python -m venv env
# python -m venv env --without-pip

source env/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -r requirements.txt --upgrade

# upgrade all packages 
pip3 list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip3 install -U 

pip freeze > requirements.txt

# trouble with pyYAML, versions older than 4.x were not safe
pip3 install --ignore-installed PyYAML

python -m unittest discover -s project_directory -p "*_test.py"

# current dir ./src
cd src

# 1. run tf_agents player services, so that can train with multiple Othello Envs
python -m othello.service.tfagent_service

# 2. start 8x8 training for tf_agents DQN
python -m othello.players.tfagents_dqn.train_dqn_custom_net_6x6
python -m othello.players.tfagents_dqn.train_dqn_custom_net_8x8
# stdbuf -oL python -m othello.players.tfagents_dqn.train_dqn_custom_net_8x8 &>> ../__tf_agents__/8x8.log
# stdbuf -oL python -m othello.players.tfagents_dqn.train_dqn_custom_net_8x8 ../__tf_agents__/8x8.log 2>&1

# logrotate 
logrotate log-truncate.cfg

``` 

# Nix notes
```
nix-env -iA lorri
nix-env -q
lorri init
nix-shell shell.nix
```
# Purescript notes
```
spago repl
```
``` 
git@github.com:rmsander/marl_ppo.git
docker run -it --gpus 1 tensorflow/tensorflow:latest-gpu-jupyter
docker run -it --gpus all tensorflow/tensorflow:latest-gpu-jupyter
```
# othello
20210113 started
- learning rate matters. 
- shortcut network helps agent to start learning. 
- wide network, deep network get into local optimal and trapped in local optimal easily.
- start simple, debug env, network, agent, then can train one, then parallel train.
- tf_agents is a good tool to work with RL
20210209 8 parallel env training with tf_agents.
