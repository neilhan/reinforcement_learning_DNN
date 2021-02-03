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

# venv. For ubuntu, debian needs the without-pip
python -m venv env --without-pip
source env/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -r requirements.txt --upgrade

pip freeze > requirements.txt

# trouble with pyYAML, versions older than 4.x were not safe
pip3 install --ignore-installed PyYAML

python -m unittest discover -s project_directory -p "*_test.py"
```

# othello
20210113

# notes
```
git@github.com:rmsander/marl_ppo.git
docker run -it --gpus 1 tensorflow/tensorflow:latest-gpu-jupyter
docker run -it --gpus all tensorflow/tensorflow:latest-gpu-jupyter