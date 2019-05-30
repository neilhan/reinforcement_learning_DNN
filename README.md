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
  - policy_learning.py - Policy learning, and value function learning.
- mountain_car
  Radial Basis Function, RBF activation neural network
  SGDRegressor, Stochastic Gradient Descent
  - q_learning.py - RBF network to solve mountain car
  - n_step.py - n-step q learning method. q with n steps.
  - td_lambda.py - td_lambda, with q-learning.

# notes

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt

pip freeze > requirements.txt

# trouble with pyYAML, versions older than 4.x were not safe
pip3 install --ignore-installed PyYAML
```
