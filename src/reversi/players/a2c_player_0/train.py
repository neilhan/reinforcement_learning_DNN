import os
import time
import tensorflow as tf

from reversi.tf_utils import set_global_seeds 
from reversi.players.a2c_player_0.A2CAgentCNN import VisionShape, A2CAgentCNN
from reversi.players.a2c_player_0.A2CAgentV0 import A2CAgentV0
from reversi.players.a2c_player_0.A2CAgentV0 import A2CAgentV0Trainer
from reversi.players.a2c_player_0.GameWrapper import GameWrapper


def learn(nn_class,
          game,
          seed=0,
          num_steps=5,
          total_timesteps=int(80e6),
          vision_shape: VisionShape = VisionShape(8, 8, 1),
          vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4,
          epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=1000):
    set_global_seeds(seed)

    # save_name = os.path.join('models', env_id + '.save')
    vision_shape = game.get_vision_shape()
    action_size = game.get_action_size()
    agent = A2CAgentV0(nn_class=nn_class,
                       vision_shape=vision_shape,
                       action_size=action_size,
                       num_steps=num_steps,
                       ent_coef=ent_coef, vf_coef=vf_coef,
                       max_grad_norm=max_grad_norm,
                       lr=lr, optimizer_alpha=alpha, optimizer_epsilon=epsilon)
    # if os.path.exists(save_name):
    #     agent.load(save_name)
    trainer = A2CAgentV0Trainer(game, agent, num_steps=num_steps, gamma=gamma)

    tstart = time.time()
    for b in range(1, total_timesteps // num_steps + 1):
        states, rewards, actions, values = trainer.get_a_training_batch()
        policy_loss, value_loss, policy_entropy = \
            agent.train(states, rewards, actions, values)
        time_cost_seconds = time.time() - tstart
        fps = int((b * num_steps) / time_cost_seconds)
        if b % log_interval == 0 or b == 1:
            print(' - - - - - - - ')
            print("nupdates", b)
            print("total_timesteps", b * num_steps)
            print("fps", fps)
            print("policy_entropy", float(policy_entropy))
            print("value_loss", float(value_loss))

            # total reward
            r = trainer.total_rewards[-100:]  # get last 100
            tr = trainer.real_total_rewards[-100:]
            if len(r) == 100:
                print("avg reward (last 100):", np.mean(r))
            if len(tr) == 100:
                print("avg total reward (last 100):", np.mean(tr))
                print("max (last 100):", np.max(tr))

            # agent.save(save_name)

    # game.close()
    # agent.save(save_name)


if __name__ == '__main__':
    # make sure using CPU. this docker doesn't have GPU.
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # grads = tf.gradients(loss_fn, params) needs disable_eager_execution
    tf.compat.v1.disable_eager_execution()

    board_size = 6
    game = GameWrapper(1, board_size)

    learn(A2CAgentCNN,
          game,
          total_timesteps=100,
          vision_shape=game.get_vision_shape())

    # can we train now?
    print('legacy version built.')
    agent.session.close()
