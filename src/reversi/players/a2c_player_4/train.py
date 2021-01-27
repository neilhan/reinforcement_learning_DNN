from typing import Any, List, Sequence, Tuple
import logging
from datetime import datetime

import numpy as np
import tensorflow as tf

from reversi.tf_utils import set_global_seeds
from reversi.players.a2c_player_4.A2CModel import A2CModel
from reversi.players.a2c_player_4.A2CTrainer import A2CTrainer
from reversi.players.a2c_player_4.GameWrapper import GameWrapperInpatient


def do_training(board_size=8,
                max_episodes=10_000, optimizer_learn_rate=0.001,
                model_save_path='./__models__/a2c_player_4/',
                tensorboard_path='./__models__/a2c_player_4_tensorboard/',
                load_saved_model=False):
    set_global_seeds(0)

    model = A2CModel(num_actions=board_size * board_size + 1)

    # Create the environment
    env = GameWrapperInpatient(board_size=board_size)

    trainer = A2CTrainer(env, model,
                         optimizer_learn_rate=optimizer_learn_rate,
                         model_save_path=model_save_path,
                         tensorboard_path=tensorboard_path,
                         load_saved_model=load_saved_model)

    trainer.train(max_episodes)

    model.save('./__models__/player_4/')

    return model


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        # level=logging.DEBUG)
                        level=logging.INFO)
    tensorboard_path = './__models__/a2c_player_4_tensorboard/'

    # OPTIONAL: ADD A TIMESTAMP FOR UNIQUE FOLDER
    timestamp = datetime.now().strftime("%Y-%m-%d--%H%M")
    tensorboard_path = tensorboard_path + '-' + timestamp

    model = do_training(max_episodes=1_000_000, board_size=6,
                        optimizer_learn_rate=0.0010,
                        model_save_path='./__models__/a2c_player_4/',
                        tensorboard_path=tensorboard_path,
                        load_saved_model=True)
