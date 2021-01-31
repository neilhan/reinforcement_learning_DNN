from typing import Any, List, Sequence, Tuple
import logging
from datetime import datetime

import numpy as np
import tensorflow as tf

from reversi.tf_utils import set_global_seeds
from reversi.players.a2c_player_4.A2CModel6x6 import A2CModel
from reversi.players.a2c_player_4.A2CTrainer import A2CTrainer
from reversi.players.a2c_player_4.GameWrapper import GameWrapperInpatient


def do_training(board_size=8,
                max_episodes=10_000,
                optimizer_learn_rate=0.001,
                model_save_path='./__models__/a2c_player_4/',
                tensorboard_path='./__models__/a2c_player_4_tensorboard/',
                load_saved_model=False,
                game_reset_random=True):
    set_global_seeds(0)

    model = A2CModel(vision_shape=(board_size, board_size, 1),
                     num_actions=board_size * board_size + 1,
                     tensorboard_path=tensorboard_path)
    if load_saved_model:
        logging.info('Loading model...')
        model.load_model(model_save_path)
        logging.info('Model loaded from: ' + model_save_path)

    # Create the environment
    env = GameWrapperInpatient(board_size=board_size)

    trainer = A2CTrainer(env, model,
                         optimizer_learn_rate=optimizer_learn_rate,
                         model_save_path=model_save_path,
                         game_reset_random=game_reset_random)

    trainer.train(max_episodes)

    model.save_model(model_save_path)

    return model


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        # level=logging.DEBUG)
                        level=logging.INFO)

    # OPTIONAL: ADD A TIMESTAMP FOR UNIQUE FOLDER
    timestamp = datetime.now().strftime("%Y-%m-%d--%H%M")

    model = do_training(max_episodes=1_000_000, board_size=6,
                        optimizer_learn_rate=1.0e-6,
                        model_save_path='./__models__/a2c_player_4_6x6/',
                        tensorboard_path='./__models__/a2c_player_4_6x6_drop_tensorboard/' + '-' + timestamp,
                        load_saved_model=True,
                        game_reset_random=False)
