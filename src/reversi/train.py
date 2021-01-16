import argparse
import os
import logging
import multiprocessing

import reversi.GameBoard as GameBoard

MODEL_PATH = '__models__'

NUM_CPU = multiprocessing.cpu_count()


def train(num_timesteps: int, num_cup: int):

    pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--steps', help='training steps', type=int, default=int(80e3))
    parser.add_argument('--numenv', help='Number of training environments', type=int, default=int(1))

    return parser.parse_args()


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    args = get_args()
    os.makedirs(MODEL_PATH, exist_ok=True)
    train(args.steps, 1)
    logging.debug('this is debug loggoing')
    logging.info('this is info loggoing')
    logging.warning('this is warning loggoing')
    logging.error('this is error loggoing')


if __name__ == '__main__':
    main()
