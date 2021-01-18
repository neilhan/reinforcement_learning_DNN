import argparse
import os
import numpy as np
import multiprocessing
import logging

from reversi.GameWrapper import GameWrapper


NUM_CPU = multiprocessing.cpu_count()
MODEL_PATH = '__models__'

def train(num_steps:int, num_works:int):
    envs = [GameWrapper(i) for i in range(num_works)]

    # ???? train


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-s', '--steps', help='Training for given number of steps. Default 10,000',
        type=int, default=int(10e3))
    parser.add_argument(
        '--num_env', help='Number of game training in parralel, default to the number of CPUs of thi current machine.',
        type=int, default=int(1))

    return parser.parse_args()


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)

    args = get_args()
    os.makedirs(MODEL_PATH, exist_ok=True)
    train(args.steps, 1)


if __name__ == '__main__':
    main()
