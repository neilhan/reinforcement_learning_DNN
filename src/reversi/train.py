import argparse
import os
import logging
import multiprocessing

MODEL_PATH = '__models__'

NUM_CPU = multiprocessing.cpu_count()


def train(num_timesteps: int, num_cup: int):
    pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--steps', help='training steps', type=int, default=int(80e3))
    parser.add_argument('--nrenv', help='Number of training environments', type=int, default=int(1))

    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(MODEL_PATH, exist_ok=True)
    train(args.steps, 1)


if __name__ == '__main__':
    main()
