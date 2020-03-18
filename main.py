"""
    main.py
"""
from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gym
import argparse
import configparser
from utils import *
from trainer import Trainer, Tester


def parse_args():
    default_base_dir = 'Data'
    # default_config_dir = 'config/config_ford.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--environment', type=str, required=False,
                        default='gym', help="env")
    # parser.add_argument('--config-dir', type=str, required=False,
    #                     default=default_config_dir, help="experiment config dir")
    parser.add_argument('--is_training', type=str, required=False,
                        default=False, help="True=train, False=evaluation")
    parser.add_argument('--test-mode', type=str, required=False,
                        default='all_test',
                        help="test mode during training",
                        choices=['no_test', 'in_train_test', 'after_train_test', 'all_test'])

    args = parser.parse_args()
    return args


def train_fn(args):
    base_dir = args.base_dir
    dirs = init_dir(base_dir)
    init_log(dirs['log'])
    config_dir = 'config/config_gym.ini'
    copy_file(config_dir, dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_dir)

    # test during training or test after training
    in_test, post_test = init_test_flag(args.test_mode)
    # Initialize environment
    print("Initializing environment")
    env = gym.make("CartPole-v0")

    trainer = Trainer(env, config, dirs, in_test)
    trainer.run()


def evaluate_fn(args):
    base_dir = args.base_dir
    dirs = init_dir(base_dir)
    config_dir = 'config/config_gym.ini'

    copy_file(config_dir, dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_dir)
    rendering = False

    print("Initializing environment")
    env = gym.make("CartPole-v0")
    tester = Tester(env, config, dirs, rendering=rendering)
    tester.run()
    print("Evaluation Done...")
    env.close()


if __name__ == '__main__':
    args = parse_args()
    if args.is_training is True:
        train_fn(args)
    else:
        evaluate_fn(args)
