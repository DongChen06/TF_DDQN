"""
    Ford_proj main.py
"""
from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gym
import datetime
from common import logger
from agents.deepq.build_graph import Q_Policy
from common.plot_func import plot

from agents.deepq.replay_buffer import ReplayBuffer
import argparse
import configparser
from utils import *


def parse_args():
    default_base_dir = 'Data'
    # default_config_dir = 'config/config_ford.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--environment', type=str, required=False,
                        default='ford', help="env")
    # parser.add_argument('--config-dir', type=str, required=False,
    #                     default=default_config_dir, help="experiment config dir")
    parser.add_argument('--is_training', type=str, required=False,
                        default=not False, help="True=train, False=evaluation")
    parser.add_argument('--test-mode', type=str, required=False,
                        default='in_train_test',
                        help="test mode during training",
                        choices=['no_test', 'in_train_test', 'after_train_test', 'all_test'])

    args = parser.parse_args()
    return args


def train_fn(args):
    base_dir = args.base_dir
    dirs = init_dir(base_dir)
    init_log(dirs['log'])
    environment = args.environment
    if environment is 'ford':
        config_dir = 'config/config_ford.ini'
    else:
        config_dir = 'config/config_gym.ini'
    copy_file(config_dir, dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_dir)

    # test during training or test after training
    in_test, post_test = init_test_flag(args.test_mode)

    gamma = config.getfloat('MODEL_CONFIG', 'gamma')
    buffer_size = int(config.getfloat('MODEL_CONFIG', 'buffer_size'))
    batch_size = int(config.getfloat('MODEL_CONFIG', 'batch_size'))
    lr_init = config.getfloat('MODEL_CONFIG', 'lr_init')
    reward_norm = config.getfloat('MODEL_CONFIG', 'reward_norm')
    reward_clip = config.getfloat('MODEL_CONFIG', 'reward_clip')

    # training config
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    rendering = int(config.getfloat('TRAIN_CONFIG', 'rendering'))
    learning_starts = int(config.getfloat('TRAIN_CONFIG', 'learning_starts'))
    train_freq = int(config.getfloat('TRAIN_CONFIG', 'train_freq'))
    test_freq = int(config.getfloat('TRAIN_CONFIG', 'test_freq'))
    log_freq = int(config.getfloat('TRAIN_CONFIG', 'log_freq'))
    print_freq = int(config.getfloat('TRAIN_CONFIG', 'print_freq'))
    target_network_update_freq = int(config.getfloat('TRAIN_CONFIG', 'target_network_update_freq'))

    eps_init = config.getfloat('MODEL_CONFIG', 'epsilon_init')
    eps_decay = config.get('MODEL_CONFIG', 'epsilon_decay')
    eps_ratio = config.getfloat('MODEL_CONFIG', 'epsilon_ratio')
    eps_min = config.getfloat('MODEL_CONFIG', 'epsilon_min')
    seed = config.getint('ENV_CONFIG', 'seed')

    if eps_decay == 'constant':
        eps_scheduler = Scheduler(eps_init, decay=eps_decay)
    else:
        eps_scheduler = Scheduler(eps_init, eps_min, total_step * eps_ratio,
                                  decay=eps_decay)

    # Initialize environment
    print("Initializing environment")
    if environment is 'ford':
        env = FordEnv(config['ENV_CONFIG'], rendering=rendering)
    else:
        env = gym.make("CartPole-v0")

    config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth = True
    sess = tf.get_default_session()
    tf.set_random_seed(seed)
    if sess is None:
        sess = make_session(config=config, make_default=True)

    try:
        policy = Q_Policy(num_actions=env.action_space.n, num_obs=env.observation_space.shape[0])
        # Create all the functions necessary to train the model
        train, update_target, debug = policy.build_graph(
            optimizer=tf.train.AdamOptimizer(learning_rate=lr_init),
            gamma=gamma,
            grad_norm_clipping=10
        )

        replay_buffer = ReplayBuffer(buffer_size)
        # Initialize the parameters and copy them to the target network.
        sess.run(tf.global_variables_initializer())
        # if restore:
        #     policy.load(sess, dirs['model'], checkpoint=None)
        update_target()

        epoch_rewards = [0.0]
        eval_rewards = []
        ob_ls = []
        steps = 0  # counting the steps in one epoch
        obs = env.reset()

        for t in range(total_step):
            # Take action and update exploration to the newest value
            steps += 1
            action = policy.forward(sess, obs[None], eps_scheduler.get(1), mode='explore')
            new_obs, rew, done, _, = env.step(action)
            if rendering:
                env.render()
            if reward_norm:
                rew = rew / reward_norm
            if reward_clip:
                rew = np.clip(rew, -reward_clip, reward_clip)
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            ob_ls.append([new_obs])
            obs = new_obs
            epoch_rewards[-1] += rew  # r_sum = -3499.51

            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
                train(obses_t, actions, rewards, obses_tp1, dones, weights)

            # Update target network periodically.
            if t > learning_starts and t % target_network_update_freq == 0:
                update_target()

            if done:
                if print_freq is not None and len(epoch_rewards) % print_freq == 0:
                    mean_100ep_reward = round(np.mean(epoch_rewards[-11:-1]), 1)
                    num_episodes = len(epoch_rewards)
                    logger.record_tabular("steps", t)
                    logger.record_tabular("episodes", num_episodes)
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("% time spent exploring", int(100 * eps_scheduler.get(1)))
                    logger.record_tabular("Current date and time: ", datetime.datetime.now())
                    logger.dump_tabular()

                # evaluation
                if in_test and len(epoch_rewards) % test_freq == 0:
                    episode_reward_eval = 0
                    obs_eval = env.reset()
                    done_eval = False
                    # print("Staring evaluating")
                    while not done_eval:
                        # Take action and update exploration to the newest value
                        action_eval = policy.forward(sess, obs_eval[None], 1, mode='eval')
                        new_obs_eval, rew_eval, done_eval, _ = env.step(action_eval)
                        obs_eval = new_obs_eval

                        if reward_norm:
                            rew_eval = rew_eval / reward_norm
                        episode_reward_eval += rew_eval

                    print("evaluating reward = ", episode_reward_eval)
                    eval_rewards.append(episode_reward_eval)

                    print("Saving model...")
                    policy.save(sess, dirs['model'], len(epoch_rewards))

                if len(epoch_rewards) % log_freq == 0:
                    np.save(dirs['results'] + '{}'.format('eval_rewards'), eval_rewards)
                    np.save(dirs['results'] + '{}'.format('epoch_rewards'), epoch_rewards)
                    np.save(dirs['results'] + '{}'.format('ob_ls'), ob_ls)

                obs = env.reset()
                epoch_rewards.append(0.0)

        env.close()
        plot(dirs['results'])

        if post_test:
            evaluate(args)

    except Exception as e:
        print("Done...")
        env.close()
        raise e


def evaluate(args):
    base_dir = args.base_dir
    dirs = init_dir(base_dir)
    environment = args.environment
    if environment is 'ford':
        config_dir = 'config/config_ford.ini'
    else:
        config_dir = 'config/config_gym.ini'
    copy_file(config_dir, dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_dir)
    rendering = False
    # Initialize environment
    print("Initializing environment")
    if environment is 'ford':
        env = FordEnv(config['ENV_CONFIG'], rendering=False)
    else:
        env = gym.make("CartPole-v0")

    try:
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            policy = Q_Policy(num_actions=env.action_space.n, num_obs=env.observation_space.shape[0])

            policy.load(sess, dirs['model'], checkpoint=10)
            print("Model loaded...")

            epoch_rewards = 0.0
            steps = 0  # counting the steps in one epoch
            reward_norm = 7314.15
            obs = env.reset()

            while True:
                # Take action and update exploration to the newest value
                steps += 1
                action = policy.forward(sess, obs[None], 1, mode='eval')
                new_obs, rew, done, _, = env.step(action)
                if rendering:
                    env.render()
                obs = new_obs

                if reward_norm:
                    rew_eval = rew / reward_norm
                epoch_rewards += rew_eval
                if done:
                    print("evaluating reward = ", epoch_rewards)
                    break

        print("Evaluation Done...")
        env.close()

    except Exception as e:
        env.close()
        raise e


if __name__ == '__main__':
    args = parse_args()
    if args.is_training is True:
        train_fn(args)
    else:
        evaluate(args)
