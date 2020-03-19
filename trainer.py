import numpy as np
from utils import *
from agents.ddqn.build_graph import Q_Policy
from common.plot_func import plot
from common import logger
from agents.ddqn.replay_buffer import ReplayBuffer
import datetime


class Trainer():
    def __init__(self, env, config, dirs, in_test=True):
        self.env = env
        self.gamma = config.getfloat('MODEL_CONFIG', 'gamma')
        self.seed = config.getint('ENV_CONFIG', 'seed')
        self.in_test = in_test
        self.dirs = dirs
        num_cpu = 8
        self._init_summary()
        self.summary_writer = tf.summary.FileWriter(dirs['log'])

        self.buffer_size = int(config.getfloat('MODEL_CONFIG', 'buffer_size'))
        self.batch_size = int(config.getfloat('MODEL_CONFIG', 'batch_size'))
        self.lr_init = config.getfloat('MODEL_CONFIG', 'lr_init')
        self.reward_norm = config.getfloat('MODEL_CONFIG', 'reward_norm')
        self.reward_clip = config.getfloat('MODEL_CONFIG', 'reward_clip')

        # training config
        self.total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
        self.rendering = int(config.getfloat('TRAIN_CONFIG', 'rendering'))
        self.learning_starts = int(config.getfloat('TRAIN_CONFIG', 'learning_starts'))
        self.train_freq = int(config.getfloat('TRAIN_CONFIG', 'train_freq'))
        self.test_freq = int(config.getfloat('TRAIN_CONFIG', 'test_freq'))
        self.log_freq = int(config.getfloat('TRAIN_CONFIG', 'log_freq'))
        self.print_freq = int(config.getfloat('TRAIN_CONFIG', 'print_freq'))
        self.target_network_update_freq = int(config.getfloat('TRAIN_CONFIG', 'target_network_update_freq'))

        self.eps_init = config.getfloat('MODEL_CONFIG', 'epsilon_init')
        self.eps_decay = config.get('MODEL_CONFIG', 'epsilon_decay')
        self.eps_ratio = config.getfloat('MODEL_CONFIG', 'epsilon_ratio')
        self.eps_min = config.getfloat('MODEL_CONFIG', 'epsilon_min')

        if self.eps_decay == 'constant':
            self.eps_scheduler = Scheduler(self.eps_init, decay=self.eps_decay)
        else:
            self.eps_scheduler = Scheduler(self.eps_init, self.eps_min, self.total_step * self.eps_ratio,
                                           decay=self.eps_decay)

        self.sess = tf.get_default_session()
        tf.set_random_seed(self.seed)
        if self.sess is None:
            self.sess = make_session(config=tf.ConfigProto(allow_soft_placement=True,
                                                           inter_op_parallelism_threads=num_cpu,
                                                           intra_op_parallelism_threads=num_cpu
                                                           ), make_default=True)

    def _init_summary(self):
        self.train_reward = tf.placeholder(tf.float32, [])
        self.train_summary = tf.summary.scalar('train_reward', self.train_reward)
        self.test_reward = tf.placeholder(tf.float32, [])
        self.test_summary = tf.summary.scalar('test_reward', self.test_reward)

    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            summ = self.sess.run(self.train_summary, {self.train_reward: reward})
        else:
            summ = self.sess.run(self.test_summary, {self.test_reward: reward})
        self.summary_writer.add_summary(summ, global_step=global_step)

    def run(self):
        policy = Q_Policy(num_actions=self.env.action_space.n, num_obs=self.env.observation_space.shape[0])
        # Create all the functions necessary to train the model
        policy.build_graph(optimizer=tf.train.AdamOptimizer(learning_rate=self.lr_init),
                           gamma=self.gamma,
                           grad_norm_clipping=10
                           )

        replay_buffer = ReplayBuffer(self.buffer_size)
        # Initialize the parameters and copy them to the target network.
        self.sess.run(tf.global_variables_initializer())
        # if restore:
        #     policy.load(sess, dirs['model'], checkpoint=None)
        policy.update_target(self.sess)

        epoch_rewards = [0.0]
        eval_rewards = []
        ob_ls = []
        steps = 0  # counting the steps in one epoch
        obs = self.env.reset()

        for t in range(self.total_step):
            # Take action and update exploration to the newest value
            steps += 1
            action = policy.forward(self.sess, obs[None], self.eps_scheduler.get(1), mode='explore')
            new_obs, rew, done, _, = self.env.step(action)
            if self.rendering:
                self.env.render()
            if self.reward_norm:
                rew = rew / self.reward_norm
            if self.reward_clip:
                rew = np.clip(rew, -self.reward_clip, self.reward_clip)
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            ob_ls.append([new_obs])
            obs = new_obs
            epoch_rewards[-1] += rew  # r_sum = -3499.51

            if t > self.learning_starts and t % self.train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(self.batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
                policy.backward(self.sess, obses_t, actions, obses_tp1, dones, rewards, weights, global_step=t,
                                summary_writer=self.summary_writer)

            # Update target network periodically.
            if t > self.learning_starts and t % self.target_network_update_freq == 0:
                policy.update_target(self.sess)

            if done:
                if self.print_freq is not None and len(epoch_rewards) % self.print_freq == 0:
                    mean_100ep_reward = round(np.mean(epoch_rewards[-99:-1]), 1)
                    num_episodes = len(epoch_rewards)
                    logger.record_tabular("steps", t)
                    logger.record_tabular("episodes", num_episodes)
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("% time spent exploring", int(100 * self.eps_scheduler.get(1)))
                    logger.record_tabular("Current date and time: ", datetime.datetime.now())
                    logger.dump_tabular()

                # evaluation
                if self.in_test and len(epoch_rewards) % self.test_freq == 0:
                    episode_reward_eval = 0
                    obs_eval = self.env.reset()
                    done_eval = False
                    # print("Staring evaluating")
                    while not done_eval:
                        # Take action and update exploration to the newest value
                        action_eval = policy.forward(self.sess, obs_eval[None], 1, mode='eval')
                        new_obs_eval, rew_eval, done_eval, _ = self.env.step(action_eval)
                        obs_eval = new_obs_eval

                        if self.reward_norm:
                            rew_eval = rew_eval / self.reward_norm
                        episode_reward_eval += rew_eval
                    self._add_summary(episode_reward_eval, t, is_train=False)
                    print("evaluating reward = ", episode_reward_eval)
                    eval_rewards.append(episode_reward_eval)

                    print("Saving model...")
                    policy.save(self.sess, self.dirs['model'], len(epoch_rewards))

                if len(epoch_rewards) % self.log_freq == 0:
                    np.save(self.dirs['results'] + '{}'.format('eval_rewards'), eval_rewards)
                    np.save(self.dirs['results'] + '{}'.format('epoch_rewards'), epoch_rewards)
                    np.save(self.dirs['results'] + '{}'.format('ob_ls'), ob_ls)

                obs = self.env.reset()
                self._add_summary(epoch_rewards[-1], global_step=t)
                self.summary_writer.flush()
                epoch_rewards.append(0.0)
        print("Training Done...")
        self.env.close()
        plot(self.dirs['results'])


class Tester():
    def __init__(self, env, config, dirs, rendering=False):
        self.env = env
        self.rendering = rendering
        self.dirs = dirs

    def run(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            policy = Q_Policy(num_actions=self.env.action_space.n, num_obs=self.env.observation_space.shape[0])

            policy.load(sess, self.dirs['model'], checkpoint=700)
            print("Model loaded...")

            epoch_rewards = 0.0
            steps = 0  # counting the steps in one epoch
            obs = self.env.reset()

            while True:
                # Take action and update exploration to the newest value
                steps += 1
                action = policy.forward(sess, obs[None], 1, mode='eval')
                new_obs, rew, done, _, = self.env.step(action)
                if self.rendering:
                    self.env.render()
                obs = new_obs

                rew_eval = rew
                epoch_rewards += rew_eval
                if done:
                    print("evaluating reward = ", epoch_rewards)
                    break
