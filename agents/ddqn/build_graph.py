import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
import os
import logging
from agents.ddqn.utils import huber_loss


def q_func(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


class Q_Policy():
    def __init__(self, num_actions, num_obs, double_q=True, scope="ddqn", reuse=None):
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.scope = scope
        self.double_q = double_q
        self.reuse = reuse
        self.act_fn = self.build_act()
        self.saver = tf.train.Saver(max_to_keep=10)

    def build_act(self, reuse=None):
        with tf.variable_scope(self.scope, reuse=reuse):
            self.observations_ph = tf.placeholder(tf.float32, [None, self.num_obs])
            q_values = q_func(self.observations_ph, self.num_actions, scope="q_func")
            return q_values

    def build_graph(self, optimizer, grad_norm_clipping=None, gamma=1.0):
        with tf.variable_scope(self.scope, reuse=None):
            # set up placeholders
            self.obs_t_input = tf.placeholder(tf.float32, [None, self.num_obs])
            self.act_t_ph = tf.placeholder(tf.int32, [None], name="action")
            self.rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
            self.obs_tp1_input = tf.placeholder(tf.float32, [None, self.num_obs])
            self.done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
            self.importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

            # q network evaluation
            q_t = q_func(self.obs_t_input, self.num_actions, scope="q_func", reuse=True)  # reuse parameters from act
            q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope=tf.get_variable_scope().name + "/q_func")

            # target q network evaluation
            q_tp1 = q_func(self.obs_tp1_input, self.num_actions, scope="target_q_func")
            target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                   scope=tf.get_variable_scope().name + "/target_q_func")

            # q scores for actions which we know were selected in the given state.
            q_t_selected = tf.reduce_sum(q_t * tf.one_hot(self.act_t_ph, self.num_actions), 1)

            # compute estimate of best possible value starting from state at t + 1
            if self.double_q:
                q_tp1_using_online_net = q_func(self.obs_tp1_input, self.num_actions, scope="q_func", reuse=True)
                q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
                q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, self.num_actions), 1)
            else:
                q_tp1_best = tf.reduce_max(q_tp1, 1)
            q_tp1_best_masked = (1.0 - self.done_mask_ph) * q_tp1_best

            # compute RHS of bellman equation
            q_t_selected_target = self.rew_t_ph + gamma * q_tp1_best_masked

            # compute the error (potentially clipped)
            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
            loss = huber_loss(td_error)
            weighted_error = tf.reduce_mean(self.importance_weights_ph * loss)

            if grad_norm_clipping is not None:
                gradients = tf.gradients(weighted_error, q_func_vars)
                grads, grad_norm = tf.clip_by_global_norm(gradients, grad_norm_clipping)
                self.optimize_expr = optimizer.apply_gradients(list(zip(grads, q_func_vars)))
            else:
                self.optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

            # update_target_fn will be called periodically to copy Q network to target Q network
            self.update_target_expr = []
            for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                       sorted(target_q_func_vars, key=lambda v: v.name)):
                self.update_target_expr.append(var_target.assign(var))
            self.update_target_expr = tf.group(*self.update_target_expr)

            # monitor training
            summaries = []
            summaries.append(tf.summary.scalar('train/%s_loss' % self.scope, weighted_error))
            summaries.append(tf.summary.scalar('train/%s_q' % self.scope, tf.reduce_mean(q_t_selected)))
            summaries.append(tf.summary.scalar('train/%s_tq' % self.scope, tf.reduce_mean(q_tp1)))
            summaries.append(tf.summary.scalar('train/%s_gradnorm' % self.scope, grad_norm))
            self.summary = tf.summary.merge(summaries)

    def forward(self, sess, obs, eps, mode='explore'):
        qs = sess.run(self.act_fn, {self.observations_ph: obs})
        if (mode == 'explore') and (np.random.random() < eps):
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(qs)
        return action

    def backward(self, sess, obs, acts, next_obs, dones, rs, weights, global_step,
                 summary_writer=None):
        if summary_writer is None:
            ops = self.optimize_expr
        else:
            ops = [self.summary, self.optimize_expr]
        outs = sess.run(ops,
                        {self.obs_t_input: obs,
                         self.act_t_ph: acts,
                         self.obs_tp1_input: next_obs,
                         self.done_mask_ph: dones,
                         self.rew_t_ph: rs,
                         self.importance_weights_ph: weights})
        if summary_writer is not None:
            summary_writer.add_summary(outs[0], global_step=global_step)

    def update_target(self, sess):
        sess.run(self.update_target_expr)

    def save(self, sess, model_dir, global_step):
        self.saver.save(sess, model_dir + 'checkpoint', global_step=global_step)

    def load(self, sess, model_dir, checkpoint=None):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if checkpoint is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        prefix = file.split('.')[0]
                        tokens = prefix.split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = prefix
                            save_step = cur_step
            else:
                save_file = 'checkpoint-' + str(int(checkpoint))
        if save_file is not None:
            self.saver.restore(sess, model_dir + save_file)
            logging.info('Checkpoint loaded: %s' % save_file)
            return True
        logging.error('Can not find old checkpoint for %s' % model_dir)
        return False
