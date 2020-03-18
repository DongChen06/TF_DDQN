import tensorflow as tf
import common.tf_util as U
import numpy as np
import tensorflow.contrib.layers as layers
import os
import logging


def q_func(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        # out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


class Q_Policy:
    def __init__(self, num_actions, num_obs, double_q=True, scope="deepq", reuse=None):
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.scope = scope
        self.double_q = double_q
        self.reuse = reuse
        self.act_fn = self.build_act()
        self.saver = tf.train.Saver(max_to_keep=5)

    def build_act(self, reuse=None):
        with tf.variable_scope(self.scope, reuse=reuse):
            self.observations_ph = tf.placeholder(tf.float32, [None, self.num_obs])
            q_values = q_func(self.observations_ph, self.num_actions, scope="q_func")
            return q_values

    def build_graph(self, optimizer, grad_norm_clipping=None, gamma=1.0):
        with tf.variable_scope(self.scope, reuse=None):
            # set up placeholders
            obs_t_input = tf.placeholder(tf.float32, [None, self.num_obs])
            act_t_ph = tf.placeholder(tf.int32, [None], name="action")
            rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
            obs_tp1_input = tf.placeholder(tf.float32, [None, self.num_obs])
            done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
            importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

            # q network evaluation
            q_t = q_func(obs_t_input, self.num_actions, scope="q_func", reuse=True)  # reuse parameters from act
            q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope=tf.get_variable_scope().name + "/q_func")

            # target q network evaluation
            q_tp1 = q_func(obs_tp1_input, self.num_actions, scope="target_q_func")
            target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                   scope=tf.get_variable_scope().name + "/target_q_func")

            # q scores for actions which we know were selected in the given state.
            q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, self.num_actions), 1)

            # compute estimate of best possible value starting from state at t + 1
            if self.double_q:
                q_tp1_using_online_net = q_func(obs_tp1_input, self.num_actions, scope="q_func", reuse=True)
                q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
                q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, self.num_actions), 1)
            else:
                q_tp1_best = tf.reduce_max(q_tp1, 1)
            q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

            # compute RHS of bellman equation
            q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

            # compute the error (potentially clipped)
            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
            loss = U.huber_loss(td_error)
            weighted_error = tf.reduce_mean(importance_weights_ph * loss)

            # compute optimization op (potentially with gradient clipping)
            if grad_norm_clipping is not None:
                gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars)
                for i, (grad, var) in enumerate(gradients):
                    if grad is not None:
                        gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
                optimize_expr = optimizer.apply_gradients(gradients)
            else:
                optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

            # update_target_fn will be called periodically to copy Q network to target Q network
            update_target_expr = []
            for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                       sorted(target_q_func_vars, key=lambda v: v.name)):
                update_target_expr.append(var_target.assign(var))
            update_target_expr = tf.group(*update_target_expr)

            # Create callable functions
            train = U.function(
                inputs=[
                    obs_t_input,
                    act_t_ph,
                    rew_t_ph,
                    obs_tp1_input,
                    done_mask_ph,
                    importance_weights_ph
                ],
                outputs=td_error,
                updates=[optimize_expr]
            )
            update_target = U.function([], [], updates=[update_target_expr])

            q_values = U.function([obs_t_input], q_t)
            return train, update_target, {'q_values': q_values}

    def forward(self, sess, obs, eps, mode='explore'):
        qs = sess.run(self.act_fn, {self.observations_ph: obs})
        if (mode == 'explore') and (np.random.random() < eps):
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(qs)
        return action

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
