import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()
from tf_utils import flatten, fc, complex_multiply, edge_gnn_layer_4, complex_H, complex_modulus, complex_modulus_all


class EdgeGNN_Q(object):
    def __init__(self, params, *args):
        self.num_hidden = params['num_hidden']
        self.num_hidden_lamda = params['num_hidden_lamda']
        self.lr = params['lr']
        self.lr_lamda = params['lr_lamda']
        self.bn = params['bn']
        self.FNN_type = params['FNN_type']
        self.process_norm = params['process_norm']
        self.initial_1 = params['initial_1']
        self.initial_2 = params['initial_2']
        self.FNN_hidden = params['FNN_hidden']
        self.bs_num_an = params['bs_num_an']
        self.bs_num_rf = params['bs_num_rf']
        self.num_users = params['num_users']
        self.power = params['power']
        self.sigma_dB = params['sigma_dB']
        self.sigma = 10.0 ** (-self.sigma_dB / 10.0)

        self.r_min = params['r_min']

        self.hidden_activation = params['hidden_activation']

        self.H = tf.placeholder(tf.float32, [None, self.num_users, self.bs_num_an, 2])
        self.batch = tf.shape(self.H)[0]
        self.train_ph = tf.placeholder(tf.bool, shape=())

        self.F_rf, self.F_bb, self.edge_frf_f, self.edge_fbb_f = self.build_network()
        self.lamda = self.build_network_lamda()
        self.Rate, self.Rate_user = self.cal_rate()
        self.Rate_user = tf.reshape(self.Rate_user, [self.batch, self.num_users])

        self.optimizer_policy = tf.train.AdamOptimizer(self.lr)
        self.optimizer_lamda = tf.train.AdamOptimizer(self.lr_lamda)

        self.cost = tf.reduce_mean(-1. * self.Rate + tf.reduce_sum(tf.nn.sigmoid(self.lamda) *
                                                                   tf.nn.relu(self.r_min - self.Rate_user),
                                                                   axis=1)
                                   )

        self.var_policy = [var for var in tf.trainable_variables() if 'policy' in var.name]
        self.grads_policy = self.optimizer_policy.compute_gradients(self.cost, self.var_policy)
        self.train_policy = self.optimizer_policy.apply_gradients(self.grads_policy)

        self.var_lamda = [var for var in tf.trainable_variables() if 'lamda' in var.name]
        self.grads_lamda = self.optimizer_lamda.compute_gradients(-1. * self.cost, self.var_lamda)
        self.train_lamda = self.optimizer_lamda.apply_gradients(self.grads_lamda)

        self.loss = - tf.reduce_mean(self.Rate)

        # self.loss = - tf.reduce_mean(self.Rate)
        # self.optimizer = tf.train.AdamOptimizer(self.lr)
        # self.optimize = self.optimizer.minimize(self.loss)

        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def build_network(self):
        edge_h_f = self.H
        edge_frf_f = tf.placeholder(tf.float32, [None, self.bs_num_an, self.bs_num_rf, 2])
        edge_fbb_f = tf.placeholder(tf.float32, [None, self.num_users, self.bs_num_rf, 2])

        edge_h, edge_frf, edge_fbb = edge_h_f, edge_frf_f, edge_fbb_f

        for l, h in enumerate(self.num_hidden[:]):
            edge_h, edge_frf, edge_fbb = edge_gnn_layer_4(edge_h, edge_frf, edge_fbb, self.H, self.FNN_hidden,
                                                          self.bs_num_an, self.bs_num_rf, self.num_users, self.bn,
                                                          self.train_ph, h, self.process_norm, self.FNN_type,
                                                          self.hidden_activation, scope="policy_" + str(l))
            if self.bn:
                edge_h = tf.layers.batch_normalization(edge_h, training=self.train_ph, reuse=False,
                                                       name='policy_bn_2' + str(l))
                edge_frf = tf.layers.batch_normalization(edge_frf, training=self.train_ph, reuse=False,
                                                         name='policy_bn_3' + str(l))
                edge_fbb = tf.layers.batch_normalization(edge_fbb, training=self.train_ph, reuse=False,
                                                         name='policy_bn_4' + str(l))
            edge_h = self.hidden_activation(edge_h)
            edge_frf = self.hidden_activation(edge_frf)
            edge_fbb = self.hidden_activation(edge_fbb)
        # output layer
        edge_h, edge_frf, edge_fbb = edge_gnn_layer_4(edge_h, edge_frf, edge_fbb, self.H, self.FNN_hidden,
                                                      self.bs_num_an, self.bs_num_rf, self.num_users, self.bn,
                                                      self.train_ph, 2, self.process_norm, self.FNN_type,
                                                      self.hidden_activation, scope="policy_gnn_out")
        edge_fbb = tf.reshape(edge_fbb, [self.batch, 1, self.num_users * self.bs_num_rf, 2])

        # output
        F_bb = edge_fbb
        F_rf = edge_frf

        F_rf = tf.div(F_rf, tf.tile(tf.sqrt(tf.reduce_sum(tf.square(F_rf), axis=3, keepdims=True)),
                                    [1, 1, 1, 2])) * 1 / np.sqrt(self.bs_num_an)
        F_bb_r = tf.reshape(F_bb, [self.batch, self.num_users, self.bs_num_rf, 2])
        norm = tf.sqrt(complex_modulus_all(complex_multiply(F_bb_r, tf.transpose(F_rf, [0, 2, 1, 3]))))
        F_bb = np.sqrt(self.num_users) * tf.div(F_bb, tf.tile(norm, [1, 1, self.num_users * self.bs_num_rf, 1]))

        return F_rf, F_bb, edge_frf_f, edge_fbb_f

    def build_network_lamda(self):
        edge_h_f = self.H
        edge_frf_f = self.edge_frf_f
        edge_fbb_f = self.edge_fbb_f

        edge_h, edge_frf, edge_fbb = edge_h_f, edge_frf_f, edge_fbb_f

        for l, h in enumerate(self.num_hidden_lamda[:]):
            edge_h, edge_frf, edge_fbb = edge_gnn_layer_4(edge_h, edge_frf, edge_fbb, self.H, self.FNN_hidden,
                                                          self.bs_num_an, self.bs_num_rf, self.num_users, self.bn,
                                                          self.train_ph, h, self.process_norm, self.FNN_type,
                                                          self.hidden_activation, scope="lamda_gnn" + str(l))
            if self.bn:
                edge_h = tf.layers.batch_normalization(edge_h, training=self.train_ph, reuse=False,
                                                       name='lamda_bn_2' + str(l))
                edge_frf = tf.layers.batch_normalization(edge_frf, training=self.train_ph, reuse=False,
                                                         name='lamda_bn_3' + str(l))
                edge_fbb = tf.layers.batch_normalization(edge_fbb, training=self.train_ph, reuse=False,
                                                         name='lamda_bn_4' + str(l))
            edge_h = self.hidden_activation(edge_h)
            edge_frf = self.hidden_activation(edge_frf)
            edge_fbb = self.hidden_activation(edge_fbb)
        # output layer
        edge_h, edge_frf, edge_fbb = edge_gnn_layer_4(edge_h, edge_frf, edge_fbb, self.H, self.FNN_hidden,
                                                      self.bs_num_an, self.bs_num_rf, self.num_users, self.bn,
                                                      self.train_ph, 1, self.process_norm, self.FNN_type,
                                                      self.hidden_activation, scope="lamda_gnn_out")

        lamda = tf.reduce_mean(edge_h, axis=[2, 3])

        return lamda

    def cal_rate(self):
        Rate_user = []
        Rate = 0
        for i in range(self.num_users):
            signal = complex_multiply(complex_multiply(self.H[:, i:(i + 1), :, :], self.F_rf),
                                      tf.transpose(self.F_bb[:, :, i * self.bs_num_rf:(i + 1) * self.bs_num_rf, :],
                                                   [0, 2, 1, 3]))
            signal_module = (self.power / self.num_users) * tf.reduce_sum(complex_modulus(signal), axis=[1, 2, 3])
            inf = 0
            for n in range(self.num_users):
                if n == i:
                    inf = inf + 0
                else:
                    inf_n = complex_multiply(complex_multiply(self.H[:, i:(i + 1), :, :], self.F_rf),
                                             tf.transpose(
                                                 self.F_bb[:, :, n * self.bs_num_rf:(n + 1) * self.bs_num_rf, :],
                                                 [0, 2, 1, 3]))
                    inf = inf + complex_modulus(inf_n)
            inf_module = (self.power / self.num_users) * tf.reduce_sum(inf, axis=[1, 2, 3]) + self.sigma
            rate = tf.math.log(1.0 + tf.div(signal_module, inf_module)) / tf.math.log(2.)  # [batch, 1]
            Rate_user = tf.concat([Rate_user, rate], axis=-1)
            Rate = Rate + rate
        return Rate, Rate_user

    def get_rate(self, sess, sample, edge_frf, edge_fbb):
        feed_dict = {self.H: sample, self.train_ph: False, self.edge_frf_f: edge_frf, self.edge_fbb_f: edge_fbb}
        rate, rate_user = sess.run([self.Rate, self.Rate_user], feed_dict=feed_dict)  # [None, 1]
        return rate, rate_user

    def get_precoding(self, sess, sample, edge_frf, edge_fbb):
        feed_dict = {self.H: sample, self.train_ph: False, self.edge_frf_f: edge_frf, self.edge_fbb_f: edge_fbb}
        F_rf, F_bb = sess.run([self.F_rf, self.F_bb], feed_dict=feed_dict)

        return F_rf, F_bb

    def learn_batch(self, sess, sample, edge_frf, edge_fbb):
        feed_dict = {self.H: sample, self.train_ph: True, self.edge_frf_f: edge_frf, self.edge_fbb_f: edge_fbb}
        loss, _, _, _ = sess.run([self.loss, self.train_policy, self.train_lamda, self.extra_update_ops],
                                 feed_dict=feed_dict)

        return loss

    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())
