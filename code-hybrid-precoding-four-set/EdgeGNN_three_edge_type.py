import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
from tf_utils import complex_multiply, edge_gnn_layer_three_edge_type, complex_H, complex_modulus, complex_modulus_all


class EdgeGNN_three_edge_type(object):
    def __init__(self, params):
        self.num_hidden = params['num_hidden']
        self.lr = params['lr']
        self.bn = params['bn']
        self.FNN_type = params['FNN_type']
        self.process_norm = params['process_norm']
        self.initial_1 = params['initial_1']
        self.initial_2 = params['initial_2']
        self.initial_3 = params['initial_3']
        self.initial_4 = params['initial_4']
        self.initial_5 = params['initial_5']
        self.FNN_hidden = params['FNN_hidden']
        self.user_num_an = params['user_num_an']
        self.bs_num_an = params['bs_num_an']
        self.bs_num_rf = params['bs_num_rf']
        self.num_users = params['num_users']
        self.power = params['power']
        self.sigma_dB = params['sigma_dB']
        self.sigma = 10.0 ** (-self.sigma_dB / 10.0)

        self.hidden_activation = params['hidden_activation']

        self.H = tf.placeholder(tf.float32, [None, self.num_users * self.user_num_an, self.bs_num_an, 2])
        self.batch = tf.shape(self.H)[0]
        self.train_ph = tf.placeholder(tf.bool, shape=())
        self.W_rf, self.F_rf, self.F_bb, self.edge_wrf_f, self.edge_frf_f, self.edge_fbb_f = self.build_network()
        self.Rate = self.cal_rate()

        self.loss = - tf.reduce_mean(self.Rate)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.optimize = self.optimizer.minimize(self.loss)

        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def build_network(self):
        edge_wrf_f = tf.placeholder(tf.float32, [None, self.num_users, self.user_num_an, 2])
        edge_h_f = self.H
        edge_frf_f = tf.placeholder(tf.float32, [None, self.bs_num_an, self.bs_num_rf, 2])
        edge_fbb_f = tf.placeholder(tf.float32, [None, self.num_users, self.bs_num_rf, 2])

        edge_wrf, edge_h, edge_frf, edge_fbb = edge_wrf_f, edge_h_f, edge_frf_f, edge_fbb_f

        for l, h in enumerate(self.num_hidden[:]):
            edge_wrf, edge_h, edge_frf, edge_fbb = edge_gnn_layer_three_edge_type(edge_wrf, edge_h, edge_frf, edge_fbb,
                                                                                  self.H, self.FNN_hidden,
                                                                                  self.user_num_an, self.bs_num_an,
                                                                                  self.bs_num_rf, self.num_users,
                                                                                  self.bn,
                                                                                  self.train_ph, h, self.process_norm,
                                                                                  self.FNN_type, self.hidden_activation,
                                                                                  scope="gnn" + str(l))
            if self.bn:
                edge_wrf = tf.layers.batch_normalization(edge_wrf, training=self.train_ph, reuse=False,
                                                         name='bn_1' + str(l))
                edge_h = tf.layers.batch_normalization(edge_h, training=self.train_ph, reuse=False,
                                                       name='bn_2' + str(l))
                edge_frf = tf.layers.batch_normalization(edge_frf, training=self.train_ph, reuse=False,
                                                         name='bn_3' + str(l))
                edge_fbb = tf.layers.batch_normalization(edge_fbb, training=self.train_ph, reuse=False,
                                                         name='bn_4' + str(l))
            edge_wrf = self.hidden_activation(edge_wrf)
            edge_h = self.hidden_activation(edge_h)
            edge_frf = self.hidden_activation(edge_frf)
            edge_fbb = self.hidden_activation(edge_fbb)
        # output layer
        edge_wrf, edge_h, edge_frf, edge_fbb = edge_gnn_layer_three_edge_type(edge_wrf, edge_h, edge_frf, edge_fbb,
                                                                              self.H, self.FNN_hidden,
                                                                              self.user_num_an, self.bs_num_an,
                                                                              self.bs_num_rf, self.num_users, self.bn,
                                                                              self.train_ph, 2, self.process_norm,
                                                                              self.FNN_type, self.hidden_activation,
                                                                              scope="gnn_out")
        # edge_wrf : [batch, 1, self.num_users*self.user_num_an, _]
        edge_wrf = tf.reshape(edge_wrf, [self.batch, 1, self.num_users * self.user_num_an, 2])
        # edge_fbb : [batch, 1, self.num_users*self.bs_num_rf, _]
        edge_fbb = tf.reshape(edge_fbb, [self.batch, 1, self.num_users * self.bs_num_rf, 2])

        # output
        W_rf = edge_wrf
        F_bb = edge_fbb
        F_rf = edge_frf

        # Constraints
        W_rf = tf.div(W_rf, tf.tile(tf.sqrt(tf.reduce_sum(tf.square(W_rf), axis=3, keepdims=True)),
                                    [1, 1, 1, 2])) * 1 / np.sqrt(self.user_num_an)
        F_rf = tf.div(F_rf, tf.tile(tf.sqrt(tf.reduce_sum(tf.square(F_rf), axis=3, keepdims=True)),
                                    [1, 1, 1, 2])) * 1 / np.sqrt(self.bs_num_an)
        F_bb_r = tf.reshape(F_bb, [self.batch, self.num_users, self.bs_num_rf, 2])
        norm = tf.sqrt(complex_modulus_all(complex_multiply(F_bb_r, tf.transpose(F_rf, [0, 2, 1, 3]))))
        F_bb = np.sqrt(self.num_users) * tf.div(F_bb, tf.tile(norm, [1, 1, self.num_users * self.bs_num_rf, 1]))

        return W_rf, F_rf, F_bb, edge_wrf_f, edge_frf_f, edge_fbb_f

    def cal_rate(self):
        Rate = 0
        for i in range(self.num_users):
            signal = complex_multiply(complex_multiply(
                complex_multiply(complex_H(self.W_rf[:, :, i * self.user_num_an:(i + 1) * self.user_num_an, :]),
                                 self.H[:, i * self.user_num_an:(i + 1) * self.user_num_an, :, :]), self.F_rf),
                                      tf.transpose(self.F_bb[:, :, i * self.bs_num_rf:(i + 1) * self.bs_num_rf, :],
                                                   [0, 2, 1, 3]))
            signal_module = (self.power / self.num_users) * tf.reduce_sum(complex_modulus(signal), axis=[1, 2, 3])
            inf = 0
            for n in range(self.num_users):
                if n == i:
                    inf = inf + 0
                else:
                    inf_n = complex_multiply(complex_multiply(
                        complex_multiply(complex_H(self.W_rf[:, :, i * self.user_num_an:(i + 1) * self.user_num_an, :]),
                                         self.H[:, i * self.user_num_an:(i + 1) * self.user_num_an, :, :]), self.F_rf),
                                             tf.transpose(
                                                 self.F_bb[:, :, n * self.bs_num_rf:(n + 1) * self.bs_num_rf, :],
                                                 [0, 2, 1, 3]))
                    inf = inf + complex_modulus(inf_n)
            inf_module = (self.power / self.num_users) * tf.reduce_sum(inf, axis=[1, 2, 3]) + self.sigma
            rate = tf.math.log(1.0 + tf.div(signal_module, inf_module)) / tf.math.log(2.)  # [batch, 1]
            Rate = Rate + rate
        return Rate

    def get_rate(self, sess, sample, edge_wrf, edge_frf, edge_fbb):
        feed_dict = {self.H: sample, self.train_ph: False,
                     self.edge_wrf_f: edge_wrf, self.edge_frf_f: edge_frf,
                     self.edge_fbb_f: edge_fbb}
        rate = sess.run(self.Rate, feed_dict=feed_dict)  # [None, 1]
        return rate

    def get_precoding(self, sess, sample, edge_wrf, edge_frf, edge_fbb):
        feed_dict = {self.H: sample, self.train_ph: False,
                     self.edge_wrf_f: edge_wrf, self.edge_frf_f: edge_frf,
                     self.edge_fbb_f: edge_fbb}
        W_rf, F_rf, F_bb = sess.run([self.W_rf, self.F_rf, self.F_bb], feed_dict=feed_dict)

        return W_rf, F_rf, F_bb

    def learn_batch(self, sess, sample, edge_wrf, edge_frf, edge_fbb):
        feed_dict = {self.H: sample, self.train_ph: True,
                     self.edge_wrf_f: edge_wrf, self.edge_frf_f: edge_frf,
                     self.edge_fbb_f: edge_fbb}
        loss, _, _ = sess.run([self.loss, self.optimize, self.extra_update_ops], feed_dict=feed_dict)

        return loss

    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())

    def model_restore(self, sess, saver, model):
        saver.restore(sess, model)
