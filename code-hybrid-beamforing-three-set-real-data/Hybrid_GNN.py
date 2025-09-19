import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_eager_execution()
from tf_utils import flatten, fc, complex_multiply, GNN_2d_layer, PENN_BF_scale, \
    complex_modulus, complex_modulus_all, model_GNN_scale, complex_H, complex_multiply_high, select_top_diag_columns, \
    edge_gnn_layer_3, edge_gnn_layer_4


class Hybrid_GNN(object):
    def __init__(self, params, *args):
        self.lr = params['lr']

        self.num_hidden_EdgeGNN = params['num_hidden_EdgeGNN']
        self.bn_EdgeGNN = params['bn_EdgeGNN']
        self.initial_1 = params['initial_1']
        self.initial_2 = params['initial_2']
        self.FNN_type = params['FNN_type']
        self.FNN_hidden = params['FNN_hidden']
        self.process_norm = params['process_norm']
        self.hidden_activation_EdgeGNN = params['hidden_activation_EdgeGNN']

        self.num_hidden_model_GNN = params['num_hidden_model_GNN']
        self.bn_model_GNN = params['bn_model_GNN']
        self.hidden_activation_model_GNN = params['hidden_activation_model_GNN']
        self.output_transfer_model_GNN = params['output_transfer_model_GNN']

        self.num_hidden_factor_fnn = params['num_hidden_factor_fnn']
        self.K_factor = params['K_factor']
        self.bn_model_GNN_FNN = params['bn_model_GNN_FNN']
        self.hidden_ac_factor_fnn = params['hidden_ac_factor_fnn']
        self.output_ac_factor_fnn = params['output_ac_factor_fnn']

        self.bs_num_an = params['bs_num_an']
        self.bs_num_rf = params['bs_num_rf']
        self.num_users = params['num_users']

        self.power = params['power']
        self.sigma_dB = params['sigma_dB']
        self.sigma = 10.0 ** (-self.sigma_dB / 10.0)

        self.H = tf.placeholder(tf.float32, [None, self.num_users, self.bs_num_an, 2])
        self.H_norm = tf.placeholder(tf.float32, [None, self.num_users, self.bs_num_an, 2])

        self.train_ph = tf.placeholder(tf.bool, shape=())

        self.batch = tf.shape(self.H)[0]
        self.F_rf, self.F_bb, self.edge_frf_f, self.edge_fbb_f = self.build_network()

        self.Rate = self.cal_rate()

        self.loss = - tf.reduce_mean(self.Rate)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.optimize = self.optimizer.minimize(self.loss)

        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def build_network(self):
        # # # # EdgeGNN for learning analog precoding
        edge_h_f = self.H_norm
        edge_frf_f = tf.placeholder(tf.float32, [None, self.bs_num_an, self.bs_num_rf, 2])
        edge_fbb_f = tf.placeholder(tf.float32, [None, self.num_users, self.bs_num_rf, 2])

        edge_h, edge_frf, edge_fbb = edge_h_f, edge_frf_f, edge_fbb_f

        for l, h in enumerate(self.num_hidden_EdgeGNN[:]):
            edge_h, edge_frf, edge_fbb = edge_gnn_layer_3(edge_h, edge_frf, edge_fbb, self.H_norm, self.FNN_hidden,
                                                          self.bs_num_an, self.bs_num_rf, self.num_users,
                                                          self.bn_EdgeGNN, self.train_ph, h, self.process_norm,
                                                          self.FNN_type, self.hidden_activation_EdgeGNN,
                                                          scope="gnn" + str(l))
            if self.bn_EdgeGNN:
                edge_h = tf.layers.batch_normalization(edge_h, training=self.train_ph, reuse=False,
                                                       name='bn_2' + str(l))
                edge_frf = tf.layers.batch_normalization(edge_frf, training=self.train_ph, reuse=False,
                                                         name='bn_3' + str(l))
                edge_fbb = tf.layers.batch_normalization(edge_fbb, training=self.train_ph, reuse=False,
                                                         name='bn_4' + str(l))
            edge_h = self.hidden_activation_EdgeGNN(edge_h)
            edge_frf = self.hidden_activation_EdgeGNN(edge_frf)
            edge_fbb = self.hidden_activation_EdgeGNN(edge_fbb)
        # output layer
        edge_h, edge_frf, edge_fbb = edge_gnn_layer_3(edge_h, edge_frf, edge_fbb, self.H_norm, self.FNN_hidden,
                                                      self.bs_num_an, self.bs_num_rf, self.num_users, self.bn_EdgeGNN,
                                                      self.train_ph, 2, self.process_norm, self.FNN_type,
                                                      self.hidden_activation_EdgeGNN, scope="gnn_out")

        # # # #
        F_rf = edge_frf
        F_rf = tf.div(F_rf, tf.tile(tf.sqrt(tf.reduce_sum(tf.square(F_rf), axis=3, keepdims=True)),
                                    [1, 1, 1, 2])) * 1 / np.sqrt(self.bs_num_an)

        Heq = tf.linalg.matmul(tf.transpose(self.H_norm, [0, 3, 1, 2]), tf.transpose(F_rf, [0, 3, 1, 2]))
        Heq = tf.transpose(Heq, [0, 2, 3, 1])  # Heq: [batch, K, Nrf, 2]

        Heq_r = Heq

        shade_mat = tf.ones([self.batch, self.num_users, self.bs_num_rf, 1])
        F_bb = model_GNN_scale(Heq_r, shade_mat, self.num_hidden_model_GNN, self.num_hidden_factor_fnn,
                               self.hidden_ac_factor_fnn, self.hidden_activation_model_GNN, self.output_ac_factor_fnn,
                               self.output_transfer_model_GNN, scope='model_GNN', is_bn_gnn=self.bn_model_GNN,
                               is_bn_fnn=self.bn_model_GNN_FNN,
                               is_train=self.train_ph, k=1, is_mul_K=self.K_factor)

        F_bb = tf.reshape(F_bb, [self.batch, self.num_users, self.bs_num_rf, 2])
        F_bb = tf.reshape(F_bb, [self.batch, 1, self.num_users*self.bs_num_rf, 2])

        # Constraints
        F_bb_r = tf.reshape(F_bb, [self.batch, self.num_users, self.bs_num_rf, 2])
        norm = tf.sqrt(complex_modulus_all(complex_multiply(F_bb_r, tf.transpose(F_rf, [0, 2, 1, 3]))))
        F_bb = np.sqrt(self.num_users) * tf.div(F_bb, tf.tile(norm, [1, 1, self.num_users * self.bs_num_rf, 1]))

        return F_rf, F_bb, edge_frf_f, edge_fbb_f

    def cal_rate(self):
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
            Rate = Rate + rate
        return Rate

    def get_rate(self, sess, sample, sample_norm, edge_frf, edge_fbb):
        feed_dict = {self.H: sample, self.H_norm: sample_norm, self.train_ph: False, self.edge_frf_f: edge_frf,
                     self.edge_fbb_f: edge_fbb}
        rate = sess.run(self.Rate, feed_dict=feed_dict)  # [None, 1]
        return rate

    def get_precoding(self, sess, sample, sample_norm, edge_frf, edge_fbb):
        feed_dict = {self.H: sample, self.H_norm: sample_norm, self.train_ph: False, self.edge_frf_f: edge_frf,
                     self.edge_fbb_f: edge_fbb}
        F_rf, F_bb = sess.run([self.F_rf, self.F_bb], feed_dict=feed_dict)

        return F_rf, F_bb

    def learn_batch(self, sess, sample, sample_norm, edge_frf, edge_fbb):
        feed_dict = {self.H: sample, self.H_norm: sample_norm, self.train_ph: True, self.edge_frf_f: edge_frf,
                     self.edge_fbb_f: edge_fbb}
        loss, _, _ = sess.run([self.loss, self.optimize, self.extra_update_ops], feed_dict=feed_dict)

        return loss

    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())




















