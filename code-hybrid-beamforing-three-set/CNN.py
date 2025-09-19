import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()
from tf_utils import flatten, fc, conv2d, complex_multiply, complex_H, complex_modulus, complex_modulus_all


class CNN(object):
    def __init__(self, params, *args):
        self.n_filter = params['n_filter']
        self.k_sz_conv = params['k_sz_conv']
        self.k_sz_pool = params['k_sz_pool']
        self.padding = params['padding']
        self.output_hidden = params['output_hidden']

        self.lr = params['lr']
        self.bn = params['bn']
        self.user_num_an = params['user_num_an']
        self.bs_num_an = params['bs_num_an']
        self.bs_num_rf = params['bs_num_rf']
        self.num_users = params['num_users']
        self.power = params['power']
        self.sigma_dB = params['sigma_dB']
        self.sigma = 10.0 ** (-self.sigma_dB / 10.0)

        self.hidden_activation = params['hidden_activation']
        self.output_activation = params['output_activation']

        self.H = tf.placeholder(tf.float32, [None, self.num_users * self.user_num_an, self.bs_num_an, 2])
        self.batch = tf.shape(self.H)[0]

        self.H_norm = tf.sqrt(complex_modulus(self.H))
        self.input = tf.concat([self.H_norm, self.H], axis=3)

        self.train_ph = tf.placeholder(tf.bool, shape=())
        self.W_rf, self.F_rf, self.F_bb = self.build_network()
        self.Rate = self.cal_rate()

        self.loss = - tf.reduce_mean(self.Rate)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.optimize = self.optimizer.minimize(self.loss)

        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def build_network(self):
        # W_rf
        x_wrf = self.input
        for l, h in enumerate(self.n_filter[:]):
            x_wrf = conv2d(x_wrf, h, self.k_sz_conv, bn=False, activation_fn=None, scope="W_rf" + str(l),
                           training=False)
            if self.bn:
                x_wrf = tf.layers.batch_normalization(x_wrf, training=self.train_ph)
            x_wrf = self.hidden_activation(x_wrf)
        x_wrf = flatten(x_wrf)
        for j, p in enumerate(self.output_hidden[:]):
            x_wrf = fc(x_wrf, p, bn=None, activation_fn=None, scope="W_rf_output_layer" + str(j))
            if self.bn:
                x_wrf = tf.layers.batch_normalization(x_wrf, training=self.train_ph)
            x_wrf = self.hidden_activation(x_wrf)
        x_wrf = fc(x_wrf, self.num_users * self.user_num_an * 2, scope='W_rf_out')
        # if self.bn:
        #     x_wrf = tf.layers.batch_normalization(x_wrf, training=self.train_ph)
        # x_wrf = self.output_activation(x_wrf)
        W_rf = tf.reshape(x_wrf, [self.batch, 1, self.num_users * self.user_num_an, 2])

        # F_rf
        x_frf = self.input
        for l, h in enumerate(self.n_filter[:]):
            x_frf = conv2d(x_frf, h, self.k_sz_conv, bn=False, activation_fn=None, scope="F_rf" + str(l),
                           training=False)
            if self.bn:
                x_frf = tf.layers.batch_normalization(x_frf, training=self.train_ph)
            x_frf = self.hidden_activation(x_frf)
        x_frf = flatten(x_frf)
        for j, p in enumerate(self.output_hidden[:]):
            x_frf = fc(x_frf, p, bn=None, activation_fn=None, scope="F_rf_output_layer" + str(j))
            if self.bn:
                x_frf = tf.layers.batch_normalization(x_frf, training=self.train_ph)
            x_frf = self.hidden_activation(x_frf)
        x_frf = fc(x_frf, self.bs_num_an * self.bs_num_rf * 2, scope='F_rf_out')
        F_rf = tf.reshape(x_frf, [self.batch, self.bs_num_an, self.bs_num_rf, 2])

        # F_bb
        x_fbb = self.input
        for l, h in enumerate(self.n_filter[:]):
            x_fbb = conv2d(x_fbb, h, self.k_sz_conv, bn=False, activation_fn=None, scope="F_bb" + str(l),
                           training=False)
            if self.bn:
                x_fbb = tf.layers.batch_normalization(x_fbb, training=self.train_ph)
            x_fbb = self.hidden_activation(x_fbb)
        x_fbb = flatten(x_fbb)
        for j, p in enumerate(self.output_hidden[:]):
            x_fbb = fc(x_fbb, p, bn=None, activation_fn=None, scope="F_bb_output_layer" + str(j))
            if self.bn:
                x_fbb = tf.layers.batch_normalization(x_fbb, training=self.train_ph)
            x_fbb = self.hidden_activation(x_fbb)
        x_fbb = fc(x_fbb, self.num_users * self.bs_num_rf * 2, scope='F_bb_out')
        F_bb = tf.reshape(x_fbb, [self.batch, 1, self.num_users * self.bs_num_rf, 2])

        # Constraints
        W_rf = tf.div(W_rf, tf.tile(tf.sqrt(tf.reduce_sum(tf.square(W_rf), axis=3, keepdims=True)),
                                    [1, 1, 1, 2])) * 1 / np.sqrt(self.user_num_an)
        F_rf = tf.div(F_rf, tf.tile(tf.sqrt(tf.reduce_sum(tf.square(F_rf), axis=3, keepdims=True)),
                                    [1, 1, 1, 2])) * 1 / np.sqrt(self.bs_num_an)
        F_bb_r = tf.reshape(F_bb, [self.batch, self.num_users, self.bs_num_rf, 2])
        norm = tf.sqrt(complex_modulus_all(complex_multiply(F_bb_r, tf.transpose(F_rf, [0, 2, 1, 3]))))
        F_bb = np.sqrt(self.num_users) * tf.div(F_bb, tf.tile(norm, [1, 1, self.num_users * self.bs_num_rf, 1]))

        return W_rf, F_rf, F_bb

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
            rate = tf.math.log(1 + tf.div(signal_module, inf_module)) / tf.math.log(tf.constant(2.))  # [batch, 1]
            Rate = Rate + rate
        return Rate

    def get_rate(self, sess, sample):
        feed_dict = {self.H: sample, self.train_ph: False}
        rate = sess.run(self.Rate, feed_dict=feed_dict)  # [None, 1]
        return rate

    def get_precoding(self, sess, sample):
        feed_dict = {self.H: sample, self.train_ph: False}
        W_rf, F_rf, F_bb = sess.run([self.W_rf, self.F_rf, self.F_bb], feed_dict=feed_dict)

        return W_rf, F_rf, F_bb

    def learn_batch(self, sess, sample):
        feed_dict = {self.H: sample, self.train_ph: True}
        loss, _, _ = sess.run([self.loss, self.optimize, self.extra_update_ops], feed_dict=feed_dict)

        return loss

    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())
