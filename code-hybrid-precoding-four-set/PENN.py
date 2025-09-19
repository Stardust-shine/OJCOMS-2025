import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()
from tf_utils import fc, penn_layer, complex_multiply, complex_H, complex_modulus, complex_modulus_all


class PENN(object):
    def __init__(self, params, *args):
        self.num_hidden = params['num_hidden']
        self.para = params['para']
        # self.layers = np.shape(self.num_hidden)[0]
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
        self.H_r = tf.reshape(self.H, [self.batch, self.num_users, self.user_num_an, self.bs_num_an, 2])

        self.train_ph = tf.placeholder(tf.bool, shape=())
        self.W_rf, self.F_rf, self.F_bb = self.build_network()
        self.Rate = self.cal_rate()

        self.loss = - tf.reduce_mean(self.Rate)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.optimize = self.optimizer.minimize(self.loss)

        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # print(self.extra_update_ops)

    def build_network(self):
        hidden_W_rf = self.H_r
        for l, h in enumerate(self.num_hidden[:]):
            hidden_W_rf = penn_layer(hidden_W_rf, h, self.para, scope='W_rf' + str(l))
            if self.bn:
                hidden_W_rf = tf.layers.batch_normalization(hidden_W_rf, training=self.train_ph)
            hidden_W_rf = self.hidden_activation(hidden_W_rf)
        hidden_W_rf = tf.reduce_sum(hidden_W_rf, axis=3)
        hidden_W_rf = fc(hidden_W_rf, 2, scope='W_rf_out')
        W_rf = tf.reshape(hidden_W_rf, [self.batch, 1, self.num_users * self.user_num_an, 2])

        hidden_F_rf = tf.transpose(self.H_r, [0, 3, 1, 2, 4])
        # [self.batch, self.bs_num_an, self.num_users, self.user_num_an, 2]
        for l, h in enumerate(self.num_hidden[:]):
            hidden_F_rf = penn_layer(hidden_F_rf, h, self.para, scope='F_rf' + str(l))
            if self.bn:
                hidden_F_rf = tf.layers.batch_normalization(hidden_F_rf, training=self.train_ph)
            hidden_F_rf = self.hidden_activation(hidden_F_rf)
        hidden_F_rf = tf.reduce_sum(hidden_F_rf, axis=[2, 3])
        hidden_F_rf = fc(hidden_F_rf, self.bs_num_rf * 2, scope='F_rf_out')
        F_rf = tf.reshape(hidden_F_rf, [self.batch, self.bs_num_an, self.bs_num_rf, 2])

        hidden_F_bb = self.H_r
        for l, h in enumerate(self.num_hidden[:]):
            hidden_F_bb = penn_layer(hidden_F_bb, h, self.para, scope='F_bb' + str(l))
            if self.bn:
                hidden_F_bb = tf.layers.batch_normalization(hidden_F_bb, training=self.train_ph)
            hidden_F_bb = self.hidden_activation(hidden_F_bb)
        hidden_F_bb = tf.reduce_sum(hidden_F_bb, axis=[2, 3])
        hidden_F_bb = fc(hidden_F_bb, self.bs_num_rf * 2, scope='F_bb_out')
        # if self.bn:
        #     hidden_F_bb = tf.layers.batch_normalization(hidden_F_bb, training=self.train_ph)
        # hidden_F_bb = self.output_activation(hidden_F_bb)
        F_bb = tf.reshape(hidden_F_bb, [self.batch, 1, self.num_users * self.bs_num_rf, 2])

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
