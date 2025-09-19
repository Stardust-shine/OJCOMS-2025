import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()
from tf_utils import complex_multiply, mdgnn_layer, complex_H, complex_modulus, complex_modulus_all


class MDGNN(object):
    def __init__(self, params, *args):
        self.num_hidden = params['num_hidden']
        self.process_norm = params['process_norm']
        self.initial_1 = params['initial_1']
        self.lr = params['lr']
        self.bn = params['bn']
        self.FNN_hidden = params['FNN_hidden']
        self.bs_num_an = params['bs_num_an']
        self.bs_num_rf = params['bs_num_rf']
        self.num_users = params['num_users']
        self.power = params['power']
        self.sigma_dB = params['sigma_dB']
        self.sigma = 10.0 ** (-self.sigma_dB / 10.0)

        self.hidden_activation = params['hidden_activation']

        self.H = tf.placeholder(tf.float32, [None, self.num_users, self.bs_num_an, 2])
        self.batch = tf.shape(self.H)[0]
        self.train_ph = tf.placeholder(tf.bool, shape=())
        self.F_rf, self.F_bb = self.build_network()
        self.Rate = self.cal_rate()

        self.loss = - tf.reduce_mean(self.Rate)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.optimize = self.optimizer.minimize(self.loss)

        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # print(self.extra_update_ops)

    def build_network(self):
        bs_rf = tf.tile(tf.reshape(tf.range(self.initial_1[0], self.initial_1[1],
                                            delta=(self.initial_1[1] - self.initial_1[0]) / self.bs_num_rf),
                                   [1, self.bs_num_rf, 1]), [self.batch, 1, 1])

        H_r = tf.expand_dims(self.H, axis=3)
        H_r = tf.tile(H_r, [1, 1, 1, self.bs_num_rf, 1])
        real_H, im_H = tf.split(H_r, [1, 1], axis=-1)

        bs_rf = tf.expand_dims(tf.expand_dims(bs_rf, axis=1), axis=1)
        bs_rf = tf.tile(bs_rf, [1, self.num_users, self.bs_num_an, 1, 1])

        for l, h in enumerate(self.num_hidden[:-1]):
            real_H, im_H, bs_rf = mdgnn_layer(real_H, im_H, bs_rf, self.bn, self.train_ph, h, self.hidden_activation,
                                              self.process_norm, scope="mdgnn" + str(l))
        real_H, im_H, bs_rf = mdgnn_layer(real_H, im_H, bs_rf, bn=False, train_ph=self.train_ph,
                                          out_shape=self.num_hidden[-1], hidden_activation=None, scope="mdgnn_out")

        F_bb = tf.reshape(tf.reduce_mean(im_H, axis=2), [self.batch, 1, self.num_users * self.bs_num_rf, 2])
        F_rf = tf.reduce_mean(bs_rf, axis=1)

        F_rf = tf.div(F_rf, tf.tile(tf.sqrt(tf.reduce_sum(tf.square(F_rf), axis=3, keepdims=True)),
                                    [1, 1, 1, 2])) * 1 / np.sqrt(self.bs_num_an)
        F_bb_r = tf.reshape(F_bb, [self.batch, self.num_users, self.bs_num_rf, 2])
        norm = tf.sqrt(complex_modulus_all(complex_multiply(F_bb_r, tf.transpose(F_rf, [0, 2, 1, 3]))))
        F_bb = np.sqrt(self.num_users) * tf.div(F_bb, tf.tile(norm, [1, 1, self.num_users * self.bs_num_rf, 1]))

        return F_rf, F_bb

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
            rate = tf.math.log(1 + tf.div(signal_module, inf_module)) / tf.math.log(tf.constant(2.))  # [batch, 1]
            Rate = Rate + rate
        return Rate

    def get_rate(self, sess, sample):
        feed_dict = {self.H: sample, self.train_ph: False}
        rate = sess.run(self.Rate, feed_dict=feed_dict)  # [None, 1]
        return rate

    def get_precoding(self, sess, sample):
        feed_dict = {self.H: sample, self.train_ph: False}
        F_rf, F_bb = sess.run([self.F_rf, self.F_bb], feed_dict=feed_dict)

        return F_rf, F_bb

    def learn_batch(self, sess, sample):
        feed_dict = {self.H: sample, self.train_ph: True}
        loss, _, _ = sess.run([self.loss, self.optimize, self.extra_update_ops], feed_dict=feed_dict)

        return loss

    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())
