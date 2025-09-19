import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()
from tf_utils import complex_multiply, mdgnn_layer, complex_H, complex_modulus, complex_modulus_all


class MDGNN(object):
    def __init__(self, params, *args):
        self.num_hidden = params['num_hidden']
        self.process_norm = params['process_norm']
        self.initial_1 = params['initial_1']
        self.initial_2 = params['initial_2']
        self.initial_3 = params['initial_3']
        self.initial_4 = params['initial_4']
        self.initial_5 = params['initial_5']
        self.lr = params['lr']
        self.bn = params['bn']
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
        self.W_rf, self.F_rf, self.F_bb = self.build_network()
        self.Rate = self.cal_rate()

        self.loss = - tf.reduce_mean(self.Rate)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.optimize = self.optimizer.minimize(self.loss)

        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # print(self.extra_update_ops)

    def build_network(self):
        user_data = tf.tile(tf.reshape(tf.range(self.initial_1[0], self.initial_1[1],
                                                delta=(self.initial_1[1] - self.initial_1[0]) / self.num_users),
                                       [1, self.num_users, 1]), [self.batch, 1, 1])
        user_rf = tf.tile(tf.reshape(tf.range(self.initial_2[0], self.initial_2[1],
                                              delta=(self.initial_2[1] - self.initial_2[0]) / self.num_users),
                                     [1, self.num_users, 1]), [self.batch, 1, 1])
        user_an = tf.tile(tf.reshape(tf.range(self.initial_3[0], self.initial_3[1],
                                              delta=(self.initial_3[1] - self.initial_3[0]) / (
                                                          self.num_users * self.user_num_an)),
                                     [1, self.num_users * self.user_num_an, 1]), [self.batch, 1, 1])
        bs_rf = tf.tile(tf.reshape(tf.range(self.initial_4[0], self.initial_4[1],
                                            delta=(self.initial_4[1] - self.initial_4[0]) / self.bs_num_rf),
                                   [1, self.bs_num_rf, 1]), [self.batch, 1, 1])
        bs_an = tf.tile(tf.reshape(tf.range(self.initial_5[0], self.initial_5[1],
                                            delta=(self.initial_5[1] - self.initial_5[0]) / self.bs_num_an),
                                   [1, self.bs_num_an, 1]), [self.batch, 1, 1])

        H_r = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.H, axis=3), axis=4), axis=5)
        H_r = tf.tile(H_r, [1, 1, 1, self.bs_num_rf, self.num_users, self.num_users, 1])
        real_H, im_H = tf.split(H_r, [1, 1], axis=-1)

        user_data = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(user_data, axis=1), axis=1), axis=1),
                                   axis=1)
        user_data = tf.tile(user_data,
                            [1, self.num_users * self.user_num_an, self.bs_num_an, self.bs_num_rf, self.num_users, 1,
                             1])

        user_rf = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(user_rf, axis=1), axis=1), axis=1),
                                 axis=1)
        user_rf = tf.tile(user_rf,
                          [1, self.num_users * self.user_num_an, self.bs_num_an, self.bs_num_rf, self.num_users, 1, 1])

        bs_rf = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(bs_rf, axis=1), axis=1), axis=4), axis=4)
        bs_rf = tf.tile(bs_rf,
                        [1, self.num_users * self.user_num_an, self.bs_num_an, 1, self.num_users, self.num_users, 1])

        for l, h in enumerate(self.num_hidden[:-1]):
            real_H, im_H, user_data, user_rf, bs_rf = mdgnn_layer(real_H, im_H, user_data, user_rf, bs_rf, self.bn,
                                                                  self.train_ph, h, self.hidden_activation,
                                                                  self.process_norm, scope="mdgnn" + str(l))
        real_H, im_H, user_data, user_rf, bs_rf = mdgnn_layer(real_H, im_H, user_data, user_rf, bs_rf, bn=False,
                                                              train_ph=self.train_ph, out_shape=self.num_hidden[-1],
                                                              hidden_activation=None, scope="mdgnn_out")

        # output layer
        # W_rf: [batch, 1, self.num_users*self.user_num_an, 2]
        W_rf = tf.expand_dims(tf.reduce_mean(real_H, axis=[2, 3, 4, 5]), axis=1)
        # F_bb: [batch, 1, self.num_users*self.bs_num_rf, 2]
        F_bb = tf.reshape(tf.reduce_mean(im_H, axis=[1, 2, 5]), [self.batch, 1, self.num_users * self.bs_num_rf, 2])
        # F_rf: [batch, self.bs_num_an, self.bs_num_rf, 2]
        F_rf = tf.reduce_mean(bs_rf, axis=[1, 4, 5])

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
