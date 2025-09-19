import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()
from tf_utils import flatten, fc, complex_multiply, cnn_gnn_layer, complex_H, complex_modulus, complex_modulus_all


class HomoCNNGNN(object):
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
        self.stride = params['stride']
        self.filters = params['filters']
        self.CNN_dim = params['CNN_dim']
        self.FNN_hidden = params['FNN_hidden']
        self.user_num_an = params['user_num_an']
        self.bs_num_an = params['bs_num_an']
        self.bs_num_rf = params['bs_num_rf']
        self.num_users = params['num_users']
        self.power = params['power']
        self.sigma_dB = params['sigma_dB']
        self.sigma = 10.0 ** (-self.sigma_dB / 10.0)

        self.hidden_activation_1 = params['hidden_activation_1']
        self.hidden_activation_2 = params['hidden_activation_2']

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
        user_rf = tf.tile(tf.reshape(tf.range(self.initial_2[0], self.initial_2[1],
                                              delta=(self.initial_2[1] - self.initial_2[0]) / self.num_users),
                                     [1, 1, self.num_users, 1]), [self.batch, 1, 1, 1])
        user_an = tf.tile(tf.reshape(tf.range(self.initial_3[0], self.initial_3[1],
                                              delta=(self.initial_3[1] - self.initial_3[0]) / (
                                                          self.num_users * self.user_num_an)),
                                     [1, 1, self.num_users * self.user_num_an, 1]), [self.batch, 1, 1, 1])
        bs_rf = tf.tile(tf.reshape(tf.range(self.initial_4[0], self.initial_4[1],
                                            delta=(self.initial_4[1] - self.initial_4[0]) / self.bs_num_rf),
                                   [1, 1, self.bs_num_rf, 1]), [self.batch, 1, 1, 1])
        bs_an = tf.tile(tf.reshape(tf.range(self.initial_5[0], self.initial_5[1],
                                            delta=(self.initial_5[1] - self.initial_5[0]) / self.bs_num_an),
                                   [1, 1, self.bs_num_an, 1]), [self.batch, 1, 1, 1])

        for l, h in enumerate(self.num_hidden[:-1]):
            user_rf, user_an, bs_rf, bs_an = cnn_gnn_layer(user_rf, user_an, bs_rf, bs_an,
                                                           self.H, self.stride, self.filters, self.CNN_dim,
                                                           self.FNN_hidden,
                                                           self.user_num_an, self.bs_num_an, self.bs_num_rf,
                                                           self.num_users,
                                                           self.bn, self.train_ph, h, self.process_norm,
                                                           self.hidden_activation_1, self.hidden_activation_2,
                                                           scope="gnn" + str(l))
        user_rf, user_an, bs_rf, bs_an = cnn_gnn_layer(user_rf, user_an, bs_rf, bs_an,
                                                       self.H, self.stride, self.filters, self.CNN_dim, self.FNN_hidden,
                                                       self.user_num_an, self.bs_num_an, self.bs_num_rf,
                                                       self.num_users,
                                                       self.bn, self.train_ph, 1, self.process_norm, None, None,
                                                       scope="out")

        # output layer
        user_rf_0 = tf.reshape(tf.tile(user_rf, [1, 1, 1, self.user_num_an]),
                               [self.batch, user_rf.get_shape().as_list()[1],
                                self.num_users * self.user_num_an, self.FNN_hidden[-1]])
        user_rf_1 = tf.reshape(tf.tile(user_rf, [1, 1, 1, self.bs_num_rf]),
                               [self.batch, user_rf.get_shape().as_list()[1],
                                self.num_users * self.bs_num_rf,
                                self.FNN_hidden[-1]])  # [batch, 1, self.num_users*self.bs_num_rf, _]
        bs_rf_1 = tf.tile(bs_rf, [1, 1, self.num_users, 1])  # [batch, 1, self.num_users*self.bs_num_rf, _]
        bs_rf_2 = tf.tile(bs_rf, [1, self.bs_num_an, 1, 1])  # [batch, self.bs_num_an, self.bs_num_rf, _]
        bs_an = tf.tile(tf.transpose(bs_an, perm=[0, 2, 1, 3]), [1, 1, self.bs_num_rf, 1])
        # output
        W_rf = fc(tf.concat([user_rf_0, user_an], axis=-1), 2, scope='W_rf')
        # W_rf: [batch, 1, self.num_users*self.user_num_an, 2]
        F_bb = fc(tf.concat([user_rf_1, bs_rf_1], axis=-1), 2, scope='F_bb')
        # F_bb: [batch, 1, self.num_users*self.bs_num_rf, 2]
        F_rf = fc(tf.concat([bs_an, bs_rf_2], axis=-1), 2, scope='F_rf')
        # F_rf: [batch, self.bs_num_an, self.bs_num_rf, 2]

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
