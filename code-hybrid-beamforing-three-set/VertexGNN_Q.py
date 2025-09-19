import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()
from tf_utils import flatten, fc, complex_multiply, vertex_gnn_layer_2, complex_H, complex_modulus, complex_modulus_all


class VertexGNN_Q(object):
    def __init__(self, params, *args):
        self.num_hidden = params['num_hidden']
        self.num_hidden_lamda = params['num_hidden_lamda']
        self.process_norm = params['process_norm']
        self.initial_1 = params['initial_1']
        self.initial_2 = params['initial_2']
        self.initial_3 = params['initial_3']
        self.lr = params['lr']
        self.lr_lamda = params['lr_lamda']
        self.bn = params['bn']
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
        self.F_rf, self.F_bb = self.build_network()
        self.lamda = self.build_network_lamda()
        self.Rate, self.Rate_user = self.cal_rate()
        self.Rate_user = tf.reshape(self.Rate_user, [self.batch, self.num_users])

        self.optimizer_policy = tf.train.AdamOptimizer(self.lr)
        self.optimizer_lamda = tf.train.AdamOptimizer(self.lr_lamda)

        # self.Rate_user_qos = tf.multiply(self.Rate_user, tf.sign(tf.nn.relu(self.Rate_user - self.r_min)))
        # self.Rate_qos = tf.reduce_sum(self.Rate_user_qos, axis=1)
        # self.cost = tf.reduce_mean(-1. * self.Rate_qos + tf.reduce_sum(tf.nn.sigmoid(self.lamda) *
        #                                                                tf.nn.relu(self.r_min - self.Rate_user),
        #                                                                axis=1)
        #                            )

        self.cost = tf.reduce_mean(-1. * self.Rate + tf.reduce_sum(5 * tf.nn.sigmoid(self.lamda) *
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

        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # print(self.extra_update_ops)

    def build_network(self):
        user = tf.tile(tf.reshape(tf.range(self.initial_1[0], self.initial_1[1],
                                           delta=(self.initial_1[1] - self.initial_1[0]) / self.num_users),
                                  [1, 1, self.num_users, 1]), [self.batch, 1, 1, 1])
        bs_rf = tf.tile(tf.reshape(tf.range(self.initial_2[0], self.initial_2[1],
                                            delta=(self.initial_2[1] - self.initial_2[0]) / self.bs_num_rf),
                                   [1, 1, self.bs_num_rf, 1]), [self.batch, 1, 1, 1])
        bs_an = tf.tile(tf.reshape(tf.range(self.initial_3[0], self.initial_3[1],
                                            delta=(self.initial_3[1] - self.initial_3[0]) / self.bs_num_an),
                                   [1, 1, self.bs_num_an, 1]), [self.batch, 1, 1, 1])

        for l, h in enumerate(self.num_hidden[:]):
            user, bs_rf, bs_an = vertex_gnn_layer_2(user, bs_rf, bs_an, self.H, self.FNN_hidden, self.bs_num_an,
                                                    self.bs_num_rf, self.num_users, self.bn, self.train_ph, h,
                                                    self.process_norm, self.hidden_activation, scope="policy_" + str(l))
            if self.bn:
                user = tf.layers.batch_normalization(user, training=self.train_ph, reuse=False,
                                                     name='policy_bn_1' + str(l))
                bs_rf = tf.layers.batch_normalization(bs_rf, training=self.train_ph, reuse=False,
                                                      name='policy_bn_4' + str(l))
                bs_an = tf.layers.batch_normalization(bs_an, training=self.train_ph, reuse=False,
                                                      name='policy_bn_5' + str(l))
            user = self.hidden_activation(user)
            bs_rf = self.hidden_activation(bs_rf)
            bs_an = self.hidden_activation(bs_an)

        # output layer
        user = tf.reshape(tf.tile(user, [1, 1, 1, self.bs_num_rf]),
                          [self.batch, user.get_shape().as_list()[1],
                           self.num_users * self.bs_num_rf, self.num_hidden[-1]])
        bs_rf_1 = tf.tile(bs_rf, [1, 1, self.num_users, 1])
        bs_rf_2 = tf.tile(bs_rf, [1, self.bs_num_an, 1, 1])
        bs_an = tf.tile(tf.transpose(bs_an, perm=[0, 2, 1, 3]), [1, 1, self.bs_num_rf, 1])
        F_bb = fc(tf.concat([user, bs_rf_1], axis=-1), 2, scope='policy_F_bb')
        F_rf = fc(tf.concat([bs_an, bs_rf_2], axis=-1), 2, scope='policy_F_rf')

        # Constraints
        F_rf = tf.div(F_rf, tf.tile(tf.sqrt(tf.reduce_sum(tf.square(F_rf), axis=3, keepdims=True)), [1, 1, 1, 2])) * \
               1 / np.sqrt(self.bs_num_an)
        F_bb_r = tf.reshape(F_bb, [self.batch, self.num_users, self.bs_num_rf, 2])
        norm = tf.sqrt(complex_modulus_all(complex_multiply(F_bb_r, tf.transpose(F_rf, [0, 2, 1, 3]))))
        F_bb = np.sqrt(self.num_users) * tf.div(F_bb, tf.tile(norm, [1, 1, self.num_users * self.bs_num_rf, 1]))

        return F_rf, F_bb

    def build_network_lamda(self):
        user = tf.tile(tf.reshape(tf.range(self.initial_1[0], self.initial_1[1],
                                           delta=(self.initial_1[1] - self.initial_1[0]) / self.num_users),
                                  [1, 1, self.num_users, 1]), [self.batch, 1, 1, 1])
        bs_rf = tf.tile(tf.reshape(tf.range(self.initial_2[0], self.initial_2[1],
                                            delta=(self.initial_2[1] - self.initial_2[0]) / self.bs_num_rf),
                                   [1, 1, self.bs_num_rf, 1]), [self.batch, 1, 1, 1])
        bs_an = tf.tile(tf.reshape(tf.range(self.initial_3[0], self.initial_3[1],
                                            delta=(self.initial_3[1] - self.initial_3[0]) / self.bs_num_an),
                                   [1, 1, self.bs_num_an, 1]), [self.batch, 1, 1, 1])

        for l, h in enumerate(self.num_hidden_lamda[:-1]):
            user, bs_rf, bs_an = vertex_gnn_layer_2(user, bs_rf, bs_an, self.H, self.FNN_hidden, self.bs_num_an,
                                                    self.bs_num_rf, self.num_users, self.bn, self.train_ph, h,
                                                    self.process_norm, self.hidden_activation, scope="lamda_" + str(l))
            if self.bn:
                user = tf.layers.batch_normalization(user, training=self.train_ph, reuse=False,
                                                     name='lamda_bn_1' + str(l))
                bs_rf = tf.layers.batch_normalization(bs_rf, training=self.train_ph, reuse=False,
                                                      name='lamda_bn_4' + str(l))
                bs_an = tf.layers.batch_normalization(bs_an, training=self.train_ph, reuse=False,
                                                      name='lamda_bn_5' + str(l))
            user = self.hidden_activation(user)
            bs_rf = self.hidden_activation(bs_rf)
            bs_an = self.hidden_activation(bs_an)

        user, bs_rf, bs_an = vertex_gnn_layer_2(user, bs_rf, bs_an, self.H, self.FNN_hidden, self.bs_num_an,
                                                self.bs_num_rf, self.num_users, self.bn, self.train_ph,
                                                self.num_hidden_lamda[-1],
                                                self.process_norm, self.hidden_activation, scope="lamda_out")

        lamda = tf.reduce_mean(user, axis=[1, 3])

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

    def get_rate(self, sess, sample):
        feed_dict = {self.H: sample, self.train_ph: False}
        rate, rate_user = sess.run([self.Rate, self.Rate_user], feed_dict=feed_dict)  # [None, 1]
        return rate, rate_user

    def get_precoding(self, sess, sample):
        feed_dict = {self.H: sample, self.train_ph: False}
        F_rf, F_bb = sess.run([self.F_rf, self.F_bb], feed_dict=feed_dict)

        return F_rf, F_bb

    def learn_batch(self, sess, sample):
        feed_dict = {self.H: sample, self.train_ph: True}
        loss, _, _, _ = sess.run([self.loss, self.train_policy, self.train_lamda, self.extra_update_ops],
                                 feed_dict=feed_dict)

        return loss

    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())
