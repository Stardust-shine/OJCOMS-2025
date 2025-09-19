import numpy as np
import os
import h5py
import numpy as np
import scipy.io as sio
from tf_utils import complex_multiply_high, mdgnn_layer_multi_cell_new, complex_H, complex_modulus, \
    complex_modulus_all, complex_multiply
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def cal_rate(H_all, F_rf, F_bb, num_users_data, num_cell, num_users, bs_num_rf, power, sigma):
    Rate = 0
    for c in range(num_cell):
        p = power / tf.reduce_sum(num_users_data[:, c, :], axis=1)
        for i in range(num_users):
            signal = complex_multiply(complex_multiply(H_all[:, c, c, i:(i + 1), :, :], F_rf[:, c, :, :, :]),
                                      tf.transpose(F_bb[:, :, c, i * bs_num_rf:(i + 1) * bs_num_rf, :],
                                                   [0, 2, 1, 3]))
            signal_module = tf.multiply(p, tf.reduce_sum(complex_modulus(signal), axis=[1, 2, 3]))
            inf = 0
            for m in range(num_cell):
                for n in range(num_users):
                    if (n == i) and (m == c):
                        inf = inf + 0
                    else:
                        inf_n = complex_multiply(
                            complex_multiply(H_all[:, m, c, i:(i + 1), :, :], F_rf[:, m, :, :, :]),
                            tf.transpose(
                                F_bb[:, :, m, n * bs_num_rf:(n + 1) * bs_num_rf, :],
                                [0, 2, 1, 3]))
                        inf = inf + complex_modulus(inf_n)
            inf_module = tf.multiply(p, tf.reduce_sum(inf, axis=[1, 2, 3])) + sigma
            rate = tf.math.log(tf.cast(1., tf.float32) + tf.div(signal_module, inf_module)) / \
                   tf.math.log(tf.cast(2., tf.float32))  # [batch, 1]
            Rate = Rate + rate
    return Rate


def cal_rate_new(H_all, F_rf, F_bb, num_users_data, num_cell, num_users, bs_num_rf, power, sigma):
    Rate = 0
    for c in range(num_cell):
        p = power / tf.reduce_sum(num_users_data[:, c, :], axis=1)
        for i in range(num_users):
            signal = complex_multiply(complex_multiply(H_all[:, c, c, i:(i + 1), :, :], F_rf[:, c, :, :, :]),
                                      tf.transpose(F_bb[:, :, c, i * bs_num_rf:(i + 1) * bs_num_rf, :],
                                                   [0, 2, 1, 3]))
            signal_module = tf.multiply(p, tf.reduce_sum(complex_modulus(signal), axis=[1, 2, 3]))
            inf = 0
            for m in range(num_cell):
                for n in range(num_users):
                    if (n == i) and (m == c):
                        inf = inf + 0
                    else:
                        inf_n = complex_multiply(
                            complex_multiply(H_all[:, m, c, i:(i + 1), :, :], F_rf[:, m, :, :, :]),
                            tf.transpose(
                                F_bb[:, :, m, n * bs_num_rf:(n + 1) * bs_num_rf, :],
                                [0, 2, 1, 3]))
                        inf = inf + complex_modulus(inf_n)
            inf_module = tf.multiply(p, tf.reduce_sum(inf, axis=[1, 2, 3])) + sigma
            rate = tf.math.log(tf.cast(1., tf.float64) + tf.div(signal_module, inf_module)) / \
                   tf.math.log(tf.cast(2., tf.float64))
            Rate = Rate + rate
    return Rate


num_an = 16
num_rf = 8
users = 5
cell_max = 10
num_cell = 10
power = 1
sigma = 10 ** ((-174 + 10 * np.log10(20 * 10 ** 6) - 30)/10)

print(sigma)

num_test = 1000

file_dir = "Cell" + str(cell_max) + "_TX" + str(num_an) + "_UE" + str(users)
solution_dir = "Cell" + str(num_cell) + "_TX" + str(num_an) + "_UE" + str(users)
# Load data
data_path = "./data/" + file_dir + ".mat"

with h5py.File(data_path, 'r') as f:
    H_all_test_32 = f['H_all_test'][:].astype(np.float32)
    H_all_test_32 = np.transpose(H_all_test_32, axes=(0, 1, 3, 4, 2, 5))
    H_all_test_32 = H_all_test_32[0:num_test, :, :, :, :, :]

    H_all_test_64 = f['H_all_test'][:].astype(np.float64)
    H_all_test_64 = np.transpose(H_all_test_64, axes=(0, 1, 3, 4, 2, 5))
    H_all_test_64 = H_all_test_64[0:num_test, :, :, :, :, :]

    n_UE_test = f['num_users_test'][:]
    n_UE_test = n_UE_test[0:num_test, :, :]

solution_path = "./data/solution_" + solution_dir + ".mat"
F_rf_all = sio.loadmat(solution_path)['F_RF_all']
F_rf_all = F_rf_all[0:num_test]
F_bb_all = sio.loadmat(solution_path)['F_BB_all']
F_bb_all = F_bb_all[0:num_test]

F_bb_all = np.transpose(F_bb_all, [0, 1, 3, 2, 4])
F_bb_all = np.reshape(F_bb_all, [num_test, 1, num_cell, users*num_rf, 2])


Rate = cal_rate(H_all_test_32, F_rf_all.astype(np.float32), F_bb_all.astype(np.float32),
                n_UE_test.astype(np.float32), num_cell,
                users, num_rf, power, sigma)

Rate_new = cal_rate_new(H_all_test_64, F_rf_all, F_bb_all, n_UE_test, num_cell, users, num_rf, power, sigma)

with tf.Session() as sess:
    print(np.mean(sess.run(Rate)))

    print(np.mean(sess.run(Rate_new)))
