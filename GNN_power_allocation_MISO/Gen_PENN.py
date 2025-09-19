import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from PENN_model import penn, get_random_block_from_data, gen_index
# from PENN_model import penn_joint, penn_1d
from powerallocation_wmmse import gen_sample, cal_sum_rate1, power_nomalize
import time
import json
import scipy.io as sio
import sys

N_TEST = 100
model_location = "./DNNmodel/PENN/model_demo.ckpt"

n_marco = 0
n_pico = 4
nUE_marco = 0
nUE_pico = 7
nTX_marco = 16
nTX_pico = 8
p_marco = 40
p_pico = 1
var_noise = 3.1623e-13

n_hidden = [2, 2, 2]
n_output = 1
# coefficient = 0.01
coefficient = 1 * 28 / 20
hidden_transfer = tf.nn.relu
output_transfer = tf.nn.softmax

time_cost = 0

p_BS = np.concatenate([np.ones(n_pico) * p_pico, np.ones(n_marco) * p_marco])
nUE_BS = np.int64(np.concatenate([np.ones(n_pico) * nUE_pico, np.ones(n_marco) * nUE_marco]))
nTX_BS = np.int64(np.concatenate([np.ones(n_pico) * nTX_pico, np.ones(n_marco) * nTX_marco]))
nUE = sum(nUE_BS)
nBS = len(nUE_BS)

access_UE_BS = np.zeros([1, len(nUE_BS), sum(nUE_BS)])
for iBS in range(len(nUE_BS)):
    access_UE_BS[0, iBS, sum(nUE_BS[0:iBS]): sum(nUE_BS[0:iBS+1])] = 1
access_BS_UE = tf.constant(np.transpose(access_UE_BS))
access_UE_BS = tf.constant(access_UE_BS)

testfilename = 'Dataset_new/Test_' + str(n_marco) + 'MBS_' + str(nUE_marco) + 'UE_' + \
               str(n_pico) + 'PBS_' + str(nUE_pico) + 'UEA1.mat'

tefile = sio.loadmat(testfilename)
A_BS_UE_test = tefile['A_BS_UE']; X_BS_test = tefile['X_BS']; X_UE_test = tefile['X_UE'];
Y_BS_test = tefile['Y_BS']; SR_test = tefile['SR']

# ================================================================================== #
A_BS_UE_test = A_BS_UE_test[0: N_TEST]; X_BS_test = X_BS_test[0: N_TEST]; X_UE_test = X_UE_test[0: N_TEST];
Y_BS_test = Y_BS_test[0: N_TEST]; SR_test = SR_test[0, 0: N_TEST]

X_test = np.zeros([A_BS_UE_test.shape[0], n_pico*n_pico*nUE_pico, nUE_pico])
for i in range(n_pico):
    for j in range(n_pico):
        X_test[:, i * n_pico * nUE_pico + j * nUE_pico:i * n_pico * nUE_pico + (j + 1) * nUE_pico, :] = \
            np.squeeze(A_BS_UE_test[:, i * nUE_pico:(i + 1) * nUE_pico, j * nUE_pico:(j + 1) * nUE_pico])
X_test = np.reshape(X_test, [A_BS_UE_test.shape[0], -1])
X_BS_test = np.reshape(X_BS_test, [X_BS_test.shape[0], -1])
Y_test = np.reshape(Y_BS_test, [Y_BS_test.shape[0], -1])

Z_UE_test = np.zeros([N_TEST, nUE])
for iBS in range(len(nUE_BS)):
    Z_UE_test[:, sum(nUE_BS[0:iBS]): sum(nUE_BS[0:iBS + 1])] = X_BS_test[:, iBS:iBS + 1]

A_BS_UE_test = A_BS_UE_test * 1e3
# ================================================================================== #

x = tf.placeholder('float', [None, nUE*nUE])
y = tf.placeholder('float', [None, nUE])

# ===================================== PENN ======================================= #
batch = tf.shape(x)[0]
index_w_1, index_b_1 = gen_index(n_pico)
index_w_2, index_b_2 = gen_index(nUE_pico)

layernum = n_hidden + [n_output]
y_pred0 = penn(n_pico, nUE_pico, x, layernum, index_w_1, index_b_1, index_w_2, index_b_2, coefficient,
               hidden_transfer=hidden_transfer, keep_prob=1.0, name='0', is_transfer=True, is_BN=True, is_test=False)
y_pred0 = tf.reshape(y_pred0, [batch, n_pico *  n_pico * nUE_pico, nUE_pico])
y_pred0 = tf.reshape(y_pred0, [batch, n_pico *  n_pico, nUE_pico, nUE_pico])
y_pred0 = tf.linalg.diag_part(y_pred0) # [batch, n_pico *  n_pico, nUE_pico]
index = np.arange(n_pico) * nUE_pico
y_pred0 = tf.gather(y_pred0, index, axis=1)
y_pred0 =  tf.reshape(y_pred0, [batch, n_pico*nUE_pico])
y_pred = output_transfer(y_pred0)

for iBS in range(len(nUE_BS)):
    if iBS == 0:
        sum_y = tf.ones([1, nUE_BS[iBS]]) * tf.reduce_sum(y_pred[:, sum(nUE_BS[0:iBS]):sum(nUE_BS[0:iBS+1])],
                                                          axis=1, keepdims=True)
        sum_y_bs = tf.ones([1, nUE_BS[iBS]]) * tf.nn.sigmoid(tf.reduce_sum(y_pred0[:, sum(nUE_BS[0:iBS]):
                                                                                      sum(nUE_BS[0:iBS+1])],
                                                             axis=1, keepdims=True))
    else:
        temp = tf.ones([1, nUE_BS[iBS]]) * tf.reduce_sum(y_pred[:, sum(nUE_BS[0:iBS]):sum(nUE_BS[0:iBS+1])],
                                                         axis=1, keepdims=True)
        sum_y = tf.concat((sum_y, temp), axis=1)
        sum_y_bs = tf.concat((sum_y_bs, tf.ones([1, nUE_BS[iBS]]) * tf.nn.sigmoid(tf.reduce_sum(
                                                                                  y_pred0[:, sum(nUE_BS[0:iBS]):
                                                                                             sum(nUE_BS[0:iBS+1])],
                                                                                  axis=1, keepdims=True))), axis=1)

y_pred = y_pred / (sum_y + 1e-5) * sum_y_bs

# =====================================无监督损失函数======================================= #
chl = tf.reshape(x, [-1, n_pico*n_pico, nUE_pico*nUE_pico])
chl = tf.reshape(chl, [-1, n_pico*n_pico, nUE_pico, nUE_pico])
chl = tf.reshape(chl, [-1, n_pico, n_pico, nUE_pico, nUE_pico])
chl = tf.transpose(chl, perm=[0, 1, 3, 2, 4])
chl = tf.transpose(tf.reshape(chl, [-1, nUE, nUE]), perm=[0, 2, 1])
chl2 = chl * chl
eye = tf.expand_dims(tf.eye(sum(nUE_BS)), axis=0)
power = tf.expand_dims(y_pred, axis=1)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

min_cost = 2020
min_epoch = 0
nnrate = 0.0
wmrate = np.sum(SR_test)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    Ratio = []
    sess.run(init)
    saver.restore(sess, model_location)
    Y_pred_test = sess.run(y_pred, feed_dict={x: X_test, y: Y_test / Z_UE_test}) * Z_UE_test
    Y_pred_test = np.reshape(Y_pred_test, [N_TEST, -1])
    Y_BS_pred_test = power_nomalize(Y_pred_test, X_BS_test, nUE_BS)
    nnrate, _ = cal_sum_rate1(A_BS_UE_test / 1e3, Y_pred_test, var_noise, nUE_BS, nTX_BS, access_UE_BS)
    print('Performance Ratio: ', '%04f' % (nnrate / wmrate * 100), '%')
