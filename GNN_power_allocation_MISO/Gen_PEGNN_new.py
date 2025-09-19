import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from GNN_model import gnn_pe_layer_new, get_random_block_from_data1, cal_unspv_cost
from powerallocation_wmmse import gen_sample, cal_sum_rate1, power_nomalize
import time
import json
import scipy.io as sio
import sys

is_plot = False
N_TEST = 1000
model_location = "./DNNmodel/VertexGNN/model_demo.ckpt"
ini_model_location = "./InitialModel/PEGNN/model_demo.ckpt"
min_per = 0.99
max_per = 1.00

n_marco = 0
n_pico = 4
nUE_marco = 0
nUE_pico = 5
nTX_marco = 16
nTX_pico = 8
p_marco = 40
p_pico = 1
var_noise = 3.1623e-13

n_hidden = [20, 10, 10, 10]
n_output = 1


time_cost = 0

activation_function = tf.nn.softmax
activation_function_fnn = tf.nn.relu

p_BS = np.concatenate([np.ones(n_pico) * p_pico, np.ones(n_marco) * p_marco])
nUE_BS = np.int64(np.concatenate([np.ones(n_pico) * nUE_pico, np.ones(n_marco) * nUE_marco]))
nTX_BS = np.int64(np.concatenate([np.ones(n_pico) * nTX_pico, np.ones(n_marco) * nTX_marco]))
nUE = sum(nUE_BS)
nBS = len(nUE_BS)

access_UE_BS = np.zeros([1, sum(nUE_BS), sum(nUE_BS)])
for iBS in range(len(nUE_BS)):
    access_UE_BS[0, sum(nUE_BS[0:iBS]): sum(nUE_BS[0:iBS+1]), sum(nUE_BS[0:iBS]): sum(nUE_BS[0:iBS+1])] = 1
access_UE_BS = access_UE_BS + np.eye(nUE)
access_BS_UE = tf.constant(np.transpose(access_UE_BS, axes=(0, 2, 1)))
access_UE_BS = tf.constant(access_UE_BS)

# ================================ Dataset ========================================= #
testfilename = 'Dataset_new/Test_' + str(n_marco) + 'MBS_' + str(nUE_marco) + 'UE_' + \
               str(n_pico) + 'PBS_' + str(nUE_pico) + 'UEA1.mat'


tefile = sio.loadmat(testfilename)

A_BS_UE_test = tefile['A_BS_UE']; X_BS_test = tefile['X_BS']; X_UE_test = tefile['X_UE'];
Y_UE_test = tefile['Y_BS']; SR_test = tefile['SR']
A_BS_UE_test = A_BS_UE_test[0: N_TEST]; X_BS_test = X_BS_test[0: N_TEST]; X_UE_test = X_UE_test[0: N_TEST];
Y_UE_test = Y_UE_test[0: N_TEST]; SR_test = SR_test[0, 0: N_TEST]

Z_UE_test = np.zeros([N_TEST, nUE, 1])
for iBS in range(len(nUE_BS)):
    Z_UE_test[:, sum(nUE_BS[0:iBS]): sum(nUE_BS[0:iBS + 1])] = X_BS_test[:, iBS:iBS + 1]

X_BS_test1 = X_BS_test
X_BS_test = Z_UE_test * 0.0

A_BS_UE_test = A_BS_UE_test * 1e3
# ================================================================================== #

x_bs = tf.placeholder('float', [None, nUE, 1])
x_ue = tf.placeholder('float', [None, nUE, 1])
a_bs_ue = tf.placeholder('float', [None, nUE, nUE, 1])
y_ue = tf.placeholder('float', [None, nUE, 1])

# ==========================stack of multiple GNN layers============================ #

h_bs = x_bs
h_ue = x_ue
for l, h in enumerate(n_hidden):
    h_bs = gnn_pe_layer_new(h_bs, h, a_bs_ue, h_ue, access_UE_BS, transfer_function=activation_function,
                            transfer_function_fnn=activation_function_fnn, name='bs' + str(l))
    h_ue = gnn_pe_layer_new(h_ue, h, tf.transpose(a_bs_ue, perm=[0, 2, 1, 3]),
                            h_bs, access_BS_UE, transfer_function=activation_function,
                            transfer_function_fnn=activation_function_fnn, name='ue' + str(l))

y_bs_pred0 = gnn_pe_layer_new(h_bs, n_output, '', h_ue, access_UE_BS, transfer_function_fnn=activation_function_fnn,
                              name='out_bs', is_transfer=False)
y_ue_pred0 = gnn_pe_layer_new(h_ue, n_output, '', h_bs, access_BS_UE, transfer_function_fnn=activation_function_fnn,
                              name='out_ue', is_transfer=False)

y_ue_pred = tf.nn.sigmoid(y_ue_pred0 + y_bs_pred0)

for iBS in range(len(nUE_BS)):
    if iBS == 0:
        sum_y = tf.ones([1, nUE_BS[iBS], 1]) * tf.reduce_sum(y_ue_pred[:, sum(nUE_BS[0:iBS]):sum(nUE_BS[0:iBS+1]), :],
                                                             axis=1, keepdims=True)
        sum_y_bs = tf.ones([1, nUE_BS[iBS], 1]) * tf.nn.sigmoid(tf.reduce_sum(y_bs_pred0[:, sum(nUE_BS[0:iBS]):
                                                                                           sum(nUE_BS[0:iBS+1]), :],
                                                                axis=1, keepdims=True))
    else:
        temp = tf.ones([1, nUE_BS[iBS], 1]) * tf.reduce_sum(y_ue_pred[:, sum(nUE_BS[0:iBS]):sum(nUE_BS[0:iBS+1]), :],
                                                            axis=1, keepdims=True)
        sum_y = tf.concat((sum_y, temp), axis=1)
        sum_y_bs = tf.concat((sum_y_bs, tf.ones([1, nUE_BS[iBS], 1]) *
                                        tf.nn.sigmoid(tf.reduce_sum(y_bs_pred0[:, sum(nUE_BS[0:iBS]):
                                                                                 sum(nUE_BS[0:iBS+1]), :],
                                                      axis=1, keepdims=True))), axis=1)

y_ue_pred = y_ue_pred / (sum_y + 1e-5) * sum_y_bs

# =================================================================================== #

chl = tf.transpose(a_bs_ue[:, :, :, 0], perm=[0, 2, 1]) / 1e3
chl2 = chl * chl
eye = tf.expand_dims(tf.eye(sum(nUE_BS)), axis=0)
power = tf.expand_dims(y_ue_pred[:, :, 0], axis=1)
cost = -1.0 * tf.reduce_mean((tf.reduce_sum(tf.log(1 + tf.reduce_sum(eye * chl2 * power, axis=2) /
                                    (tf.reduce_sum((1 - eye) * chl2 * power, axis=2) + var_noise)), axis=1)))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

wmrate = np.sum(SR_test)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    saver.restore(sess, model_location)
    Y_BS_pred_test = sess.run(y_ue_pred, feed_dict={x_bs: X_BS_test, x_ue: X_UE_test,
                                                    a_bs_ue: A_BS_UE_test,
                                                    y_ue: Y_UE_test / Z_UE_test}) * Z_UE_test
    Y_BS_pred_test = np.reshape(Y_BS_pred_test, [N_TEST, -1])
    Y_BS_pred_test = power_nomalize(Y_BS_pred_test, X_BS_test1[:, :, 0], nUE_BS)
    nnrate, _ = cal_sum_rate1(A_BS_UE_test / 1e3, Y_BS_pred_test, var_noise, nUE_BS, nTX_BS, access_UE_BS)
    print('Performance Ratio: ', '%04f' % (nnrate / wmrate * 100), '%')

    for num in range(N_TEST):
        start_time = time.time()
        Y_BS_pred_test = sess.run(y_ue_pred, feed_dict={x_bs: [X_BS_test[num]], x_ue: [X_UE_test[num]],
                                                        a_bs_ue: [A_BS_UE_test[num]],
                                                        y_ue: [Y_UE_test[num] / Z_UE_test[num]]})
        print('test time:', time.time() - start_time)
