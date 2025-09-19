import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
from powerallocation_wmmse import gen_sample, cal_sum_rate1, power_nomalize
from EdgeGNN_model import gen_sample, Edge_GNN_kp, Edge_GNN_qt, Edge_GNN_edge_type_only, Edge_GNN_vertex_type_only, \
    get_random_block_from_data
import time
import json
import scipy.io as sio

is_plot = False
N_TRAIN = 30
LEARNING_RATE = 0.01
BATCH_SIZE = 16
MAX_EPOCHS = 5000
N_TEST = 1000
model_location = "./DNNmodel/EdgeGNN/model_demo.ckpt"
ini_model_location = "./InitialModel/PEGNN/model_demo.ckpt"
min_per = 0.99
max_per = 1.00

n_marco = 0
n_pico = 4
nUE_marco = 0
nUE_pico = 10
nTX_marco = 16
nTX_pico = 8
p_marco = 40
p_pico = 1
var_noise = 3.1623e-13

n_hidden = [20, 10, 10, 10]
# n_hidden = [30, 20, 20, 20]
n_output = 1
transfer_function = tf.nn.relu

time_cost = 0

p_BS = np.concatenate([np.ones(n_pico) * p_pico, np.ones(n_marco) * p_marco])
nUE_BS = np.int64(np.concatenate([np.ones(n_pico) * nUE_pico, np.ones(n_marco) * nUE_marco]))
nTX_BS = np.int64(np.concatenate([np.ones(n_pico) * nTX_pico, np.ones(n_marco) * nTX_marco]))
nUE = sum(nUE_BS)
nBS = len(nUE_BS)

access_UE_BS = np.zeros([1, sum(nUE_BS), sum(nUE_BS)])
for iBS in range(len(nUE_BS)):
    access_UE_BS[0, sum(nUE_BS[0:iBS]): sum(nUE_BS[0:iBS + 1]), sum(nUE_BS[0:iBS]): sum(nUE_BS[0:iBS + 1])] = 1
access_UE_BS = access_UE_BS + np.eye(nUE)
access_BS_UE = tf.constant(np.transpose(access_UE_BS, axes=(0, 2, 1)))
access_UE_BS = tf.constant(access_UE_BS)

# ================================ Dataset ========================================= #

trainfilename = 'Dataset_new/Train_' + str(n_marco) + 'MBS_' + str(nUE_marco) + 'UE_' + \
                str(n_pico) + 'PBS_' + str(nUE_pico) + 'UEA1.mat'
testfilename = 'Dataset_new/Test_' + str(n_marco) + 'MBS_' + str(nUE_marco) + 'UE_' + \
               str(n_pico) + 'PBS_' + str(nUE_pico) + 'UEA1.mat'

trfile = sio.loadmat(trainfilename);
tefile = sio.loadmat(testfilename)

A_BS_UE = trfile['A_BS_UE'];
X_BS = trfile['X_BS'];
X_UE = trfile['X_UE'];
Y_UE = trfile['Y_BS'];
SR = trfile['SR']
A_BS_UE = A_BS_UE[0: N_TRAIN];
X_BS = X_BS[0: N_TRAIN];
X_UE = X_UE[0: N_TRAIN];
Y_UE = Y_UE[0: N_TRAIN];
SR = SR[0, 0:N_TRAIN]

X_1, X_2, X_3, Index_1, Index_2, Index_3 = gen_sample(n_pico, nUE_pico, A_BS_UE)

A_BS_UE_test = tefile['A_BS_UE'];
X_BS_test = tefile['X_BS'];
X_UE_test = tefile['X_UE'];
Y_UE_test = tefile['Y_BS'];
SR_test = tefile['SR']
A_BS_UE_test = A_BS_UE_test[0: N_TEST];
X_BS_test = X_BS_test[0: N_TEST];
X_UE_test = X_UE_test[0: N_TEST];
Y_UE_test = Y_UE_test[0: N_TEST];
SR_test = SR_test[0, 0: N_TEST]

X_1_test, X_2_test, X_3_test, Index_1_test, Index_2_test, Index_3_test = gen_sample(n_pico, nUE_pico, A_BS_UE_test)

Z_UE = np.zeros([N_TRAIN, nUE, 1]);
Z_UE_test = np.zeros([N_TEST, nUE, 1])
for iBS in range(len(nUE_BS)):
    Z_UE[:, sum(nUE_BS[0:iBS]): sum(nUE_BS[0:iBS + 1])] = X_BS[:, iBS:iBS + 1]
    Z_UE_test[:, sum(nUE_BS[0:iBS]): sum(nUE_BS[0:iBS + 1])] = X_BS_test[:, iBS:iBS + 1]

X_BS1 = X_BS;
X_BS_test1 = X_BS_test
X_BS = Z_UE * 0.0;
X_BS_test = Z_UE_test * 0.0

A_BS_UE = A_BS_UE * 1e3
A_BS_UE_test = A_BS_UE_test * 1e3

x_1 = tf.placeholder('float', [None, nUE, nUE, 1])
x_2 = tf.placeholder('float', [None, nUE, nUE, 1])
x_3 = tf.placeholder('float', [None, nUE, nUE, 1])
index_1 = tf.placeholder('float', [None, nUE, nUE, 1])
index_2 = tf.placeholder('float', [None, nUE, nUE, 1])
index_3 = tf.placeholder('float', [None, nUE, nUE, 1])
a_bs_ue = tf.placeholder('float', [None, nUE, nUE, 1])
keep_prob_1 = tf.placeholder(tf.float32)
keep_prob_2 = tf.placeholder(tf.float32)
# ==========================stack of multiple GNN layers============================ #

x_1_output, x_2_output, x_3_output = Edge_GNN_kp(nUE, x_1, x_2, x_3, index_1, index_2, index_3,
                                                 n_hidden + [n_output], keep_prob_1, keep_prob_2,
                                                 transfer_function=transfer_function,
                                                 pooling_function=tf.reduce_mean,
                                                 name='0', is_test=True, is_transfer=True, is_BN=True)
x_1_output = tf.reduce_sum(x_1_output, axis=1)
y_bs_pred0 = x_1_output
y_ue_pred = tf.nn.sigmoid(x_1_output)

# y_ue_pred = tf.exp(y_ue_pred0)
for iBS in range(len(nUE_BS)):
    if iBS == 0:
        sum_y = tf.ones([1, nUE_BS[iBS], 1]) * tf.reduce_sum(y_ue_pred[:, sum(nUE_BS[0:iBS]):sum(nUE_BS[0:iBS + 1]), :],
                                                             axis=1, keepdims=True)
        sum_y_bs = tf.ones([1, nUE_BS[iBS], 1]) * tf.nn.sigmoid(tf.reduce_sum(y_bs_pred0[:, sum(nUE_BS[0:iBS]):
                                                                                            sum(nUE_BS[0:iBS + 1]), :],
                                                                              axis=1, keepdims=True))
    else:
        temp = tf.ones([1, nUE_BS[iBS], 1]) * tf.reduce_sum(y_ue_pred[:, sum(nUE_BS[0:iBS]):sum(nUE_BS[0:iBS + 1]), :],
                                                            axis=1, keepdims=True)
        sum_y = tf.concat((sum_y, temp), axis=1)
        sum_y_bs = tf.concat((sum_y_bs, tf.ones([1, nUE_BS[iBS], 1]) *
                              tf.nn.sigmoid(tf.reduce_sum(y_bs_pred0[:, sum(nUE_BS[0:iBS]):
                                                                        sum(nUE_BS[0:iBS + 1]), :],
                                                          axis=1, keepdims=True))), axis=1)

y_ue_pred = y_ue_pred / (sum_y + 1e-5) * sum_y_bs

# =================================================================================== #

chl = tf.transpose(a_bs_ue[:, :, :, 0], perm=[0, 2, 1]) / 1e3
chl2 = chl * chl
eye = tf.expand_dims(tf.eye(sum(nUE_BS)), axis=0)
power = tf.expand_dims(y_ue_pred[:, :, 0], axis=1)
cost = -1.0 * tf.reduce_mean((tf.reduce_sum(tf.log(1 + tf.reduce_sum(eye * chl2 * power, axis=2) /
                                                   (tf.reduce_sum((1 - eye) * chl2 * power, axis=2) + var_noise)),
                                            axis=1)))

optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, 0.9).minimize(cost)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

min_cost = 2020
min_epoch = 0
nnrate = 0.0
wmrate = np.sum(SR_test)
if is_plot is True:
    TEST_COST = np.ones(MAX_EPOCHS) * np.nan
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    # saver.restore(sess, model_location)
    Ratio = []
    for epoch in range(MAX_EPOCHS):
        total_batch = int(N_TRAIN / BATCH_SIZE)
        for _ in range(total_batch):
            start = time.time()
            batch_x_1, batch_x_2, batch_x_3, batch_index_1, batch_index_2, batch_index_3, batch_a_bs_ue = \
                get_random_block_from_data(X_1, X_2, X_3, Index_1, Index_2, Index_3, A_BS_UE, BATCH_SIZE)
            _, train_cost = sess.run([optimizer, cost], feed_dict={x_1: batch_x_1,
                                                                   x_2: batch_x_2,
                                                                   x_3: batch_x_3,
                                                                   index_1: batch_index_1,
                                                                   index_2: batch_index_2,
                                                                   index_3: batch_index_3,
                                                                   a_bs_ue: batch_a_bs_ue,
                                                                   keep_prob_1: 1.0,
                                                                   keep_prob_2: 0.5})
            time_cost = time_cost + time.time() - start

        test_cost = sess.run(cost, feed_dict={x_1: X_1_test, x_2: X_2_test, x_3: X_3_test, index_1: Index_1_test,
                                              index_2: Index_2_test, index_3: Index_3_test, a_bs_ue: A_BS_UE_test,
                                              keep_prob_1: 1.0, keep_prob_2: 1.0})

        if epoch % 1 == 0:
            Y_BS_pred_test = sess.run(y_ue_pred, feed_dict={x_1: X_1_test, x_2: X_2_test, x_3: X_3_test,
                                                            index_1: Index_1_test, index_2: Index_2_test,
                                                            index_3: Index_3_test, a_bs_ue: A_BS_UE_test,
                                                            keep_prob_1: 1.0, keep_prob_2: 1.0}) * Z_UE_test
            Y_BS_pred_test = np.reshape(Y_BS_pred_test, [N_TEST, -1])
            Y_BS_pred_test = power_nomalize(Y_BS_pred_test, X_BS_test1[:, :, 0], nUE_BS)
            nnrate, _ = cal_sum_rate1(A_BS_UE_test / 1e3, Y_BS_pred_test, var_noise, nUE_BS, nTX_BS, access_UE_BS)
            Ratio.append(nnrate)
            sio.savemat('Result/EdgeGNN.mat', {'ratio': np.array(Ratio), 'wmrate': wmrate,
                                               'min_cost': min_cost, 'min_epoch': min_epoch})
            if epoch == 0:
                start_time = time.time()
            print('Epoch:', '%04d' % (epoch + 1), 'Train cost', '%04f' % train_cost, 'Test cost', '%04f' % test_cost,
                  'Performance Ratio: ', '%04f' % (nnrate / wmrate * 100), '%', 'sum time:', time.time() - start_time)
        if is_plot is True:
            TEST_COST[epoch] = abs(test_cost)
        if test_cost <= min_cost:
            min_cost = test_cost
            min_epoch = epoch

    start = time.time()
    Y_BS_pred_test = sess.run(y_ue_pred, feed_dict={x_1: X_1_test, x_2: X_2_test, x_3: X_3_test,
                                                    index_1: Index_1_test, index_2: Index_2_test,
                                                    index_3: Index_3_test, a_bs_ue: A_BS_UE_test,
                                                    keep_prob_1: 1.0, keep_prob_2: 1.0}) * Z_UE_test
    test_time = time.time() - start
    Y_BS_pred_test = np.reshape(Y_BS_pred_test, [N_TEST, -1])
    Y_BS_pred_test = power_nomalize(Y_BS_pred_test, X_BS_test1[:, :, 0], nUE_BS)
    Y_BS_test1 = np.reshape(Y_UE_test, [N_TEST, -1])
    access_UE_BS = access_UE_BS.eval()
    nnrate, _ = cal_sum_rate1(A_BS_UE_test / 1e3, Y_BS_pred_test, var_noise, nUE_BS, nTX_BS, access_UE_BS)
    wmrate, SR_test1 = cal_sum_rate1(A_BS_UE_test / 1e3, Y_BS_test1, var_noise, nUE_BS, nTX_BS, access_UE_BS)
    ratio = nnrate / wmrate
    Ratio.append(ratio)
    sio.savemat('Result/EdgeGNN.mat', {'ratio': np.array(Ratio), 'wmrate': wmrate,
                                       'min_cost': min_cost, 'min_epoch': min_epoch})
