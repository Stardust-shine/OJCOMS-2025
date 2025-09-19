import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf
import numpy as np
from GNN_model import gnn_pe_layer_new, get_random_block_from_data1, cal_unspv_cost
from powerallocation_wmmse import gen_sample, cal_sum_rate1, power_nomalize
import time
import json
import scipy.io as sio

is_plot = False
N_TRAIN = 10000
LEARNING_RATE = 0.01
BATCH_SIZE = 64
MAX_EPOCHS = 100
N_TEST = 1000
model_location = "./DNNmodel/VertexGNN/model_demo.ckpt"
ini_model_location = "./InitialModel/PEGNN/model_demo.ckpt"
min_per = 0.9
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

activation_function = tf.nn.tanh
activation_function_fnn = tf.nn.relu

rau = 1 / 0.311
C = 43.3
r_min = 1


time_cost = 0

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

trainfilename = 'Dataset_new/Train_' + str(n_marco) + 'MBS_' + str(nUE_marco) + 'UE_' + \
                str(n_pico) + 'PBS_' + str(nUE_pico) + 'UEA1_weight_fair.mat'
testfilename = 'Dataset_new/Test_' + str(n_marco) + 'MBS_' + str(nUE_marco) + 'UE_' + \
               str(n_pico) + 'PBS_' + str(nUE_pico) + 'UEA1_weight_fair.mat'


trfile = sio.loadmat(trainfilename); tefile = sio.loadmat(testfilename)

A_BS_UE = trfile['A_BS_UE']; X_BS = trfile['X_BS']; X_UE = trfile['X_UE'];
Y_UE = trfile['Y_BS']; SR = trfile['SR']

A_BS_UE = A_BS_UE[0: N_TRAIN]; X_BS = X_BS[0: N_TRAIN]; X_UE = X_UE[0: N_TRAIN];
Y_UE = Y_UE[0: N_TRAIN]; SR = SR[0, 0:N_TRAIN]

A_BS_UE_test = tefile['A_BS_UE']; X_BS_test = tefile['X_BS']; X_UE_test = tefile['X_UE'];
Y_UE_test = tefile['Y_BS']; SR_test = tefile['SR']

A_BS_UE_test = A_BS_UE_test[0: N_TEST]; X_BS_test = X_BS_test[0: N_TEST]; X_UE_test = X_UE_test[0: N_TEST];
Y_UE_test = Y_UE_test[0: N_TEST]; SR_test = SR_test[0, 0: N_TEST]
SR_UE_test = tefile['SR_UE']
SR_UE_test = SR_UE_test[0: N_TEST]

Z_UE = np.zeros([N_TRAIN, nUE, 1]); Z_UE_test = np.zeros([N_TEST, nUE, 1])
for iBS in range(len(nUE_BS)):
    Z_UE[:, sum(nUE_BS[0:iBS]): sum(nUE_BS[0:iBS+1])] = X_BS[:, iBS:iBS+1]
    Z_UE_test[:, sum(nUE_BS[0:iBS]): sum(nUE_BS[0:iBS + 1])] = X_BS_test[:, iBS:iBS + 1]

X_BS1 = X_BS; X_BS_test1 = X_BS_test
X_BS = Z_UE * 0.0; X_BS_test = Z_UE_test * 0.0

A_BS_UE = A_BS_UE * 1e3
A_BS_UE_test = A_BS_UE_test * 1e3

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

# y_ue_pred = tf.exp(y_ue_pred0)
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
weight = tf.square(tf.matrix_diag_part(a_bs_ue[:, :, :, 0]) / 1e3)  # [batch, nUE]
weight = tf.div(1., tf.abs(tf.log(weight * p_pico / var_noise)/tf.log(2.)))  # [batch, nUE]

chl = tf.transpose(a_bs_ue[:, :, :, 0], perm=[0, 2, 1]) / 1e3

chl2 = chl * chl
eye = tf.expand_dims(tf.eye(sum(nUE_BS)), axis=0)
power = tf.expand_dims(y_ue_pred[:, :, 0], axis=1)

rate_user = tf.log(1 + tf.reduce_sum(eye * chl2 * power, axis=2) / (tf.reduce_sum((1 - eye) * chl2 * power, axis=2)
                                                                    + var_noise))
rate_user_weight = tf.multiply(rate_user, weight)

cost = -1.0 * tf.reduce_mean((tf.reduce_sum(rate_user_weight, axis=1)))
# cost = -1.0 * tf.reduce_mean((tf.reduce_sum(tf.log(1 + tf.reduce_sum(eye * chl2 * power, axis=2) /
#                                     (tf.reduce_sum((1 - eye) * chl2 * power, axis=2) + var_noise)), axis=1)))

# cost = tf.reduce_mean(tf.square(y_ue_pred - y_ue))

optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, 0.9).minimize(cost)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

min_cost = 2020
min_epoch = 0
nnrate = 0.0
wmrate = np.sum(SR_test)
if is_plot is True:
    TEST_COST = np.ones(MAX_EPOCHS) * np.nan
    # fig, ax = plt.subplots(figsize=[15, 5])
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    # saver.restore(sess, model_location)
    Ratio = []
    for epoch in range(MAX_EPOCHS):
        if epoch == 0:
            start_time = time.time()
        total_batch = int(N_TRAIN/BATCH_SIZE)
        for t_b in range(total_batch):
            start = time.time()
            batch_x_bs, batch_x_ue, batch_a_bs_ue, batch_y_bs = get_random_block_from_data1(X_BS, X_UE, A_BS_UE,
                                                                                            Y_UE/Z_UE, BATCH_SIZE)
            _, train_cost = sess.run([optimizer, cost], feed_dict={x_bs: batch_x_bs,
                                                                   x_ue: batch_x_ue,
                                                                   a_bs_ue: batch_a_bs_ue,
                                                                   y_ue: batch_y_bs})
            time_cost = time_cost + time.time() - start

            # if t_b % 5 == 0:
            #     Y_BS_pred_test = sess.run(y_ue_pred, feed_dict={x_bs: X_BS_test, x_ue: X_UE_test,
            #                                                     a_bs_ue: A_BS_UE_test,
            #                                                     y_ue: Y_UE_test / Z_UE_test}) * Z_UE_test
            #     Y_BS_pred_test = np.reshape(Y_BS_pred_test, [N_TEST, -1])
            #     Y_BS_pred_test = power_nomalize(Y_BS_pred_test, X_BS_test1[:, :, 0], nUE_BS)
            #     nnrate, rate_user = cal_sum_rate1(A_BS_UE_test / 1e3, Y_BS_pred_test, var_noise, nUE_BS, nTX_BS,
            #                                       access_UE_BS)
            #     print('Epoch:', '%04d' % (epoch + 1),
            #           'mini-Epoch', '%04d' % (t_b + 1),
            #           'Performance Ratio: ', '%04f' % (nnrate / wmrate * 100), '%',
            #           'sum time:', time.time() - start_time)

        test_cost = sess.run(cost, feed_dict={x_bs: X_BS_test, x_ue: X_UE_test,
                                              a_bs_ue: A_BS_UE_test, y_ue: Y_UE_test/Z_UE_test})
        if epoch % 1 == 0:
            Y_BS_pred_test = sess.run(y_ue_pred, feed_dict={x_bs: X_BS_test, x_ue: X_UE_test,
                                                            a_bs_ue: A_BS_UE_test,
                                                            y_ue: Y_UE_test / Z_UE_test}) * Z_UE_test
            weight_test = sess.run(weight, feed_dict={x_bs: X_BS_test, x_ue: X_UE_test,
                                                      a_bs_ue: A_BS_UE_test,
                                                      y_ue: Y_UE_test / Z_UE_test})
            Y_BS_pred_test = np.reshape(Y_BS_pred_test, [N_TEST, -1])
            Y_BS_pred_test = power_nomalize(Y_BS_pred_test, X_BS_test1[:, :, 0], nUE_BS)
            nnrate, rate_user = cal_sum_rate1(A_BS_UE_test / 1e3, Y_BS_pred_test, var_noise, nUE_BS, nTX_BS, access_UE_BS)

            Jain = np.mean(np.divide(np.square(np.sum(rate_user, axis=1)), nUE * np.sum(np.square(rate_user), axis=1)))
            EE = np.mean(np.divide(np.sum(rate_user, axis=1), rau * np.sum(Y_BS_pred_test, axis=1) + C))
            ratio_UE = np.sum(rate_user > r_min) / np.sum(np.ones(np.shape(rate_user)))
            rate_mean_user = np.mean(np.mean(rate_user, axis=1))

            rate_UE_learned_low = np.partition(rate_user, int(nUE * 0.05), axis=1)[:, :int(nUE * 0.05)]
            rate_UE_wmmse_low = np.partition(SR_UE_test, int(nUE * 0.05), axis=1)[:, :int(nUE * 0.05)]

            rate_UE_learned_low_mean = np.mean(np.sum(rate_UE_learned_low, axis=-1))
            rate_UE_wmmse_low_mean = np.mean(np.sum(rate_UE_wmmse_low, axis=-1))

            WSR_wmmse = np.mean(np.sum(np.multiply(weight_test, SR_UE_test), axis=-1))
            WSR_learning = np.mean(np.sum(np.multiply(weight_test, rate_user), axis=-1))

            Ratio.append(nnrate)
            sio.savemat('Result/PGNN.mat', {'ratio': np.array(Ratio), 'wmrate': wmrate,
                                            'min_cost': min_cost, 'min_epoch': min_epoch})

            print('Epoch:', '%04d' % (epoch + 1), 'Train cost', '%04f' % train_cost, 'Test cost', '%04f' % test_cost,
                  'Performance Ratio WSE: ', '%04f' % (WSR_learning / WSR_wmmse * 100), '%',
                  'WSE_test:', WSR_learning,
                  'Performance Ratio Low: ', '%04f' % (rate_UE_learned_low_mean / rate_UE_wmmse_low_mean * 100), '%',
                  'sum time:', time.time() - start_time)
        if is_plot is True:
            TEST_COST[epoch] = abs(test_cost)
            # plot_cost(TEST_COST, ax, test_cost)
        if test_cost <= min_cost:
            min_cost = test_cost
            min_epoch = epoch

    # saver.restore(sess, model_location)
    # for i in range(N_TEST):
    #     start = time.time()
    #     Y_BS_pred_test = sess.run(y_ue_pred, feed_dict={x_bs: X_BS_test[i:i+1], x_ue: X_UE_test[i:i+1],
    #                                                     a_bs_ue: A_BS_UE_test[i:i+1],
    #                                                     y_ue: Y_UE_test[i:i+1] / Z_UE_test[i:i+1]}) * Z_UE_test[i:i+1]
    #     print('Inference time:', time.time() - start)
    start = time.time()
    Y_BS_pred_test = sess.run(y_ue_pred, feed_dict={x_bs: X_BS_test, x_ue: X_UE_test,
                                                    a_bs_ue: A_BS_UE_test,
                                                    y_ue: Y_UE_test/Z_UE_test}) * Z_UE_test
    test_time = time.time() - start
    Y_BS_pred_test = np.reshape(Y_BS_pred_test, [N_TEST, -1])
    # Y_BS_pred_test = np.round(Y_BS_pred_test)
    Y_BS_pred_test = power_nomalize(Y_BS_pred_test, X_BS_test1[:, :, 0], nUE_BS)
    Y_BS_test1 = np.reshape(Y_UE_test, [N_TEST, -1])
    access_UE_BS = access_UE_BS.eval()
    nnrate, _ = cal_sum_rate1(A_BS_UE_test/1e3, Y_BS_pred_test, var_noise, nUE_BS, nTX_BS, access_UE_BS)
    wmrate, SR_test1 = cal_sum_rate1(A_BS_UE_test/1e3, Y_BS_test1, var_noise, nUE_BS, nTX_BS, access_UE_BS)
    ratio = nnrate / wmrate
    Ratio.append(ratio)
    sio.savemat('Result/PGNN.mat', {'ratio': np.array(Ratio), 'wmrate': wmrate,
                                    'min_cost': min_cost, 'min_epoch': min_epoch})
