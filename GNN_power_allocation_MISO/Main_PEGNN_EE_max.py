import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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
MAX_EPOCHS = 5000
N_TEST = 128
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

activation_function = tf.nn.relu
activation_function_fnn = tf.nn.relu

rau = 1 / 0.311
C = 43.3
r_min = 0.1

lamda_learning = True

lamda_constant = 2

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
                str(n_pico) + 'PBS_' + str(nUE_pico) + 'UEA1.mat'
testfilename = 'Dataset_new/Test_' + str(n_marco) + 'MBS_' + str(nUE_marco) + 'UE_' + \
               str(n_pico) + 'PBS_' + str(nUE_pico) + 'UEA1.mat'


trfile = sio.loadmat(trainfilename); tefile = sio.loadmat(testfilename)

A_BS_UE = trfile['A_BS_UE']; X_BS = trfile['X_BS']; X_UE = trfile['X_UE'];
Y_UE = trfile['Y_BS']; SR = trfile['SR']

A_BS_UE = A_BS_UE[0: N_TRAIN]; X_BS = X_BS[0: N_TRAIN]; X_UE = X_UE[0: N_TRAIN];
Y_UE = Y_UE[0: N_TRAIN]; SR = SR[0, 0:N_TRAIN]

A_BS_UE_test = tefile['A_BS_UE']; X_BS_test = tefile['X_BS']; X_UE_test = tefile['X_UE'];
Y_UE_test = tefile['Y_BS']; SR_test = tefile['SR']
SR_UE_test = tefile['SR_UE']

A_BS_UE_test = A_BS_UE_test[0: N_TEST]; X_BS_test = X_BS_test[0: N_TEST]; X_UE_test = X_UE_test[0: N_TEST];
Y_UE_test = Y_UE_test[0: N_TEST]; SR_test = SR_test[0, 0: N_TEST]
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
                            transfer_function_fnn=activation_function_fnn, name='Vertex_gnn_bs' + str(l))
    h_ue = gnn_pe_layer_new(h_ue, h, tf.transpose(a_bs_ue, perm=[0, 2, 1, 3]),
                            h_bs, access_BS_UE, transfer_function=activation_function,
                            transfer_function_fnn=activation_function_fnn, name='Vertex_gnn_ue' + str(l))

y_bs_pred0 = gnn_pe_layer_new(h_bs, n_output, '', h_ue, access_UE_BS, transfer_function_fnn=activation_function_fnn,
                              name='Vertex_gnn_out_bs', is_transfer=False)
y_ue_pred0 = gnn_pe_layer_new(h_ue, n_output, '', h_bs, access_BS_UE, transfer_function_fnn=activation_function_fnn,
                              name='Vertex_gnn_out_ue', is_transfer=False)

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

if lamda_learning:
    h_bs_lamda = x_bs
    h_ue_lamda = x_ue
    for l, h in enumerate(n_hidden):
        h_bs_lamda = gnn_pe_layer_new(h_bs_lamda, h, a_bs_ue, h_ue_lamda, access_UE_BS,
                                      transfer_function=activation_function,
                                      transfer_function_fnn=activation_function_fnn, name='lamda_bs' + str(l))
        h_ue_lamda = gnn_pe_layer_new(h_ue_lamda, h, tf.transpose(a_bs_ue, perm=[0, 2, 1, 3]),
                                      h_bs_lamda, access_BS_UE, transfer_function=activation_function,
                                      transfer_function_fnn=activation_function_fnn, name='lamda_ue' + str(l))

    y_bs_pred0_lamda = gnn_pe_layer_new(h_bs_lamda, n_output, '', h_ue_lamda, access_UE_BS,
                                        transfer_function_fnn=activation_function_fnn,
                                        name='lamda_out_bs', is_transfer=False)
    y_ue_pred0_lamda = gnn_pe_layer_new(h_ue_lamda, n_output, '', h_bs_lamda, access_BS_UE,
                                        transfer_function_fnn=activation_function_fnn,
                                        name='lamda_out_ue', is_transfer=False)

    output_lamda = y_bs_pred0_lamda + y_ue_pred0_lamda
    lamda = tf.reduce_mean(output_lamda, axis=2)

# =================================================================================== #

chl = tf.transpose(a_bs_ue[:, :, :, 0], perm=[0, 2, 1]) / 1e3

chl2 = chl * chl
eye = tf.expand_dims(tf.eye(sum(nUE_BS)), axis=0)
power = tf.expand_dims(y_ue_pred[:, :, 0], axis=1)

rate_user = tf.log(1 + tf.reduce_sum(eye * chl2 * power, axis=2) / (tf.reduce_sum((1 - eye) * chl2 * power, axis=2) + var_noise))


if lamda_learning:
    cost = tf.reduce_mean(-1 * tf.reduce_sum(rate_user, axis=1)/(rau * tf.reduce_sum(y_ue_pred, axis=[1, 2]) + C) + \
           tf.reduce_sum(tf.nn.relu(lamda) * tf.nn.relu(r_min - rate_user / tf.math.log(2.0)), axis=1))
else:
    cost = tf.reduce_mean(-1 * tf.reduce_sum(rate_user, axis=1)/(rau * tf.reduce_sum(y_ue_pred, axis=[1, 2]) + C) + \
           tf.reduce_sum(lamda_constant * tf.nn.relu(r_min - rate_user / tf.math.log(2.0)), axis=1))
    # cost = tf.reduce_mean(tf.reduce_sum(tf.exp(tf.nn.relu(r_min - rate_user / tf.math.log(2.0))), axis=1))

optimizer_gnn = tf.train.RMSPropOptimizer(LEARNING_RATE, 0.9)
var_gnn = [var for var in tf.trainable_variables() if 'Vertex_gnn' in var.name]
grads_gnn = optimizer_gnn.compute_gradients(cost, var_gnn)
train_gnn = optimizer_gnn.apply_gradients(grads_gnn)

if lamda_learning:
    optimizer_lamda = tf.train.RMSPropOptimizer(LEARNING_RATE, 0.9)
    var_lamda = [var for var in tf.trainable_variables() if 'lamda' in var.name]
    grads_lamda = optimizer_lamda.compute_gradients(-1 * cost, var_lamda)
    train_lamda = optimizer_lamda.apply_gradients(grads_lamda)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

min_cost = 2020
min_epoch = 0
nnrate = 0.0

wmrate = np.sum(SR_test)
wmEE = np.mean(SR_test / (rau * np.sum(Y_UE_test, axis=1, keepdims=True) + C))
wm_ratio_UE = np.sum(SR_UE_test > r_min) / np.sum(np.ones(np.shape(SR_UE_test)))

if is_plot is True:
    TEST_COST = np.ones(MAX_EPOCHS) * np.nan
    # fig, ax = plt.subplots(figsize=[15, 5])
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    # saver.restore(sess, model_location)
    Ratio = []
    for epoch in range(MAX_EPOCHS):
        total_batch = int(N_TRAIN/BATCH_SIZE)
        for _ in range(total_batch):
            start = time.time()
            if epoch == 0:
                start_time = time.time()
            batch_x_bs, batch_x_ue, batch_a_bs_ue, batch_y_bs = get_random_block_from_data1(X_BS, X_UE, A_BS_UE,
                                                                                            Y_UE/Z_UE, BATCH_SIZE)
            if lamda_learning:
                _, _, train_cost = sess.run([train_gnn, train_lamda, cost], feed_dict={x_bs: batch_x_bs,
                                                                                       x_ue: batch_x_ue,
                                                                                       a_bs_ue: batch_a_bs_ue,
                                                                                       y_ue: batch_y_bs})
            else:
                _, train_cost = sess.run([train_gnn, cost], feed_dict={x_bs: batch_x_bs,
                                                                       x_ue: batch_x_ue,
                                                                       a_bs_ue: batch_a_bs_ue,
                                                                       y_ue: batch_y_bs})
            time_cost = time_cost + time.time() - start

        test_cost = sess.run(cost, feed_dict={x_bs: X_BS_test, x_ue: X_UE_test,
                                              a_bs_ue: A_BS_UE_test, y_ue: Y_UE_test/Z_UE_test})
        if epoch % 1 == 0:
            Y_BS_pred_test = sess.run(y_ue_pred, feed_dict={x_bs: X_BS_test, x_ue: X_UE_test,
                                                            a_bs_ue: A_BS_UE_test,
                                                            y_ue: Y_UE_test / Z_UE_test}) * Z_UE_test
            Y_BS_pred_test = np.reshape(Y_BS_pred_test, [N_TEST, -1])
            Y_BS_pred_test = power_nomalize(Y_BS_pred_test, X_BS_test1[:, :, 0], nUE_BS)
            nnrate, rate_user = cal_sum_rate1(A_BS_UE_test / 1e3, Y_BS_pred_test, var_noise, nUE_BS, nTX_BS, access_UE_BS)

            nnrate_constraint = np.sum(np.multiply((rate_user > r_min), rate_user))

            EE = np.mean(np.divide(np.sum(rate_user, axis=1), rau * np.sum(Y_BS_pred_test, axis=1) + C))
            ratio_UE = np.sum(rate_user > r_min) / np.sum(np.ones(np.shape(rate_user)))

            Ratio.append(nnrate)
            sio.savemat('Result/PGNN.mat', {'ratio': np.array(Ratio), 'wmrate': wmrate,
                                            'min_cost': min_cost, 'min_epoch': min_epoch})

            print('Epoch:', '%04d' % (epoch + 1), 'Train cost', '%04f' % train_cost, 'Test cost', '%04f' % test_cost,
                  'Performance Ratio: ', '%04f' % (nnrate / wmrate * 100), '%',
                  'Performance Ratio Constraint: ', '%04f' % (nnrate_constraint / wmrate * 100), '%'
                  'EE', '%04f' % (EE / wmEE * 100),
                  'ratio_UE', '%04f' % ratio_UE, '%', 'wMRatio UE: ', '%04f' % wm_ratio_UE,
                  'sum time:', time.time() - start_time)
        if is_plot is True:
            TEST_COST[epoch] = abs(test_cost)
            # plot_cost(TEST_COST, ax, test_cost)
        if test_cost <= min_cost:
            min_cost = test_cost
            min_epoch = epoch

    start = time.time()
    Y_BS_pred_test = sess.run(y_ue_pred, feed_dict={x_bs: X_BS_test, x_ue: X_UE_test,
                                                    a_bs_ue: A_BS_UE_test,
                                                    y_ue: Y_UE_test/Z_UE_test}) * Z_UE_test
    test_time = time.time() - start
    Y_BS_pred_test = np.reshape(Y_BS_pred_test, [N_TEST, -1])
    Y_BS_pred_test = power_nomalize(Y_BS_pred_test, X_BS_test1[:, :, 0], nUE_BS)
    Y_BS_test1 = np.reshape(Y_UE_test, [N_TEST, -1])
    access_UE_BS = access_UE_BS.eval()
    nnrate, _ = cal_sum_rate1(A_BS_UE_test/1e3, Y_BS_pred_test, var_noise, nUE_BS, nTX_BS, access_UE_BS)
    wmrate, SR_test1 = cal_sum_rate1(A_BS_UE_test/1e3, Y_BS_test1, var_noise, nUE_BS, nTX_BS, access_UE_BS)
    ratio = nnrate / wmrate
    Ratio.append(ratio)
    sio.savemat('Result/PGNN.mat', {'ratio': np.array(Ratio), 'wmrate': wmrate,
                                    'min_cost': min_cost, 'min_epoch': min_epoch})
