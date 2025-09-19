import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
from PENN_model import penn, get_random_block_from_data, gen_index
# from PENN_model import penn_joint, penn_1d
from powerallocation_wmmse import gen_sample, cal_sum_rate1, power_nomalize
import time
import json
import scipy.io as sio

N_TRAIN = 300
is_plot = False
# LEARNING_RATE = float(sys.argv[3])
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MAX_EPOCHS = 10000
N_TEST = 500
model_location = "./DNNmodel/PENN/model_demo.ckpt"
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

n_hidden = [2, 2, 2]
n_output = 1
coefficient = 0.01
# 0.01
hidden_transfer = tf.nn.tanh
output_transfer = tf.nn.softmax

time_cost = 0

p_BS = np.concatenate([np.ones(n_pico) * p_pico, np.ones(n_marco) * p_marco])
nUE_BS = np.int64(np.concatenate([np.ones(n_pico) * nUE_pico, np.ones(n_marco) * nUE_marco]))
nTX_BS = np.int64(np.concatenate([np.ones(n_pico) * nTX_pico, np.ones(n_marco) * nTX_marco]))
nUE = sum(nUE_BS)
nBS = len(nUE_BS)

access_UE_BS = np.zeros([1, len(nUE_BS), sum(nUE_BS)])
for iBS in range(len(nUE_BS)):
    access_UE_BS[0, iBS, sum(nUE_BS[0:iBS]): sum(nUE_BS[0:iBS + 1])] = 1
access_BS_UE = tf.constant(np.transpose(access_UE_BS))
access_UE_BS = tf.constant(access_UE_BS)

# ================================================================================== #

trainfilename = 'Dataset_new/Train_' + str(n_marco) + 'MBS_' + str(nUE_marco) + 'UE_' + \
                str(n_pico) + 'PBS_' + str(nUE_pico) + 'UEA1.mat'
testfilename = 'Dataset_new/Test_' + str(n_marco) + 'MBS_' + str(nUE_marco) + 'UE_' + \
               str(n_pico) + 'PBS_' + str(nUE_pico) + 'UEA1.mat'

trfile = sio.loadmat(trainfilename);
tefile = sio.loadmat(testfilename)
A_BS_UE = trfile['A_BS_UE'];
X_BS = trfile['X_BS'];
X_UE = trfile['X_UE'];
Y_BS = trfile['Y_BS'];
SR = trfile['SR']
A_BS_UE_test = tefile['A_BS_UE'];
X_BS_test = tefile['X_BS'];
X_UE_test = tefile['X_UE'];
Y_BS_test = tefile['Y_BS'];
SR_test = tefile['SR']

# ================================================================================== #

A_BS_UE = A_BS_UE[0: N_TRAIN];
X_BS = X_BS[0: N_TRAIN];
X_UE = X_UE[0: N_TRAIN];
Y_BS = Y_BS[0: N_TRAIN];
SR = SR[0, 0:N_TRAIN]
A_BS_UE_test = A_BS_UE_test[0: N_TEST];
X_BS_test = X_BS_test[0: N_TEST];
X_UE_test = X_UE_test[0: N_TEST];
Y_BS_test = Y_BS_test[0: N_TEST];
SR_test = SR_test[0, 0: N_TEST]

X = np.zeros([A_BS_UE.shape[0], n_pico * n_pico * nUE_pico, nUE_pico])
for i in range(n_pico):
    for j in range(n_pico):
        X[:, i * n_pico * nUE_pico + j * nUE_pico:i * n_pico * nUE_pico + (j + 1) * nUE_pico, :] = \
            np.squeeze(A_BS_UE[:, i * nUE_pico:(i + 1) * nUE_pico, j * nUE_pico:(j + 1) * nUE_pico])
X = np.reshape(X, [A_BS_UE.shape[0], -1])
X_BS = np.reshape(X_BS, [X_BS.shape[0], -1])
Y = np.reshape(Y_BS, [Y_BS.shape[0], -1])

X_test = np.zeros([A_BS_UE_test.shape[0], n_pico * n_pico * nUE_pico, nUE_pico])
for i in range(n_pico):
    for j in range(n_pico):
        X_test[:, i * n_pico * nUE_pico + j * nUE_pico:i * n_pico * nUE_pico + (j + 1) * nUE_pico, :] = \
            np.squeeze(A_BS_UE_test[:, i * nUE_pico:(i + 1) * nUE_pico, j * nUE_pico:(j + 1) * nUE_pico])
X_test = np.reshape(X_test, [A_BS_UE_test.shape[0], -1])
X_BS_test = np.reshape(X_BS_test, [X_BS_test.shape[0], -1])
Y_test = np.reshape(Y_BS_test, [Y_BS_test.shape[0], -1])

Z_UE = np.zeros([N_TRAIN, nUE]);
Z_UE_test = np.zeros([N_TEST, nUE])
for iBS in range(len(nUE_BS)):
    Z_UE[:, sum(nUE_BS[0:iBS]): sum(nUE_BS[0:iBS + 1])] = X_BS[:, iBS:iBS + 1]
    Z_UE_test[:, sum(nUE_BS[0:iBS]): sum(nUE_BS[0:iBS + 1])] = X_BS_test[:, iBS:iBS + 1]

A_BS_UE = A_BS_UE * 1e3
A_BS_UE_test = A_BS_UE_test * 1e3
# ================================================================================== #

x = tf.placeholder('float', [None, nUE * nUE])
y = tf.placeholder('float', [None, nUE])

# ===================================== PENN ======================================= #
batch = tf.shape(x)[0]
index_w_1, index_b_1 = gen_index(n_pico)
index_w_2, index_b_2 = gen_index(nUE_pico)

layernum = n_hidden + [n_output]
y_pred0 = penn(n_pico, nUE_pico, x, layernum, index_w_1, index_b_1, index_w_2, index_b_2, coefficient,
               hidden_transfer=hidden_transfer, keep_prob=1.0, name='0', is_transfer=True, is_BN=True,
               is_test=False)
y_pred0 = tf.reshape(y_pred0, [batch, n_pico * n_pico * nUE_pico, nUE_pico])
y_pred0 = tf.reshape(y_pred0, [batch, n_pico * n_pico, nUE_pico, nUE_pico])
y_pred0 = tf.linalg.diag_part(y_pred0)
index = np.arange(n_pico) * nUE_pico
y_pred0 = tf.gather(y_pred0, index, axis=1)
y_pred0 = tf.reshape(y_pred0, [batch, n_pico * nUE_pico])
y_pred = output_transfer(y_pred0)

for iBS in range(len(nUE_BS)):
    if iBS == 0:
        sum_y = tf.ones([1, nUE_BS[iBS]]) * tf.reduce_sum(y_pred[:, sum(nUE_BS[0:iBS]):sum(nUE_BS[0:iBS + 1])],
                                                          axis=1, keepdims=True)
        sum_y_bs = tf.ones([1, nUE_BS[iBS]]) * tf.nn.sigmoid(tf.reduce_sum(y_pred0[:, sum(nUE_BS[0:iBS]):
                                                                                      sum(nUE_BS[0:iBS + 1])],
                                                                           axis=1, keepdims=True))
    else:
        temp = tf.ones([1, nUE_BS[iBS]]) * tf.reduce_sum(y_pred[:, sum(nUE_BS[0:iBS]):sum(nUE_BS[0:iBS + 1])],
                                                         axis=1, keepdims=True)
        sum_y = tf.concat((sum_y, temp), axis=1)
        sum_y_bs = tf.concat((sum_y_bs, tf.ones([1, nUE_BS[iBS]]) * tf.nn.sigmoid(tf.reduce_sum(
            y_pred0[:, sum(nUE_BS[0:iBS]):
                       sum(nUE_BS[0:iBS + 1])],
            axis=1, keepdims=True))), axis=1)

y_pred = y_pred / (sum_y + 1e-5) * sum_y_bs

# =====================================无监督损失函数======================================= #
chl = tf.reshape(x, [-1, n_pico * n_pico, nUE_pico * nUE_pico])
chl = tf.reshape(chl, [-1, n_pico * n_pico, nUE_pico, nUE_pico])
chl = tf.reshape(chl, [-1, n_pico, n_pico, nUE_pico, nUE_pico])
chl = tf.transpose(chl, perm=[0, 1, 3, 2, 4])
chl = tf.transpose(tf.reshape(chl, [-1, nUE, nUE]), perm=[0, 2, 1])
chl2 = chl * chl
eye = tf.expand_dims(tf.eye(sum(nUE_BS)), axis=0)
power = tf.expand_dims(y_pred, axis=1)
cost = -1.0 * tf.reduce_mean((tf.reduce_sum(tf.log(1 + tf.reduce_sum(eye * chl2 * power, axis=2) /
                                                   (tf.reduce_sum((1 - eye) * chl2 * power, axis=2) + var_noise)),
                                            axis=1)))

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
# print(var)

min_cost = 2020
min_epoch = 0
nnrate = 0.0
wmrate = np.sum(SR_test)
if is_plot is True:
    TEST_COST = np.ones(MAX_EPOCHS) * np.nan
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    Ratio = []
    sess.run(init)
    # saver.restore(sess, model_location)
    for epoch in range(MAX_EPOCHS):
        total_batch = int(N_TRAIN / BATCH_SIZE)
        for _ in range(total_batch):
            start = time.time()
            batch_x, batch_y = get_random_block_from_data(X, Y / Z_UE, BATCH_SIZE)
            _, train_cost = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            time_cost = time_cost + time.time() - start

        test_cost = sess.run(cost, feed_dict={x: X_test, y: Y_test / Z_UE_test})
        if epoch % 1 == 0:
            Y_pred_test = sess.run(y_pred, feed_dict={x: X_test, y: Y_test / Z_UE_test}) * Z_UE_test
            Y_pred_test = np.reshape(Y_pred_test, [N_TEST, -1])
            Y_BS_pred_test = power_nomalize(Y_pred_test, X_BS_test, nUE_BS)
            nnrate, _ = cal_sum_rate1(A_BS_UE_test / 1e3, Y_pred_test, var_noise, nUE_BS, nTX_BS, access_UE_BS)
            Ratio.append(nnrate)
            sio.savemat('Result/PENN.mat', {'ratio': np.array(Ratio), 'wmrate': wmrate,
                                            'min_cost': min_cost, 'min_epoch': min_epoch})
            print('Epoch:', '%04d' % (epoch + 1), 'Train cost', '%04f' % train_cost, 'Test cost', '%04f' % test_cost,
                  'Performance Ratio: ', '%04f' % (nnrate / wmrate * 100), '%')
        if is_plot is True:
            TEST_COST[epoch] = abs(test_cost)
            # plot_cost(TEST_COST, ax)
        if test_cost <= min_cost:
            min_cost = test_cost
            min_epoch = epoch

    start = time.time()
    Y_pred_test = sess.run(y_pred, feed_dict={x: X_test, y: Y_test / Z_UE_test}) * Z_UE_test
    test_time = time.time() - start
    Y_pred_test = np.reshape(Y_pred_test, [N_TEST, -1])
    # Y_pred_test = np.round(Y_pred_test)
    Y_BS_pred_test = power_nomalize(Y_pred_test, X_BS_test, nUE_BS)
    Y_test1 = np.reshape(Y_test, [N_TEST, -1])
    access_UE_BS = access_UE_BS.eval()
    nnrate, _ = cal_sum_rate1(A_BS_UE_test[:, :, :, :] / 1e3, Y_BS_pred_test, var_noise, nUE_BS, nTX_BS, access_UE_BS)
    wmrate, SR_test1 = cal_sum_rate1(A_BS_UE_test[:, :, :, :] / 1e3, Y_test1, var_noise, nUE_BS, nTX_BS, access_UE_BS)
    ratio = nnrate / wmrate
    Ratio.append(nnrate)
    sio.savemat('Result/PENN.mat', {'ratio': np.array(Ratio), 'wmrate': wmrate,
                                    'min_cost': min_cost, 'min_epoch': min_epoch})
