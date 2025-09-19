import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from importlib import import_module
from scipy.io import loadmat
import time


def gen_test(test_params, *run_id):
    NN_type = test_params['NN_type']
    NN = None
    if NN_type == 'VertexGNN':
        NN = import_module('VertexGNN').VertexGNN
    elif NN_type == 'EdgeGNN':
        NN = import_module('EdgeGNN').EdgeGNN
    network = NN(test_params)

    bs_num_an = test_params['bs_num_an']
    bs_num_rf = test_params['bs_num_rf']
    num_users = test_params['num_users']
    user_num_an = test_params['user_num_an']
    Ncl = test_params['Ncl']
    Nray = test_params['Nray']

    bs_num_an_train = test_params['bs_num_an_train']
    bs_num_rf_train = test_params['bs_num_rf_train']
    num_users_train = test_params['num_users_train']
    user_num_an_train = test_params['user_num_an_train']

    model_dir = "./result/K" + str(num_users_train) + "_N" + str(user_num_an_train) + "X"+ str(bs_num_an_train) + \
                "_Ncl" + str(Ncl) + "_Nray" + str(Nray)

    test_dir = str(num_users) + "_N" + str(user_num_an) + "X" + str(bs_num_an) + "_Ncl" + str(Ncl) + "_Nray" + str(Nray)
    # Load data
    num_test = test_params['num_test']
    data_test = loadmat("./data/setH_K" + test_dir + "_number" + str(10000) + ".mat")['H'].astype(np.float32)
    data_test = np.transpose(data_test, axes=(0, 2, 3, 1))
    data_test = data_test[0:num_test, :, :, :]

    # Save test results
    saver = tf.train.Saver()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        network.initialize(sess)
        if NN_type == 'EdgeGNN':
            initial_1 = test_params['initial_1']
            initial_2 = test_params['initial_2']
            initial_3 = test_params['initial_3']
            edge_wrf = np.tile(np.reshape(np.linspace(initial_1[0], initial_1[1], num_users*user_num_an),
                                          [1, num_users, user_num_an, 1]), (num_test, 1, 1, 2))
            edge_frf = np.tile(np.reshape(np.linspace(initial_2[0], initial_2[1], bs_num_an*bs_num_rf),
                                          [1, bs_num_an, bs_num_rf, 1]), (num_test, 1, 1, 2))
            edge_fbb = np.tile(np.reshape(np.linspace(initial_3[0], initial_3[1], num_users*bs_num_rf),
                                          [1, num_users, bs_num_rf, 1]), (num_test, 1, 1, 2))

        model = model_dir + '/checkpoints/' + NN_type + '/model.ckpt'
        saver.restore(sess, model)
        # network.model_restore(sess, saver, model)

        sum_rate_all = []

        for i in range(num_test):

            start_time = time.time()
            if NN_type == 'EdgeGNN':
                sum_rate = network.get_rate(sess, [data_test[i]], [edge_wrf[i]], [edge_frf[i]], [edge_fbb[i]])
            else:
                sum_rate = network.get_rate(sess, [data_test[i]])

            sum_rate_all.append(sum_rate)
            print('time:', time.time() - start_time)
        sum_rate_mean = np.mean(np.array(sum_rate_all))
        print('sum_rate_test:', sum_rate_mean)







