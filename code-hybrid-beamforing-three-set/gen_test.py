import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from importlib import import_module
from scipy.io import loadmat
import time
import os
import scipy.io as sio


def test(test_params, run_id):
    NN_type = test_params['NN_type']
    NN = None
    if NN_type == 'VertexGNN':
        NN = import_module('VertexGNN').VertexGNN
    elif NN_type == 'EdgeGNN':
        NN = import_module('EdgeGNN').EdgeGNN
    elif NN_type == 'FNN':
        NN = import_module('FNN').FNN
    elif NN_type == 'CNN':
        NN = import_module('CNN').CNN
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
    Ncl_train = test_params['Ncl_train']
    Nray_train = test_params['Nray_train']
    model_dir = "./result/K" + str(num_users_train) + "_N" + str(user_num_an_train) + "X"+ str(bs_num_an_train) + \
                "_Ncl" + str(Ncl_train) + "_Nray" + str(Nray_train)

    test_dir = str(num_users) + "_N" + str(user_num_an) + "X" + str(bs_num_an) + "_Ncl" + str(Ncl) + "_Nray" + str(Nray)

    # Load data
    data_test = loadmat("./data/setH_K" + test_dir + "_number" + str(10000) + ".mat")['H'].astype(np.float32)
    data_test = np.transpose(data_test, axes=(0, 2, 3, 1))

    # Save test results
    save_dir = model_dir
    saver = tf.train.Saver(tf.global_variables())
    if not os.path.exists(save_dir + "/test/" + test_dir):
        os.makedirs(save_dir + "/test/" + test_dir)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        network.initialize(sess)
        model = tf.train.latest_checkpoint(model_dir + '/checkpoints/' + NN_type + '/')
        saver.restore(sess, model)
        sum_rate = network.get_rate(sess, data_test)
        sum_rate_mean = np.mean(sum_rate)
        print('sum_rate_test:', sum_rate_mean)







