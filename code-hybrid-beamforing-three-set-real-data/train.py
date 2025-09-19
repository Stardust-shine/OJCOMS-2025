import tensorflow.compat.v1 as tf
import h5py
import numpy as np
tf.disable_eager_execution()
from importlib import import_module
from scipy.io import loadmat
import time
import os
import scipy.io as sio
from gen_data.gen_dataset import generate_transform
from help import norm_data


def train(train_params, run_id):
    NN_type = train_params['NN_type']
    NN = None
    if NN_type == 'VertexGNN':
        NN = import_module('VertexGNN').VertexGNN
    elif NN_type == 'EdgeGNN':
        NN = import_module('EdgeGNN').EdgeGNN
    elif NN_type == 'Hybrid_GNN':
        NN = import_module('Hybrid_GNN').Hybrid_GNN

    network = NN(train_params)
    bs_num_an = train_params['bs_num_an']
    bs_num_rf = train_params['bs_num_rf']

    num_users = train_params['num_users']
    Ncl = train_params['Ncl']
    Nray = train_params['Nray']
    batch = train_params['batch_size']
    epoch = train_params['epoch']
    saver_threshold = train_params['saver_threshold']

    if (NN_type == 'EdgeGNN_QoS') or (NN_type == 'VertexGNN_QoS'):
        r_min = train_params['r_min']

    num_train = train_params['num_train']
    num_test = train_params['num_test']

    file_dir = str(num_users) + "_N" + str(bs_num_an) + "_Ncl" + str(Ncl) + "_Nray" + str(Nray)

    # Load data
    train_path = "./data/setH_K" + file_dir + '_train' + ".mat"
    test_path = "./data/setH_K" + file_dir + '_test' + ".mat"

    with h5py.File(train_path, 'r') as f:
        data_train = f['H'][:]
        data_train = np.transpose(data_train, axes=(0, 2, 3, 1))
        data_train = data_train[0:num_train, :, :, :]

        shade_mat = f['shade'][:]
        if len(np.shape(shade_mat)) == 2:
            shade_mat = np.expand_dims(shade_mat[0:num_train, :], 2)
        else:
            shade_mat = shade_mat[0:num_train, :, :]

        n_UE_train = f['num_users'][:]
        n_UE_train = n_UE_train[0:num_train, :]

    with h5py.File(test_path, 'r') as f:
        data_test = f['H'][:]
        data_test = np.transpose(data_test, axes=(0, 2, 3, 1))
        data_test = data_test[0:num_test, :, :, :]

        shade_mat_0 = f['shade'][:]
        if len(np.shape(shade_mat_0)) == 2:
            shade_mat_0 = np.expand_dims(shade_mat_0[0:num_test, :], 2)
        else:
            shade_mat_0 = shade_mat_0[0:num_test, :, :]

        n_UE_test = f['num_users'][:]
        n_UE_test = n_UE_test[0:num_test, :]

    data_train_norm, data_test_norm = norm_data(data_train, data_test)

    # Save results
    save_dir = "./result/K" + file_dir
    saver = tf.train.Saver(max_to_keep=5)  # Model saver
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    sum_rate_record = []
    sum_rate_qos_record = []
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        network.initialize(sess)

        if (NN_type == 'EdgeGNN') or (NN_type == 'Hybrid_GNN'):
            initial_1 = train_params['initial_1']
            initial_2 = train_params['initial_2']
            # Initialize the feature of each edge - train
            edge_frf = np.tile(np.reshape(np.linspace(initial_1[0], initial_1[1], bs_num_an * bs_num_rf),
                                          [1, bs_num_an, bs_num_rf, 1]), (batch, 1, 1, 2))
            edge_fbb = np.tile(np.reshape(np.linspace(initial_2[0], initial_2[1], num_users * bs_num_rf),
                                          [1, num_users, bs_num_rf, 1]), (batch, 1, 1, 2))
            # Initialize the feature of each edge - test
            edge_frf_0 = np.tile(np.reshape(np.linspace(initial_1[0], initial_1[1], bs_num_an * bs_num_rf),
                                            [1, bs_num_an, bs_num_rf, 1]), (num_test, 1, 1, 2))
            edge_fbb_0 = np.tile(np.reshape(np.linspace(initial_2[0], initial_2[1], num_users * bs_num_rf),
                                            [1, num_users, bs_num_rf, 1]), (num_test, 1, 1, 2))

        sum_rate_max = 0
        noise_matrix = np.random.rand(1, 2, bs_num_an, bs_num_rf)
        for e in range(epoch):
            # Train
            start_time = time.time()
            num_iter = int(np.floor(np.shape(data_train_norm)[0] / batch))
            for ite in range(num_iter):
                sample = data_train[ite * batch:(ite + 1) * batch, :, :, :]
                sample_norm = data_train_norm[ite * batch:(ite + 1) * batch, :, :, :]
                UE_sample = n_UE_train[ite * batch:(ite + 1) * batch, :]
                shade_sample = shade_mat[ite * batch:(ite + 1) * batch, :, :]
                if (NN_type == 'EdgeGNN') or (NN_type == 'Hybrid_GNN'):
                    sum_rate_train = - network.learn_batch(sess, sample, sample_norm, edge_frf, edge_fbb)
                else:
                    sum_rate_train = - network.learn_batch(sess, sample, sample_norm)

            # Test
            if NN_type == 'EdgeGNN':
                sum_rate = network.get_rate(sess, data_test, data_test_norm, edge_frf_0, edge_fbb_0)
            elif NN_type == 'Hybrid_GNN':
                sum_rate = network.get_rate(sess, data_test, data_test_norm, edge_frf_0, edge_fbb_0)
            else:
                sum_rate = network.get_rate(sess, data_test)

            # Result Print
            sum_rate_mean = np.mean(sum_rate)
            sum_rate_record.append(sum_rate_mean)
            print('run_id:', run_id, 'epoch:', e, 'sum_rate_train:', sum_rate_train, 'sum-rate_test:',
                  sum_rate_mean, 'time_epoch:', time.time() - start_time)

            # Save
            sio.savemat(save_dir + "/result.mat", {'rate_mean': sum_rate_record, 'rate_sample': sum_rate})
            # if sum_rate_max < sum_rate_mean:
            #     sum_rate_max = sum_rate_mean
            #     if not os.path.exists(save_dir + '/checkpoints/' + NN_type):
            #         os.makedirs(save_dir + '/checkpoints/' + NN_type)
            #     saver.save(sess, save_dir + '/checkpoints/' + NN_type + '/model.ckpt')
