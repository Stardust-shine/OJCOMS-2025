import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
from importlib import import_module
from scipy.io import loadmat
import time
import os
import scipy.io as sio
import h5py
from help import top_n_rows_per_block, norm_data


def train(train_params, run_id):
    NN_type = train_params['NN_type']
    NN = None
    if NN_type == 'VertexGNN':
        NN = import_module('VertexGNN').VertexGNN
    elif NN_type == 'EdgeGNN':
        NN = import_module('EdgeGNN').EdgeGNN
    elif NN_type == 'FNN':
        NN = import_module('FNN').FNN
    elif NN_type == 'CNN':
        NN = import_module('CNN').CNN
    elif NN_type == 'PENN':
        NN = import_module('PENN').PENN
    elif NN_type == 'PENN_new':
        NN = import_module('PENN_new').PENN_new
    elif NN_type == 'Flexible_VertexGNN':
        NN = import_module('Flexible_VertexGNN').Flexible_VertexGNN
    elif NN_type == 'MDGNN':
        NN = import_module('MDGNN').MDGNN
    elif NN_type == 'Att_MDGNN':
        NN = import_module('Att_MDGNN').Att_MDGNN
    elif NN_type == 'RGNN':
        NN = import_module('RGNN').RGNN
    elif NN_type == 'Att_EdgeGNN':
        NN = import_module('Att_EdgeGNN').Att_EdgeGNN
    elif NN_type == 'Hybrid_GNN':
        NN = import_module('Hybrid_GNN').Hybrid_GNN
    elif NN_type == 'Hybrid_GNN_new':
        NN = import_module('Hybrid_GNN_new').Hybrid_GNN_new
    elif NN_type == 'MDGNN_new':
        NN = import_module('MDGNN_new').MDGNN_new
    elif NN_type == 'Hybrid_GNN_v':
        NN = import_module('Hybrid_GNN_v').Hybrid_GNN_v
    network = NN(train_params)

    bs_num_an = train_params['bs_num_an']
    bs_num_rf = train_params['bs_num_rf']
    num_users = train_params['num_users']
    num_users_nei = train_params['num_users_nei']
    num_cell = train_params['num_cell']
    num_cell_max = train_params['num_cell_max']

    batch = train_params['batch_size']
    epoch = train_params['epoch']
    saver_threshold = train_params['saver_threshold']

    num_train = train_params['num_train']
    num_test = train_params['num_test']

    model_dir = "Cell" + str(num_cell) + "_TX" + str(bs_num_an) + "_UE" + str(num_users)
    file_dir = "Cell" + str(num_cell_max) + "_TX" + str(bs_num_an) + "_UE" + str(num_users)

    data_path = "./data/" + file_dir + ".mat"

    with h5py.File(data_path, 'r') as f:
        # Load train data
        H_all_train = f['H_all_train'][:]
        H_all_train = np.transpose(H_all_train, axes=(0, 1, 3, 4, 2, 5))
        H_all_train = H_all_train[0:num_train, :, :, :, :, :]

        H_cell_train = f['H_cell_train'][:]
        H_cell_train = np.transpose(H_cell_train, axes=(0, 1, 3, 2, 4))

        Large_scale_train = f['Large_scale_train'][:]
        Large_scale_train = np.transpose(Large_scale_train, axes=(0, 2, 1, 3))

        Large_scale_nei_train = top_n_rows_per_block(Large_scale_train, num_users_nei, bs_num_an)
        Large_scale_nei_train = np.reshape(Large_scale_nei_train, [np.shape(Large_scale_nei_train)[0], num_users_nei,
                                                                   num_cell_max, bs_num_an, 2])
        Large_scale_nei_train = np.transpose(Large_scale_nei_train, [0, 2, 1, 3, 4])

        n_UE_train = f['num_users_train'][:]
        n_UE_train = n_UE_train[0:num_train, :, :]

        H_all_test = f['H_all_test'][:]
        H_all_test = np.transpose(H_all_test, axes=(0, 1, 3, 4, 2, 5))
        H_all_test = H_all_test[0:num_test, :, :, :, :, :]

        H_cell_test = f['H_cell_test'][:]
        H_cell_test = np.transpose(H_cell_test, axes=(0, 1, 3, 2, 4))

        Large_scale_test = f['Large_scale_test'][:]
        Large_scale_test = np.transpose(Large_scale_test, axes=(0, 2, 1, 3))

        Large_scale_nei_test = top_n_rows_per_block(Large_scale_test, num_users_nei, bs_num_an)
        Large_scale_nei_test = np.reshape(Large_scale_nei_test, [np.shape(Large_scale_nei_test)[0], num_users_nei,
                                                                 num_cell_max, bs_num_an, 2])
        Large_scale_nei_test = np.transpose(Large_scale_nei_test, [0, 2, 1, 3, 4])

        n_UE_test = f['num_users_test'][:]
        n_UE_test = n_UE_test[0:num_test, :, :]

    H_cell_train_nor, H_cell_test_nor, Large_scale_nei_train_nor, Large_scale_nei_test_nor = \
        norm_data(H_cell_train, H_cell_test, Large_scale_nei_train, Large_scale_nei_test)

    H_cell_train_nor = H_cell_train_nor[0:num_train, 0:num_cell, :, :, :]
    H_cell_test_nor = H_cell_test_nor[0:num_test, 0:num_cell, :, :, :]
    Large_scale_nei_train_nor = Large_scale_nei_train_nor[0:num_train, 0:num_cell, :, :, :]
    Large_scale_nei_test_nor = Large_scale_nei_test_nor[0:num_test, 0:num_cell, :, :, :]
    H_all_train = H_all_train[0:num_train, 0:num_cell, 0:num_cell, :, :, :]
    H_all_test = H_all_test[0:num_test, 0:num_cell, 0:num_cell, :, :, :]
    n_UE_train = n_UE_train[0:num_train, 0:num_cell, :]
    n_UE_test = n_UE_test[0:num_test, 0:num_cell, :]

    # Save results
    save_dir = "./result/" + model_dir
    saver = tf.train.Saver(max_to_keep=5)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    sum_rate_max = 0

    sum_rate_record = []
    loss_train = []
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        network.initialize(sess)

        for e in range(epoch):
            # Train
            start_time = time.time()
            num_iter = int(np.floor(num_train / batch))
            for ite in range(num_iter):
                H_all_sample = H_all_train[ite * batch:(ite + 1) * batch, :, :, :, :, :]
                H_cell_sample = H_cell_train_nor[ite * batch:(ite + 1) * batch, :, :, :, :]
                UE_sample = n_UE_train[ite * batch:(ite + 1) * batch, :]
                Large_sample = Large_scale_nei_train_nor[ite * batch:(ite + 1) * batch, :, :, :, :]

                sum_rate_train = - network.learn_batch(sess, H_all_sample, H_cell_sample, Large_sample, UE_sample)

            # Test
            F_rf, F_bb = network.get_precoding(sess, H_all_test, H_cell_test_nor, Large_scale_nei_test_nor, n_UE_test)
            sum_rate, sum_rate_loss = network.get_rate(sess, H_all_test, H_cell_test_nor, Large_scale_nei_test_nor,
                                                       n_UE_test)
            sum_rate_all_cell_mean = np.mean(sum_rate)
            sum_rate_loss_mean = np.mean(sum_rate_loss)
            sum_rate_record.append(sum_rate_all_cell_mean)
            loss_train.append(sum_rate_train)
            print('run_id:', run_id, 'epoch:', e, 'sum_rate_train:', sum_rate_train, 'sum_rate_all_cell_test:',
                  sum_rate_all_cell_mean, 'sum_rate_loss:', sum_rate_loss_mean, 'time_epoch:', time.time() - start_time)

            # Save
            sio.savemat(save_dir + "/result.mat", {'loss_train': loss_train,
                                                   'rate_mean': sum_rate_record,
                                                   'rate_sample': sum_rate,
                                                   'F_rf': F_rf, 'F_bb': F_bb})
            if sum_rate_max < sum_rate_all_cell_mean:
                sum_rate_max = sum_rate_all_cell_mean
                if not os.path.exists(save_dir + '/checkpoints/' + NN_type):
                    os.makedirs(save_dir + '/checkpoints/' + NN_type)
                saver.save(sess, save_dir + '/checkpoints/' + NN_type + '/model.ckpt')
