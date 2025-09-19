import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from importlib import import_module
from scipy.io import loadmat
import time
import os
import scipy.io as sio


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
    elif NN_type == 'flexible_VertexGNN':
        NN = import_module('flexible_VertexGNN').flexible_VertexGNN
    elif NN_type == 'flexible_EdgeGNN':
        NN = import_module('flexible_EdgeGNN').flexible_EdgeGNN
    elif NN_type == 'MDGNN':
        NN = import_module('MDGNN').MDGNN
    elif NN_type == 'RGNN':
        NN = import_module('RGNN').RGNN
    elif NN_type == 'REGNN':
        NN = import_module('REGNN').REGNN
    elif NN_type == 'HomoVertexGNN':
        NN = import_module('HomoVertexGNN').HomoVertexGNN
    elif NN_type == 'VanillaHetVertexGNN':
        NN = import_module('VanillaHetVertexGNN').VanillaHetVertexGNN
    elif NN_type == 'PGNN_GJ':
        NN = import_module('PGNN_GJ').PGNN_GJ
    elif NN_type == 'VertexGNNEdgeType':
        NN = import_module('VertexGNNEdgeType').VertexGNNEdgeType
    elif NN_type == 'EdgeGNNEdgeType':
        NN = import_module('EdgeGNNEdgeType').EdgeGNNEdgeType
    elif NN_type == 'EdgeGNNVertexType':
        NN = import_module('EdgeGNNVertexType').EdgeGNNVertexType
    elif NN_type == 'ModelEdgeGNN':
        NN = import_module('ModelEdgeGNN').ModelEdgeGNN
    elif NN_type == 'HomoCNNGNN':
        NN = import_module('HomoCNNGNN').HomoCNNGNN
    elif NN_type == 'EdgeGNN_three_edge_type':
        NN = import_module('EdgeGNN_three_edge_type').EdgeGNN_three_edge_type
    network = NN(train_params)
    bs_num_an = train_params['bs_num_an']
    bs_num_rf = train_params['bs_num_rf']

    num_users = train_params['num_users']
    user_num_an = train_params['user_num_an']
    Ncl = train_params['Ncl']
    Nray = train_params['Nray']
    batch = train_params['batch_size']
    epoch = train_params['epoch']
    saver_threshold = train_params['saver_threshold']

    num_train = train_params['num_train']
    num_test = train_params['num_test']
    num_data = train_params['num_data']

    file_dir = str(num_users) + "_N" + str(user_num_an) + "X"+ str(bs_num_an) + "_Ncl" + str(Ncl) + "_Nray" + str(Nray)

    # Load data
    data_train = loadmat("./data/setH_K" + file_dir + "_number" + str(num_data) + ".mat")['H'].astype(np.float32)
    data_train = np.transpose(data_train, axes=(0, 2, 3, 1))
    data_train = data_train[0:num_train, :, :, :]

    data_test = loadmat("./data/setH_K" + file_dir + "_number" + str(10000) + ".mat")['H'].astype(np.float32)
    data_test = np.transpose(data_test, axes=(0, 2, 3, 1))
    data_test = data_test[0:num_test, :, :, :]

    # Save results
    save_dir = "./result/K" + file_dir
    saver = tf.train.Saver(tf.global_variables())  # Model saver
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    sum_rate_record = []
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Initialize weight and feature
        network.initialize(sess)

        if (NN_type == 'EdgeGNN') or (NN_type == 'flexible_EdgeGNN') or (NN_type == 'EdgeGNNEdgeType') or \
                (NN_type == 'EdgeGNNVertexType') or (NN_type == 'ModelEdgeGNN') \
                or (NN_type == 'EdgeGNN_three_edge_type'):
            initial_1 = train_params['initial_1']
            initial_2 = train_params['initial_2']
            initial_3 = train_params['initial_3']
            initial_4 = train_params['initial_4']
            # Initialize the feature of each vertex - train
            edge_wrf = np.tile(np.reshape(np.linspace(initial_1[0], initial_1[1], num_users*user_num_an),
                                          [1, num_users, user_num_an, 1]), (batch, 1, 1, 2))
            edge_frf = np.tile(np.reshape(np.linspace(initial_2[0], initial_2[1], bs_num_an*bs_num_rf),
                                          [1, bs_num_an, bs_num_rf, 1]), (batch, 1, 1, 2))
            edge_fbb = np.tile(np.reshape(np.linspace(initial_3[0], initial_3[1], num_users*bs_num_rf),
                                          [1, num_users, bs_num_rf, 1]), (batch, 1, 1, 2))
            # Initialize the feature of each vertex - test
            edge_wrf_0 = np.tile(np.reshape(np.linspace(initial_1[0], initial_1[1], num_users*user_num_an),
                                            [1, num_users, user_num_an, 1]), (num_test, 1, 1, 2))
            edge_frf_0 = np.tile(np.reshape(np.linspace(initial_2[0], initial_2[1], bs_num_an*bs_num_rf),
                                            [1, bs_num_an, bs_num_rf, 1]), (num_test, 1, 1, 2))
            edge_fbb_0 = np.tile(np.reshape(np.linspace(initial_3[0], initial_3[1], num_users*bs_num_rf),
                                            [1, num_users, bs_num_rf, 1]), (num_test, 1, 1, 2))
        elif NN_type == 'RGNN':
            shade_mat = np.ones([batch, num_users * user_num_an, bs_num_an, bs_num_rf, 1])
            shade_mat_0 = np.ones([num_test, num_users * user_num_an, bs_num_an, bs_num_rf, 1])

        # Training
        sum_rate_max = 0
        for e in range(epoch):
            start_time = time.time()
            num_iter = int(np.floor(np.shape(data_train)[0]/batch))
            for ite in range(num_iter):
                sample = data_train[ite*batch:(ite+1)*batch, :, :, :]
                if (NN_type == 'EdgeGNN') or (NN_type == 'flexible_EdgeGNN') or (NN_type == 'EdgeGNNEdgeType') or \
                        (NN_type == 'EdgeGNNVertexType') or (NN_type == 'ModelEdgeGNN') \
                        or (NN_type == 'EdgeGNN_three_edge_type'):
                    sum_rate_train = - network.learn_batch(sess, sample, edge_wrf, edge_frf, edge_fbb)
                elif NN_type == 'RGNN':
                    sum_rate_train = - network.learn_batch(sess, sample, shade_mat)
                else:
                    sum_rate_train = - network.learn_batch(sess, sample)

            if (NN_type == 'EdgeGNN') or (NN_type == 'flexible_EdgeGNN') or (NN_type == 'EdgeGNNEdgeType') or \
                    (NN_type == 'EdgeGNNVertexType') or (NN_type == 'ModelEdgeGNN') \
                    or (NN_type == 'EdgeGNN_three_edge_type'):
                sum_rate = network.get_rate(sess, data_test, edge_wrf_0, edge_frf_0, edge_fbb_0)
            elif NN_type == 'RGNN':
                sum_rate = network.get_rate(sess, data_test, shade_mat_0)
            else:
                sum_rate = network.get_rate(sess, data_test)
            sum_rate_mean = np.mean(sum_rate)
            sum_rate_record.append(sum_rate_mean)
            print('run_id:', run_id, 'epoch:', e, 'sum_rate_train:', sum_rate_train, 'sum-rate_test:', sum_rate_mean,
                  'time_epoch:', time.time() - start_time)

            # Save
            # sio.savemat(save_dir + "/result.mat", {'rate_mean': sum_rate_record, 'W_rf': W_rf, 'F_rf': F_rf,
            #                                        'F_bb': F_bb, 'rate_sample': sum_rate})
            # if sum_rate_max < sum_rate_mean:
            #     sum_rate_max = sum_rate_mean
            #     if not os.path.exists(save_dir + '/checkpoints/' + NN_type):
            #         os.makedirs(save_dir + '/checkpoints/' + NN_type)
            #     saver.save(sess, save_dir + '/checkpoints/' + NN_type + '/model.ckpt')









