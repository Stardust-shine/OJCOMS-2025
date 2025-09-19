import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from importlib import import_module

train_params = {'lr': 2e-4,  # **
                'num_hidden': [64]*12,  # **
                'FNN_hidden': [],  # **
                'FNN_type': 'non-linear',  # 'non-linear'
                'process_norm': 1.0,  # **
                'initial_1': [0.1, 2],  # **
                'initial_2': [0.1, 2],  # **
                'initial_3': [0.1, 2],  # **
                'initial_4': [0.1, 2],  # **
                'initial_5': [0.1, 2],  # **
                'bs_num_an': 8,
                'bs_num_rf': 6,
                'num_users': 2,
                'user_num_an': 2,
                'Ncl': 8,
                'Nray': 10,
                'bn': True,
                'epoch': 600,
                'power': 1,
                'sigma_dB': 10,
                'batch_size': 128,
                'hidden_activation': tf.nn.softmax,  # tf.nn.leaky_relu
                'NN_type': 'VertexGNNEdgeType',  # VertexGNN EdgeGNN
                'saver_threshold': 9.2,
                'num_train': 1000,
                'num_test': 500,
                'num_data': 500000
                }


num_runs = 1
train = import_module('train').train
[train(train_params, run_id) for run_id in range(num_runs)]

# Test_PE = import_module('Test_PE').Test_PE
# [Test_PE(train_params, run_id) for run_id in range(num_runs)]


