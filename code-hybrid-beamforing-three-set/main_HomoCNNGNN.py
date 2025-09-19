import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from importlib import import_module

train_params = {'lr': 4e-4,  # **
                'num_hidden': [64]*12,  # **
                'FNN_hidden': [9, 32, 32, 2],  # **
                'CNN_dim': [5, 32, 32, 6],  # **
                'stride': [1, 1, 1, 1],  # **
                'filters': [1, 1, 1, 1],  # **
                'FNN_type': 'non-linear',  # 'non-linear'
                'initial_1': [0.1, 2],  # **
                'initial_2': [0.1, 2],  # **
                'initial_3': [0.1, 2],  # **
                'bs_num_an': 8,
                'bs_num_rf': 6,
                'num_users': 2,
                'Ncl': 8,
                'Nray': 10,
                'bn': True,
                'epoch': 2000,
                'power': 1,
                'sigma_dB': 10,
                'batch_size': 1024,
                'hidden_activation_1': tf.nn.softmax,  # tf.nn.leaky_relu
                'hidden_activation_2': tf.nn.softmax,  # tf.nn.leaky_relu
                'NN_type': 'HomoCNNGNN',  # VertexGNN EdgeGNN
                'saver_threshold': 9.2,
                'num_train': 30000,
                'num_test': 1000
                }


num_runs = 1
train = import_module('train').train
[train(train_params, run_id) for run_id in range(num_runs)]

# Test_PE = import_module('Test_PE').Test_PE
# [Test_PE(train_params, run_id) for run_id in range(num_runs)]


