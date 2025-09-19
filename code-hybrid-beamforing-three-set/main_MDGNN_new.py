import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from importlib import import_module

train_params = {'lr': 1e-3,  # **
                'num_hidden': [64]*8 + [2],  # **
                'FNN_hidden': [],  # **
                'FNN_type': 'non-linear',  # 'non-linear'
                'process_norm': 1.0,  # **
                'initial_1': [0.1, 2],  # **
                'bs_num_an': 32,
                'bs_num_rf': 12,
                'num_users': 10,
                'Ncl': 8,
                'Nray': 10,
                'bn': True,
                'epoch': 5000,
                'power': 1,
                'sigma_dB': 10,
                'batch_size': 256,
                'hidden_activation': tf.nn.softmax,  # tf.keras.layers.LeakyReLU(alpha=0.6)
                'NN_type': 'MDGNN_new',  # VertexGNN EdgeGNN
                'saver_threshold': 9.2,
                'num_train': 5000,
                'num_test': 1000
                }


num_runs = 1
train = import_module('train').train
[train(train_params, run_id) for run_id in range(num_runs)]

# Test_PE = import_module('Test_PE').Test_PE
# [Test_PE(train_params, run_id) for run_id in range(num_runs)]


