import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from importlib import import_module

train_params = {'lr': 1e-4,  # **
                'num_hidden': [128]*10,  # **
                'FNN_hidden': [],  # **
                'FNN_type': 'linear',  # 'non-linear'
                'process_norm': 1.,  # **
                'initial_1': [0.1, 2],  # **
                'initial_2': [0.1, 2],  # **
                'bs_num_an': 16,
                'bs_num_rf': 6,
                'num_users': 4,
                'Ncl': 8,
                'Nray': 10,
                'bn': True,
                'epoch': 5000,
                'power': 1,
                'sigma_dB': 10,
                'batch_size': 1024,
                'hidden_activation': tf.nn.relu,  # tf.keras.layers.LeakyReLU(alpha=0.6)
                'NN_type': 'EdgeGNN',  # VertexGNN EdgeGNN
                'saver_threshold': 9.2,
                'num_train': 30000,
                'num_test': 1000
                }


num_runs = 1
train = import_module('train').train
[train(train_params, run_id) for run_id in range(num_runs)]

# Test_PE = import_module('Test_PE').Test_PE
# [Test_PE(train_params, run_id) for run_id in range(num_runs)]


