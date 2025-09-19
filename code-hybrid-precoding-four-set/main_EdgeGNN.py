import numpy as np
# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
import tensorflow as tf
from importlib import import_module

train_params = {'lr': 1e-4,  # **
                'num_hidden': [128]*12,  # **
                'FNN_hidden': [],  # **
                'FNN_type': 'linear',  # 'non-linear'
                'process_norm': 1,  # **
                'initial_1': [0.1, 2],  # **
                'initial_2': [0.1, 2],  # **
                'initial_3': [0.1, 2],  # **
                'initial_4': [0.1, 2],  # **
                'initial_5': [0.1, 2],  # **
                'bs_num_an': 128,
                'bs_num_rf': 64,
                'num_users': 2,
                'user_num_an': 32,
                'Ncl': 8,
                'Nray': 10,
                'bn': True,
                'epoch': 3000,
                'power': 1,
                'sigma_dB': 10,
                'batch_size': 2,
                'hidden_activation': tf.nn.tanh,  # tf.keras.layers.LeakyReLU(alpha=0.6)
                'NN_type': 'EdgeGNN',  # VertexGNN EdgeGNN
                'saver_threshold': 9.2,
                'num_train': 4,
                'num_test': 500,
                'num_data': 500000
                }


num_runs = 1
train = import_module('train').train
[train(train_params, run_id) for run_id in range(num_runs)]


