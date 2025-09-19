import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
# import tensorflow as tf
from importlib import import_module

test_params = {'lr': 1e-4,  # **
               'num_hidden': [128] * 12,  # **
               'FNN_hidden': [],  # **
               'FNN_type': 'linear',  # 'non-linear'
               'process_norm': 1,  # **
               'initial_1': [0.1, 2],  # **
               'initial_2': [0.1, 2],  # **
               'initial_3': [0.1, 2],  # **
               'initial_4': [0.1, 2],  # **
               'initial_5': [0.1, 2],  # **
               'bs_num_an': 8,
               'bs_num_rf': 6,
               'num_users': 2,
               'user_num_an': 2,
               'bs_num_an_train': 8,
               'bs_num_rf_train': 6,
               'num_users_train': 2,
               'user_num_an_train': 2,
               'Ncl': 8,
               'Nray': 10,
               'bn': True,
               'epoch': 4000,
               'power': 1,
               'sigma_dB': 10,
               'batch_size': 64,
               'hidden_activation': tf.nn.tanh,  # tf.keras.layers.LeakyReLU(alpha=0.6)
               'NN_type': 'EdgeGNN',
               'num_test': 500
               }

gen_test = import_module('gen_test').gen_test
gen_test(test_params)