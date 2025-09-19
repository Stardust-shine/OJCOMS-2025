import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from importlib import import_module

general_params = {'lr': 1e-4,  # **
                  'num_hidden': [128] * 8,  # **
                  'FNN_hidden': [],  # **
                  'FNN_type': 'non-linear',  # 'non-linear'
                  'process_norm': 0.7,  # **
                  'initial_1': [0.1, 2],  # **
                  'initial_2': [0.1, 2],  # **
                  'initial_3': [0.1, 2],  # **
                  'initial_4': [0.1, 2],  # **
                  'initial_5': [0.1, 2],  # **
                  'bn': True,
                  'epoch': 400,
                  'power': 1,
                  'sigma_dB': 10,
                  'batch_size': 1024,
                  'hidden_activation': tf.nn.tanh
                  }

model_params = {'bs_num_an_train': 6,
                'bs_num_rf_train': 2,
                'num_users_train': 2,
                'user_num_an_train': 2,
                'Ncl_train': 8,
                'Nray_train': 10,
                'NN_type': 'VertexGNN'  # VertexGNN EdgeGNN
                }

test_params = {'bs_num_an': 6,
               'bs_num_rf': 2,
               'num_users': 2,
               'user_num_an': 2,
               'Ncl': 8,
               'Nray': 10
               }


num_runs = 1
test = import_module('gen_test').test

[test({**general_params, **model_params, **test_params}, run_id) for run_id in range(num_runs)]


