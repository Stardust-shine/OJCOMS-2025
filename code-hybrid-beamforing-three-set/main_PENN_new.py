import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from importlib import import_module

train_params = {'lr': 1e-3,
                'num_hidden': [64]*6,
                'bs_num_an': 16,
                'bs_num_rf': 10,
                'num_users': 5,
                'Ncl': 8,
                'Nray': 10,
                'bn': True,
                'epoch': 5000,
                'power': 1,
                'sigma_dB': 10,
                'batch_size': 1024,
                'hidden_activation': tf.nn.tanh,
                'output_activation': tf.nn.tanh,
                'para': 1,
                'NN_type': 'PENN_new',
                'saver_threshold': 9.2,
                'num_train': 100000,
                'num_test': 1000
                }


test_params = {'lr': 1e-3,
               'num_hidden': [64] * 6,
               'bs_num_an': 8,
               'bs_num_rf': 6,
               'num_users': 3,
               'num_users_train': 2,
               'Ncl': 8,
               'Nray': 10,
               'bn': True,
               'epoch': 5000,
               'power': 1,
               'sigma_dB': 10,
               'batch_size': 1024,
               'hidden_activation': tf.nn.tanh,
               'output_activation': tf.nn.tanh,
               'para': 1,
               'NN_type': "PENN_new",
               'saver_threshold': 9.2,
               'num_train': 100000,
               'num_test': 1000
               }


num_runs = 1

train = import_module('Gen_train').Gen_train
[train(train_params, run_id) for run_id in range(num_runs)]

# Test = import_module('Test_Gen').Test_Gen
# [Test(test_params, run_id) for run_id in range(num_runs)]


