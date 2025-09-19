import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from importlib import import_module

train_params = {'lr': 1e-3,  # Critic learning rate
                'num_hidden': [1024, 1024, 1024, 1024],
                'bs_num_an': 8,
                'bs_num_rf': 6,
                'num_users': 2,
                'Ncl': 8,
                'Nray': 10,
                'bn': True,
                'epoch': 1000,
                'power': 1,
                'sigma_dB': 10,
                'batch_size': 1024,
                'hidden_activation': tf.nn.relu,
                'output_activation': tf.nn.tanh,
                'NN_type': 'FNN',
                'saver_threshold': 9.2,
                'num_train': 30000,
                'num_test': 1000
                }


num_runs = 1
train = import_module('train').train

[train(train_params, run_id) for run_id in range(num_runs)]


