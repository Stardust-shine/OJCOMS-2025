import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from importlib import import_module

train_params = {'lr': 1e-3,  # Critic learning rate
                'n_filter': [20, 20],
                'k_sz_conv': [2, 2],
                'k_sz_pool': [2, 2],
                'padding': 'SAME',
                'output_hidden': [20, 20],
                'bs_num_an': 16,
                'bs_num_rf': 3,
                'num_users': 3,
                'user_num_an': 4,
                'Ncl': 4,
                'Nray': 5,
                'bn': True,
                'epoch': 100,
                'power': 1,
                'sigma_dB': 10,
                'batch_size': 1024,
                'hidden_activation': tf.nn.relu,
                'output_activation': tf.nn.tanh,
                'NN_type': 'CNN'
                }


num_runs = 1
train = import_module('train').train

[train(train_params, run_id) for run_id in range(num_runs)]