import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from importlib import import_module

train_params = {'lr': 1e-3,  # Critic learning rate
                'num_hidden': [128]*6,
                'bs_num_an': 8,
                'bs_num_rf': 6,
                'num_users': 2,
                'user_num_an': 2,
                'Ncl': 8,
                'Nray': 10,
                'bn': True,
                'epoch': 10000,
                'power': 1,
                'sigma_dB': 10,
                'batch_size': 32,
                'hidden_activation': tf.nn.tanh,
                'output_activation': tf.nn.tanh,
                'para': 0.1,
                'NN_type': 'PENN',
                'saver_threshold': 9.2,
                'num_train': 1000,
                'num_test': 500
                }


num_runs = 1
train = import_module('train').train

[train(train_params, run_id) for run_id in range(num_runs)]


