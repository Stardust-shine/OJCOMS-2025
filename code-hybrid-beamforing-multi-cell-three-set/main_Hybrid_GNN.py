import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.keras.backend.clear_session()
from importlib import import_module
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

train_params = {'lr': 1e-2,  # **
                'num_hidden_EdgeGNN': [128]*8,  # **
                'bn_EdgeGNN': True,
                'initial_1': [0.1, 2],  # **
                'initial_2': [-2, -0.1],  # **
                'FNN_hidden': [],  # **
                'FNN_type': 'linear',  # 'non-linear'
                'process_norm': 1.0,  # **
                'hidden_activation_EdgeGNN': tf.nn.relu,

                'num_hidden_model_GNN': [16, 16, 16, 8, 1],  # **
                'bn_model_GNN': False,
                'hidden_activation_model_GNN': None,
                'output_transfer_model_GNN': True,

                'num_hidden_factor_fnn': [],
                'K_factor': False,
                'bn_model_GNN_FNN': False,
                'hidden_ac_factor_fnn': None,
                'output_ac_factor_fnn': None,

                'bs_num_an': 16,
                'bs_num_rf': 8,
                'num_users': 5,
                'Large_scale_nei': False,
                'num_users_nei': 5,
                'num_cell': 5,
                'num_cell_max': 10,

                'Ncl': 8,
                'Nray': 10,
                'epoch': 10000,
                'power': 1,
                'sigma_dB': -1 * (-174 + 10 * np.log10(20 * 10 ** 6) - 30),
                'batch_size': 512,
                'NN_type': 'Hybrid_GNN',
                'saver_threshold': 9.2,
                'num_train': 300000,
                'num_test': 1000,
                'type': tf.float32
                }

num_runs = 1

train = import_module('train').train
[train(train_params, run_id) for run_id in range(num_runs)]



