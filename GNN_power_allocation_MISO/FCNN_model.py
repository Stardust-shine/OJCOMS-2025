import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np


def ini_weights(in_shape, out_shape, name, stddev=0.1):
    with tf.variable_scope(name, reuse=True):
        with tf.variable_scope('weights', reuse=True):
            try:
                weight_mat = tf.get_variable(name='weight')
            except:
                weight_mat = tf.Variable(tf.truncated_normal([in_shape, out_shape], stddev=stddev), name='weight')
    return weight_mat


def ini_bias(out_shape, name):
    with tf.variable_scope(name, reuse=True):
        with tf.variable_scope('bias', reuse=True):
            try:
                bias_vec = tf.get_variable(name='bias')
            except:
                bias_vec = tf.Variable(tf.ones([out_shape]) * 0.1, name='bias')
    return bias_vec


def add_layer(input, inshape, outshape, name, stddev=0.1, keep_prob=1, transfer_function=tf.nn.relu, is_BN=True,
              is_transfer=True):
    W = ini_weights(inshape, outshape, name, stddev)
    b = ini_bias(outshape, name)
    output = tf.matmul(input, W) + b
    if is_transfer is True:
        output = transfer_function(output)
    output = tf.nn.dropout(output, keep_prob=keep_prob)
    if is_BN is True:
        output = tf.layers.batch_normalization(output, training=True)
    return output


def fcnn(input, layernum, transfer_activation=tf.nn.relu, output_activation=tf.nn.sigmoid, is_transfer=True,
         is_BN=False):
    hidden = dict()
    for i in range(len(layernum)):
        if i == 0:
            hidden[str(i)] = add_layer(input, int(input.shape[1]), layernum[i], 'layer' + str(i + 1),
                                       transfer_function=transfer_activation)
            output = hidden[str(i)]
        elif i != len(layernum) - 1:
            hidden[str(i)] = add_layer(hidden[str(i - 1)], layernum[i - 1], layernum[i], 'layer' + str(i + 1),
                                       transfer_function=transfer_activation)
        else:
            output = add_layer(hidden[str(i - 1)], layernum[i - 1], layernum[i], 'layer' + str(i + 1),
                               transfer_function=output_activation, is_BN=is_BN, is_transfer=is_transfer)
    return output


def weight_normalize_1d_pe(W, b, K, shade_mat):
    n_row = W.shape[0] / K
    n_col = W.shape[1] / K
    W_d = np.zeros([n_row, n_col]);
    W_nd = np.zeros([n_row, n_col])
    for i in range(K):
        for j in range(K):
            if i == j:
                W_d = W_d + W[n_row * i:n_row * (i + 1), n_col * i:n_col * (i + 1)]
            else:
                W_nd = W_nd + W[n_row * i:n_row * (i + 1), n_col * i:n_col * (i + 1)]

    W_d = W_d / K;
    W_nd = W_nd / (K * K - K)
    W_d_big = np.tile(W_d, (K, K));
    W_nd_big = np.tile(W_nd, (K, K))
    W_nml = W_d_big * shade_mat + W_nd_big * (1 - shade_mat)

    return W_nml, b


def get_random_block_from_data(A, B, batch_size):
    start_index = np.random.randint(0, len(A) - batch_size)
    return A[start_index: start_index + batch_size], \
           B[start_index: start_index + batch_size]
