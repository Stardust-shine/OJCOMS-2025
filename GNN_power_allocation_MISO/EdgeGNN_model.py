import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


def ini_weights(in_shape, out_shape, name, stddev=0.1):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights', reuse=tf.AUTO_REUSE):
            weight_mat = tf.get_variable(name='weight', shape=[in_shape, out_shape],
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=stddev))
    return weight_mat


def ini_bias(out_shape, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('biases', reuse=tf.AUTO_REUSE):
            bias_vec = tf.get_variable(name='bias', shape=[out_shape], initializer=tf.zeros_initializer())
    return bias_vec


def edge_gnn_layer(num_obj, x_1, x_2, x_3, index_1, index_2, index_3, out_shape, transfer_function=tf.nn.sigmoid,
                   pooling_function=tf.reduce_mean, name='0', is_test=True, is_transfer=True, is_BN=True):
    x_shape = np.shape(x_1).as_list()
    in_shape = x_shape[3]
    index_1 = tf.tile(index_1, [1, 1, 1, out_shape])
    index_2 = tf.tile(index_2, [1, 1, 1, out_shape])
    index_3 = tf.tile(index_3, [1, 1, 1, out_shape])

    # update x_1
    W11 = ini_weights(in_shape, out_shape, name=name + 'W11')
    W12 = ini_weights(in_shape, out_shape, name=name + 'W12')
    W13 = ini_weights(in_shape, out_shape, name=name + 'W13')
    W14 = ini_weights(in_shape, out_shape, name=name + 'W14')
    W15 = ini_weights(in_shape, out_shape, name=name + 'W15')
    pro_inf_1 = tf.matmul(x_2, W12) + tf.matmul(x_3, W13)
    pro_inf_2 = tf.matmul(x_2, W14) + tf.matmul(x_3, W15)
    agg_inf = tf.tile(tf.reduce_sum(pro_inf_1, axis=1, keepdims=True) / (num_obj - 1), [1, num_obj, 1, 1]) + \
              tf.tile(tf.reduce_sum(pro_inf_2, axis=2, keepdims=True) / (num_obj - 1), [1, 1, num_obj, 1])
    x_1_new = tf.matmul(x_1, W11) + agg_inf

    # update x_2
    W21 = ini_weights(in_shape, out_shape, name=name + 'W21')
    W22 = ini_weights(in_shape, out_shape, name=name + 'W22')
    W23 = ini_weights(in_shape, out_shape, name=name + 'W23')
    W24 = ini_weights(in_shape, out_shape, name=name + 'W24')
    W25 = ini_weights(in_shape, out_shape, name=name + 'W25')
    W26 = ini_weights(in_shape, out_shape, name=name + 'W26')
    W27 = ini_weights(in_shape, out_shape, name=name + 'W27')
    pro_inf_1 = tf.matmul(x_1, W21) + tf.matmul(x_2, W22) + tf.matmul(x_3, W23)
    pro_inf_2 = tf.matmul(x_1, W24) + tf.matmul(x_2, W25) + tf.matmul(x_3, W26)
    agg_inf = (tf.tile(tf.reduce_sum(pro_inf_1, axis=1, keepdims=True), [1, num_obj, 1, 1]) - tf.matmul(x_2, W22)) / (
                num_obj - 1) + \
              (tf.tile(tf.reduce_sum(pro_inf_2, axis=2, keepdims=True), [1, 1, num_obj, 1]) - tf.matmul(x_2, W25)) / (
                          num_obj - 1)
    x_2_new = tf.matmul(x_2, W27) + agg_inf

    # update x_3
    W31 = ini_weights(in_shape, out_shape, name=name + 'W31')
    W32 = ini_weights(in_shape, out_shape, name=name + 'W32')
    W33 = ini_weights(in_shape, out_shape, name=name + 'W33')
    W34 = ini_weights(in_shape, out_shape, name=name + 'W34')
    W35 = ini_weights(in_shape, out_shape, name=name + 'W35')
    W36 = ini_weights(in_shape, out_shape, name=name + 'W36')
    W37 = ini_weights(in_shape, out_shape, name=name + 'W37')
    pro_inf_1 = tf.matmul(x_1, W31) + tf.matmul(x_2, W32) + tf.matmul(x_3, W33)
    pro_inf_2 = tf.matmul(x_1, W34) + tf.matmul(x_2, W35) + tf.matmul(x_3, W36)
    agg_inf = (tf.tile(tf.reduce_sum(pro_inf_1, axis=1, keepdims=True), [1, num_obj, 1, 1]) - tf.matmul(x_3, W33))/(num_obj - 1) + \
              (tf.tile(tf.reduce_sum(pro_inf_2, axis=2, keepdims=True), [1, 1, num_obj, 1]) - tf.matmul(x_3, W36))/(num_obj - 1)
    x_3_new = tf.matmul(x_3, W37) + agg_inf

    if is_transfer is True:
        x_1_new = transfer_function(x_1_new)
        x_2_new = transfer_function(x_2_new)
        x_3_new = transfer_function(x_3_new)
        x_1_new = tf.multiply(x_1_new, index_1)
        x_2_new = tf.multiply(x_2_new, index_2)
        x_3_new = tf.multiply(x_3_new, index_3)
    if is_BN is True:
        x_1_new = tf.layers.batch_normalization(x_1_new, training=is_test, name=name+'_b1')
        x_2_new = tf.layers.batch_normalization(x_2_new, training=is_test, name=name+'_b2')
        x_3_new = tf.layers.batch_normalization(x_3_new, training=is_test, name=name+'_b3')
    x_1_new = tf.multiply(x_1_new, index_1)
    x_2_new = tf.multiply(x_2_new, index_2)
    x_3_new = tf.multiply(x_3_new, index_3)

    return x_1_new, x_2_new, x_3_new


def Edge_GNN(num_obj, x_1, x_2, x_3, index_1, index_2, index_3, n_hidden, transfer_function=tf.nn.sigmoid,
             pooling_function=tf.reduce_mean, name='0', is_test=True, is_transfer=True, is_BN=True):
    numlayer = len(n_hidden)
    x_1_hidden = x_1
    x_2_hidden = x_2
    x_3_hidden = x_3
    for i, h in enumerate(n_hidden[:-1]):
        x_1_hidden, x_2_hidden, x_3_hidden = edge_gnn_layer(num_obj, x_1_hidden, x_2_hidden, x_3_hidden, index_1,
                                                            index_2, index_3, h, transfer_function=transfer_function,
                                                            name=name + 'layer' + str(i), is_test=is_test,
                                                            is_transfer=is_transfer, is_BN=is_BN)
    x_1_hidden, x_2_hidden, x_3_hidden = edge_gnn_layer(num_obj, x_1_hidden, x_2_hidden, x_3_hidden, index_1, index_2,
                                                        index_3, n_hidden[-1],  transfer_function=transfer_function,
                                                        name=name + 'layer_out', is_test=is_test,
                                                        is_transfer=False, is_BN=False)
    return x_1_hidden, x_2_hidden, x_3_hidden


def edge_gnn_layer_kp(num_obj, x_1, x_2, x_3, index_1, index_2, index_3, out_shape, keep_prob_1, keep_prob_2,
                      transfer_function=tf.nn.sigmoid, pooling_function=tf.reduce_mean, name='0', is_test=True,
                      is_transfer=True, is_BN=True):
    x_shape = np.shape(x_1).as_list()
    in_shape = x_shape[3]
    index_1 = tf.tile(index_1, [1, 1, 1, out_shape])
    index_2 = tf.tile(index_2, [1, 1, 1, out_shape])
    index_3 = tf.tile(index_3, [1, 1, 1, out_shape])

    # update x_1
    # tf.nn.dropout(output, keep_prob=keep_prob)
    W11 = ini_weights(in_shape, out_shape, name=name + 'W11')
    W12 = ini_weights(in_shape, out_shape, name=name + 'W12')
    W13 = ini_weights(in_shape, out_shape, name=name + 'W13')
    W14 = ini_weights(in_shape, out_shape, name=name + 'W14')
    W15 = ini_weights(in_shape, out_shape, name=name + 'W15')
    pro_inf_1 = tf.nn.dropout(tf.matmul(x_2, W12), keep_prob=keep_prob_1) + \
                tf.nn.dropout(tf.matmul(x_3, W13), keep_prob=keep_prob_1)
    pro_inf_2 = tf.nn.dropout(tf.matmul(x_2, W14), keep_prob=keep_prob_1) + \
                tf.nn.dropout(tf.matmul(x_3, W15), keep_prob=keep_prob_1)
    agg_inf = tf.tile(tf.reduce_sum(pro_inf_1, axis=1, keepdims=True) / (num_obj - 1), [1, num_obj, 1, 1]) + \
              tf.tile(tf.reduce_sum(pro_inf_2, axis=2, keepdims=True) / (num_obj - 1), [1, 1, num_obj, 1])
    x_1_new = tf.nn.dropout(tf.matmul(x_1, W11), keep_prob=keep_prob_2) + agg_inf

    # update x_2
    W21 = ini_weights(in_shape, out_shape, name=name + 'W21')
    W22 = ini_weights(in_shape, out_shape, name=name + 'W22')
    W23 = ini_weights(in_shape, out_shape, name=name + 'W23')
    W24 = ini_weights(in_shape, out_shape, name=name + 'W24')
    W25 = ini_weights(in_shape, out_shape, name=name + 'W25')
    W26 = ini_weights(in_shape, out_shape, name=name + 'W26')
    W27 = ini_weights(in_shape, out_shape, name=name + 'W27')
    pro_inf_1 = tf.nn.dropout(tf.matmul(x_1, W21), keep_prob=keep_prob_1) + \
                tf.nn.dropout(tf.matmul(x_2, W22), keep_prob=keep_prob_1) + \
                tf.nn.dropout(tf.matmul(x_3, W23), keep_prob=keep_prob_1)
    pro_inf_2 = tf.nn.dropout(tf.matmul(x_1, W24), keep_prob=keep_prob_1) + \
                tf.nn.dropout(tf.matmul(x_2, W25), keep_prob=keep_prob_1) +\
                tf.nn.dropout(tf.matmul(x_3, W26), keep_prob=keep_prob_1)
    agg_inf = (tf.tile(tf.reduce_sum(pro_inf_1, axis=1, keepdims=True), [1, num_obj, 1, 1]) - tf.matmul(x_2, W22)) / (
                num_obj - 1) + \
              (tf.tile(tf.reduce_sum(pro_inf_2, axis=2, keepdims=True), [1, 1, num_obj, 1]) - tf.matmul(x_2, W25)) / (
                          num_obj - 1)
    x_2_new = tf.nn.dropout(tf.matmul(x_2, W27), keep_prob=keep_prob_2) + agg_inf

    # update x_3
    W31 = ini_weights(in_shape, out_shape, name=name + 'W31')
    W32 = ini_weights(in_shape, out_shape, name=name + 'W32')
    W33 = ini_weights(in_shape, out_shape, name=name + 'W33')
    W34 = ini_weights(in_shape, out_shape, name=name + 'W34')
    W35 = ini_weights(in_shape, out_shape, name=name + 'W35')
    W36 = ini_weights(in_shape, out_shape, name=name + 'W36')
    W37 = ini_weights(in_shape, out_shape, name=name + 'W37')
    pro_inf_1 = tf.nn.dropout(tf.matmul(x_1, W31), keep_prob=keep_prob_1) + \
                tf.nn.dropout(tf.matmul(x_2, W32), keep_prob=keep_prob_1) + \
                tf.nn.dropout(tf.matmul(x_3, W33), keep_prob=keep_prob_1)
    pro_inf_2 = tf.nn.dropout(tf.matmul(x_1, W34), keep_prob=keep_prob_1) + \
                tf.nn.dropout(tf.matmul(x_2, W35), keep_prob=keep_prob_1) + \
                tf.nn.dropout(tf.matmul(x_3, W36), keep_prob=keep_prob_1)
    agg_inf = (tf.tile(tf.reduce_sum(pro_inf_1, axis=1, keepdims=True), [1, num_obj, 1, 1]) - tf.matmul(x_3, W33))/(num_obj - 1) + \
              (tf.tile(tf.reduce_sum(pro_inf_2, axis=2, keepdims=True), [1, 1, num_obj, 1]) - tf.matmul(x_3, W36))/(num_obj - 1)
    x_3_new = tf.nn.dropout(tf.matmul(x_3, W37), keep_prob=keep_prob_2) + agg_inf

    if is_transfer is True:
        x_1_new = transfer_function(x_1_new)
        x_2_new = transfer_function(x_2_new)
        x_3_new = transfer_function(x_3_new)
        x_1_new = tf.multiply(x_1_new, index_1)
        x_2_new = tf.multiply(x_2_new, index_2)
        x_3_new = tf.multiply(x_3_new, index_3)
    if is_BN is True:
        x_1_new = tf.layers.batch_normalization(x_1_new, training=is_test)
        x_2_new = tf.layers.batch_normalization(x_2_new, training=is_test)
        x_3_new = tf.layers.batch_normalization(x_3_new, training=is_test)
    x_1_new = tf.multiply(x_1_new, index_1)
    x_2_new = tf.multiply(x_2_new, index_2)
    x_3_new = tf.multiply(x_3_new, index_3)

    return x_1_new, x_2_new, x_3_new


def Edge_GNN_kp(num_obj, x_1, x_2, x_3, index_1, index_2, index_3, n_hidden, keep_prob_1, keep_prob_2,
                transfer_function=tf.nn.sigmoid, pooling_function=tf.reduce_mean, name='0',
                is_test=True, is_transfer=True, is_BN=True):
    numlayer = len(n_hidden)
    x_1_hidden = x_1
    x_2_hidden = x_2
    x_3_hidden = x_3
    for i, h in enumerate(n_hidden[:-1]):
        x_1_hidden, x_2_hidden, x_3_hidden = edge_gnn_layer_kp(num_obj, x_1_hidden, x_2_hidden, x_3_hidden, index_1,
                                                               index_2, index_3, h, keep_prob_1, keep_prob_2,
                                                               transfer_function=transfer_function,
                                                               name=name + 'layer' + str(i), is_test=is_test,
                                                               is_transfer=is_transfer, is_BN=is_BN)
    x_1_hidden, x_2_hidden, x_3_hidden = edge_gnn_layer_kp(num_obj, x_1_hidden, x_2_hidden, x_3_hidden, index_1,
                                                           index_2,
                                                           index_3, n_hidden[-1], keep_prob_1, keep_prob_2,
                                                           transfer_function=transfer_function,
                                                           name=name + 'layer_out', is_test=is_test,
                                                           is_transfer=False, is_BN=False)
    return x_1_hidden, x_2_hidden, x_3_hidden


def edge_gnn_layer_edge_only(num_obj, x_1, x_2, x_3, index_1, index_2, index_3, out_shape,
                             transfer_function=tf.nn.sigmoid, pooling_function=tf.reduce_mean, name='0', is_test=True,
                             is_transfer=True, is_BN=True):
    x_shape = np.shape(x_1).as_list()
    in_shape = x_shape[3]
    index_1 = tf.tile(index_1, [1, 1, 1, out_shape])
    index_2 = tf.tile(index_2, [1, 1, 1, out_shape])
    index_3 = tf.tile(index_3, [1, 1, 1, out_shape])

    # update x_1
    W11 = ini_weights(in_shape, out_shape, name=name + 'W11')
    W12 = ini_weights(in_shape, out_shape, name=name + 'W12')
    W13 = ini_weights(in_shape, out_shape, name=name + 'W13')

    pro_inf_1 = tf.matmul(x_2, W12) + tf.matmul(x_3, W13)
    agg_inf = tf.tile(tf.reduce_sum(pro_inf_1, axis=1, keepdims=True) / (num_obj - 1), [1, num_obj, 1, 1]) + \
              tf.tile(tf.reduce_sum(pro_inf_1, axis=2, keepdims=True) / (num_obj - 1), [1, 1, num_obj, 1])
    x_1_new = tf.matmul(x_1, W11) + agg_inf

    # update x_2
    W21 = ini_weights(in_shape, out_shape, name=name + 'W21')
    W22 = ini_weights(in_shape, out_shape, name=name + 'W22')
    W23 = ini_weights(in_shape, out_shape, name=name + 'W23')
    W27 = ini_weights(in_shape, out_shape, name=name + 'W27')

    pro_inf_1 = tf.matmul(x_1, W21) + tf.matmul(x_2, W22) + tf.matmul(x_3, W23)
    agg_inf = (tf.tile(tf.reduce_sum(pro_inf_1, axis=1, keepdims=True), [1, num_obj, 1, 1]) - tf.matmul(x_2, W22)) / (
                num_obj - 1) + \
              (tf.tile(tf.reduce_sum(pro_inf_1, axis=2, keepdims=True), [1, 1, num_obj, 1]) - tf.matmul(x_2, W22)) / (
                          num_obj - 1)
    x_2_new = tf.matmul(x_2, W27) + agg_inf

    # update x_3
    W31 = ini_weights(in_shape, out_shape, name=name + 'W31')
    W32 = ini_weights(in_shape, out_shape, name=name + 'W32')
    W33 = ini_weights(in_shape, out_shape, name=name + 'W33')
    W37 = ini_weights(in_shape, out_shape, name=name + 'W37')

    pro_inf_1 = tf.matmul(x_1, W31) + tf.matmul(x_2, W32) + tf.matmul(x_3, W33)
    agg_inf = (tf.tile(tf.reduce_sum(pro_inf_1, axis=1, keepdims=True), [1, num_obj, 1, 1]) - tf.matmul(x_3, W33))/(num_obj - 1) + \
              (tf.tile(tf.reduce_sum(pro_inf_1, axis=2, keepdims=True), [1, 1, num_obj, 1]) - tf.matmul(x_3, W33))/(num_obj - 1)
    x_3_new = tf.matmul(x_3, W37) + agg_inf

    if is_transfer is True:
        x_1_new = transfer_function(x_1_new)
        x_2_new = transfer_function(x_2_new)
        x_3_new = transfer_function(x_3_new)
        x_1_new = tf.multiply(x_1_new, index_1)
        x_2_new = tf.multiply(x_2_new, index_2)
        x_3_new = tf.multiply(x_3_new, index_3)
    if is_BN is True:
        x_1_new = tf.layers.batch_normalization(x_1_new, training=is_test)
        x_2_new = tf.layers.batch_normalization(x_2_new, training=is_test)
        x_3_new = tf.layers.batch_normalization(x_3_new, training=is_test)
    x_1_new = tf.multiply(x_1_new, index_1)
    x_2_new = tf.multiply(x_2_new, index_2)
    x_3_new = tf.multiply(x_3_new, index_3)

    return x_1_new, x_2_new, x_3_new


def Edge_GNN_edge_type_only(num_obj, x_1, x_2, x_3, index_1, index_2, index_3, n_hidden,
                            transfer_function=tf.nn.sigmoid, pooling_function=tf.reduce_mean, name='0', is_test=True,
                            is_transfer=True, is_BN=True):
    numlayer = len(n_hidden)
    x_1_hidden = x_1
    x_2_hidden = x_2
    x_3_hidden = x_3
    for i, h in enumerate(n_hidden[:-1]):
        x_1_hidden, x_2_hidden, x_3_hidden = edge_gnn_layer_edge_only(num_obj, x_1_hidden, x_2_hidden, x_3_hidden,
                                                                      index_1, index_2, index_3, h,
                                                                      transfer_function=transfer_function,
                                                                      name=name + 'layer' + str(i), is_test=is_test,
                                                                      is_transfer=is_transfer, is_BN=is_BN)
    x_1_hidden, x_2_hidden, x_3_hidden = edge_gnn_layer_edge_only(num_obj, x_1_hidden, x_2_hidden, x_3_hidden, index_1,
                                                                  index_2, index_3, n_hidden[-1],
                                                                  transfer_function=transfer_function,
                                                                  name=name + 'layer_out', is_test=is_test,
                                                                  is_transfer=False, is_BN=False)
    return x_1_hidden, x_2_hidden, x_3_hidden


def edge_gnn_layer_vertex_type_only(num_obj, x_1, x_2, x_3, index_1, index_2, index_3, out_shape,
                                    transfer_function=tf.nn.sigmoid, pooling_function=tf.reduce_mean, name='0',
                                    is_test=True, is_transfer=True, is_BN=True):
    x_shape = np.shape(x_1).as_list()
    in_shape = x_shape[3]
    index_1 = tf.tile(index_1, [1, 1, 1, out_shape])
    index_2 = tf.tile(index_2, [1, 1, 1, out_shape])
    index_3 = tf.tile(index_3, [1, 1, 1, out_shape])

    W11 = ini_weights(in_shape, out_shape, name=name + 'W11')
    W12 = ini_weights(in_shape, out_shape, name=name + 'W12')
    W13 = ini_weights(in_shape, out_shape, name=name + 'W13')
    pro_1 = tf.matmul(x_2, W12) + tf.matmul(x_3, W12)
    pro_2 = tf.matmul(x_2, W13) + tf.matmul(x_3, W13)
    pro_inf_1 = tf.matmul(x_1, W12) + tf.matmul(x_2, W12) + tf.matmul(x_3, W12)
    pro_inf_2 = tf.matmul(x_1, W13) + tf.matmul(x_2, W13) + tf.matmul(x_3, W13)
    # update x_1
    agg_inf = tf.tile(tf.reduce_sum(pro_1, axis=1, keepdims=True) / (num_obj - 1), [1, num_obj, 1, 1]) + \
              tf.tile(tf.reduce_sum(pro_2, axis=2, keepdims=True) / (num_obj - 1), [1, 1, num_obj, 1])
    x_1_new = tf.matmul(x_1, W11) + agg_inf

    # update x_2
    agg_inf = (tf.tile(tf.reduce_sum(pro_inf_1, axis=1, keepdims=True), [1, num_obj, 1, 1]) - tf.matmul(x_2, W12)) / (
                num_obj - 1) + \
              (tf.tile(tf.reduce_sum(pro_inf_2, axis=2, keepdims=True), [1, 1, num_obj, 1]) - tf.matmul(x_2, W13)) / (
                          num_obj - 1)
    x_2_new = tf.matmul(x_2, W11) + agg_inf

    # update x_3
    agg_inf = (tf.tile(tf.reduce_sum(pro_inf_1, axis=1, keepdims=True), [1, num_obj, 1, 1]) - tf.matmul(x_3, W12))/(num_obj - 1) + \
              (tf.tile(tf.reduce_sum(pro_inf_2, axis=2, keepdims=True), [1, 1, num_obj, 1]) - tf.matmul(x_3, W13))/(num_obj - 1)
    x_3_new = tf.matmul(x_3, W11) + agg_inf

    if is_transfer is True:
        x_1_new = transfer_function(x_1_new)
        x_2_new = transfer_function(x_2_new)
        x_3_new = transfer_function(x_3_new)
        x_1_new = tf.multiply(x_1_new, index_1)
        x_2_new = tf.multiply(x_2_new, index_2)
        x_3_new = tf.multiply(x_3_new, index_3)
    if is_BN is True:
        x_1_new = tf.layers.batch_normalization(x_1_new, training=is_test)
        x_2_new = tf.layers.batch_normalization(x_2_new, training=is_test)
        x_3_new = tf.layers.batch_normalization(x_3_new, training=is_test)
    x_1_new = tf.multiply(x_1_new, index_1)
    x_2_new = tf.multiply(x_2_new, index_2)
    x_3_new = tf.multiply(x_3_new, index_3)

    return x_1_new, x_2_new, x_3_new


def Edge_GNN_vertex_type_only(num_obj, x_1, x_2, x_3, index_1, index_2, index_3, n_hidden,
                              transfer_function=tf.nn.sigmoid, pooling_function=tf.reduce_mean, name='0', is_test=True,
                              is_transfer=True, is_BN=True):
    numlayer = len(n_hidden)
    x_1_hidden = x_1
    x_2_hidden = x_2
    x_3_hidden = x_3
    for i, h in enumerate(n_hidden[:-1]):
        x_1_hidden, x_2_hidden, x_3_hidden = edge_gnn_layer_vertex_type_only(num_obj, x_1_hidden, x_2_hidden,
                                                                             x_3_hidden, index_1, index_2, index_3, h,
                                                                             transfer_function=transfer_function,
                                                                             name=name + 'layer' + str(i),
                                                                             is_test=is_test,
                                                                             is_transfer=is_transfer, is_BN=is_BN)
    x_1_hidden, x_2_hidden, x_3_hidden = edge_gnn_layer_vertex_type_only(num_obj, x_1_hidden, x_2_hidden, x_3_hidden,
                                                                         index_1, index_2, index_3, n_hidden[-1],
                                                                         transfer_function=transfer_function,
                                                                         name=name + 'layer_out', is_test=is_test,
                                                                         is_transfer=False, is_BN=False)
    return x_1_hidden, x_2_hidden, x_3_hidden


def gen_sample(num_obj_1, num_obj_2, H):
    # H: [batch, num_obj_1*num_obj_2, num_obj_1*num_obj_2, 1]
    H = np.squeeze(H)
    batch = np.shape(H)[0]
    x_1 = np.diagonal(H, axis1=1, axis2=2)
    x_1 = np.array([np.diag(x_1[i, :]) for i in range(batch)])
    H_o = H - x_1
    x_2 = np.zeros(shape=np.shape(H))
    x_3 = np.zeros(shape=np.shape(H))
    for i in range(num_obj_1):
        for j in range(num_obj_1):
            if i == j:
                x_2[:, i * num_obj_2:(i + 1) * num_obj_2, j * num_obj_2:(j + 1) * num_obj_2] = \
                    np.squeeze(H_o[:, i * num_obj_2:(i + 1) * num_obj_2, j * num_obj_2:(j + 1) * num_obj_2])
            else:
                x_3[:, i * num_obj_2:(i + 1) * num_obj_2, j * num_obj_2:(j + 1) * num_obj_2] = \
                    np.squeeze(H_o[:, i * num_obj_2:(i + 1) * num_obj_2, j * num_obj_2:(j + 1) * num_obj_2])
    x_1 = np.expand_dims(x_1, axis=3)
    x_2 = np.expand_dims(x_2, axis=3)
    x_3 = np.expand_dims(x_3, axis=3)
    index_1 = np.sign(np.abs(x_1))
    index_2 = np.sign(np.abs(x_2))
    index_3 = np.sign(np.abs(x_3))
    return x_1, x_2, x_3, index_1, index_2, index_3


def get_random_block_from_data(A, B, C, D, E, F, G, batch_size):
    start_index = np.random.randint(0, len(A) - batch_size)
    return A[start_index: start_index + batch_size], \
           B[start_index: start_index + batch_size], \
           C[start_index: start_index + batch_size], \
           D[start_index: start_index + batch_size], \
           E[start_index: start_index + batch_size], \
           F[start_index: start_index + batch_size], \
           G[start_index: start_index + batch_size]


def edge_gnn_layer_two_edge_type(num_obj, x_1, x_2, x_3, index_1, index_2, index_3, out_shape,
                                 transfer_function=tf.nn.sigmoid, pooling_function=tf.reduce_mean, name='0',
                                 is_test=True, is_transfer=True, is_BN=True):
    x_shape = np.shape(x_1).as_list()
    in_shape = x_shape[3]
    index_1 = tf.tile(index_1, [1, 1, 1, out_shape])
    index_2 = tf.tile(index_2, [1, 1, 1, out_shape])
    index_3 = tf.tile(index_3, [1, 1, 1, out_shape])

    # update x_1
    W11 = ini_weights(in_shape, out_shape, name=name + 'W11')
    W12 = ini_weights(in_shape, out_shape, name=name + 'W12')
    W14 = ini_weights(in_shape, out_shape, name=name + 'W14')
    pro_inf_1 = tf.matmul(x_2, W12) + tf.matmul(x_3, W12)
    pro_inf_2 = tf.matmul(x_2, W14) + tf.matmul(x_3, W14)
    agg_inf = tf.tile(tf.reduce_sum(pro_inf_1, axis=1, keepdims=True) / (num_obj - 1), [1, num_obj, 1, 1]) + \
              tf.tile(tf.reduce_sum(pro_inf_2, axis=2, keepdims=True) / (num_obj - 1), [1, 1, num_obj, 1])
    x_1_new = tf.matmul(x_1, W11) + agg_inf

    # update x_2
    W21 = ini_weights(in_shape, out_shape, name=name + 'W21')
    W22 = ini_weights(in_shape, out_shape, name=name + 'W22')
    W24 = ini_weights(in_shape, out_shape, name=name + 'W24')
    W25 = ini_weights(in_shape, out_shape, name=name + 'W25')
    W2 = ini_weights(in_shape, out_shape, name=name + 'W27')
    pro_inf_1 = tf.matmul(x_1, W21) + tf.matmul(x_2, W22) + tf.matmul(x_3, W22)
    pro_inf_2 = tf.matmul(x_1, W24) + tf.matmul(x_2, W25) + tf.matmul(x_3, W25)
    agg_inf = (tf.tile(tf.reduce_sum(pro_inf_1, axis=1, keepdims=True), [1, num_obj, 1, 1]) - tf.matmul(x_2, W22)) / (
                num_obj - 1) + \
              (tf.tile(tf.reduce_sum(pro_inf_2, axis=2, keepdims=True), [1, 1, num_obj, 1]) - tf.matmul(x_2, W25)) / (
                          num_obj - 1)
    x_2_new = tf.matmul(x_2, W2) + agg_inf

    # update x_3
    pro_inf_1 = tf.matmul(x_1, W21) + tf.matmul(x_2, W22) + tf.matmul(x_3, W22)
    pro_inf_2 = tf.matmul(x_1, W24) + tf.matmul(x_2, W25) + tf.matmul(x_3, W25)
    agg_inf = (tf.tile(tf.reduce_sum(pro_inf_1, axis=1, keepdims=True), [1, num_obj, 1, 1]) - tf.matmul(x_3, W22))/(num_obj - 1) + \
              (tf.tile(tf.reduce_sum(pro_inf_2, axis=2, keepdims=True), [1, 1, num_obj, 1]) - tf.matmul(x_3, W25))/(num_obj - 1)
    x_3_new = tf.matmul(x_3, W2) + agg_inf

    if is_transfer is True:
        x_1_new = transfer_function(x_1_new)
        x_2_new = transfer_function(x_2_new)
        x_3_new = transfer_function(x_3_new)
        x_1_new = tf.multiply(x_1_new, index_1)
        x_2_new = tf.multiply(x_2_new, index_2)
        x_3_new = tf.multiply(x_3_new, index_3)
    if is_BN is True:
        x_1_new = tf.layers.batch_normalization(x_1_new, training=is_test, name=name+'_b1')
        x_2_new = tf.layers.batch_normalization(x_2_new, training=is_test, name=name+'_b2')
        x_3_new = tf.layers.batch_normalization(x_3_new, training=is_test, name=name+'_b2', reuse=True)
    x_1_new = tf.multiply(x_1_new, index_1)
    x_2_new = tf.multiply(x_2_new, index_2)
    x_3_new = tf.multiply(x_3_new, index_3)

    return x_1_new, x_2_new, x_3_new


def Edge_GNN_two_edge_type(num_obj, x_1, x_2, x_3, index_1, index_2, index_3, n_hidden, transfer_function=tf.nn.sigmoid,
                           pooling_function=tf.reduce_mean, name='0', is_test=True, is_transfer=True, is_BN=True):
    numlayer = len(n_hidden)
    x_1_hidden = x_1
    x_2_hidden = x_2
    x_3_hidden = x_3
    for i, h in enumerate(n_hidden[:-1]):
        x_1_hidden, x_2_hidden, x_3_hidden = edge_gnn_layer_two_edge_type(num_obj, x_1_hidden, x_2_hidden, x_3_hidden,
                                                                          index_1,
                                                                          index_2, index_3, h,
                                                                          transfer_function=transfer_function,
                                                                          name=name + 'layer' + str(i), is_test=is_test,
                                                                          is_transfer=is_transfer, is_BN=is_BN)
    x_1_hidden, x_2_hidden, x_3_hidden = edge_gnn_layer_two_edge_type(num_obj, x_1_hidden, x_2_hidden, x_3_hidden,
                                                                      index_1, index_2,
                                                                      index_3, n_hidden[-1],
                                                                      transfer_function=transfer_function,
                                                                      name=name + 'layer_out', is_test=is_test,
                                                                      is_transfer=False, is_BN=False)
    return x_1_hidden, x_2_hidden, x_3_hidden