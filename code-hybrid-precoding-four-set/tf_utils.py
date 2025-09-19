"""Utility functions for tensorflow"""
import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()


def complex_modulus(x):
    """
    x : 4d tensor [batch, dim_1, dim_2, 2]
    out = |x_{ij}|^2  4d tensor [batch, dim_1, dim_2, 1]
    """
    output = tf.reduce_sum(tf.square(x), axis=3, keepdims=True)
    return output


def complex_modulus_all(x):
    """
    x : 4d tensor [batch, dim_1, dim_2, 2]
    out = \sum_i \sum_j |x_{ij}|^2  4d tensor [batch, 1, 1, 1]
    """
    output = tf.reduce_sum(tf.square(x), axis=[1, 2, 3], keepdims=True)
    return output


def complex_multiply(x, y):
    """
    x : 4d tensor [batch, dim_1, dim_2, 2]
    y : 4d tensor [batch, dim_2, dim_3, 2]
    out = x*y  4d tensor [batch, dim_1, dim_3, 2]
    """
    output = tf.stack([tf.linalg.matmul(x[..., 0], y[..., 0]) - tf.linalg.matmul(x[..., 1], y[..., 1]),
                       tf.linalg.matmul(x[..., 0], y[..., 1]) + tf.linalg.matmul(x[..., 1], y[..., 0])],
                      axis=-1)
    return output


def complex_H(x):
    """
    x : 4d tensor [batch, dim_1, dim_2, 2]
    out = x^H  4d tensor [batch, dim_1, dim_2, 2]
    """
    x_1, x_2 = tf.split(x, [1, 1], axis=3)
    x = tf.concat([x_1, -1 * x_2], axis=3)
    return x


def fc(x, n_output, bn=False, training=False, scope="fc", activation_fn=None,
       initializer_w=tf.keras.initializers.VarianceScaling(mode="fan_in"),
       initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    """fully connected layer with relu activation wrapper
    Args
      x:          2d tensor [batch, n_input]
      n_output    output size
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", shape=[x.get_shape()[-1], n_output], initializer=initializer_w)
        b = tf.get_variable("b", shape=[n_output], initializer=initializer_b)
        fc = tf.add(tf.matmul(x, W), b)
        if activation_fn == '(tanh+1)*0.5':
            if bn is True:
                fc = tf.layers.batch_normalization(fc, training=training, momentum=0.99, epsilon=0.00001)
            fc = 0.5 * (tf.tanh(fc) + 1)
        elif activation_fn is not None:
            if bn is True:
                fc = tf.layers.batch_normalization(fc, training=training, momentum=0.99, epsilon=0.00001)
            fc = activation_fn(fc)
    return fc


def penn_layer(x, n_output, para, bn=False, training=False, scope="fc", activation_fn=None,
               initializer_w=tf.keras.initializers.VarianceScaling(mode="fan_in"),
               initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    """fully connected layer with relu activation wrapper
    Args
      x:          5d tensor [batch, Nt, K, Nr, n_input]
      n_output    output size
    """
    x_shape = x.get_shape()
    batch = tf.shape(x)[0]
    d1 = x_shape[1]
    d2 = x_shape[2]
    d3 = x_shape[3]
    n_input = int(x.get_shape()[-1])
    with tf.variable_scope(scope):
        w1 = tf.get_variable("w1", shape=[n_input, n_output], initializer=initializer_w) * para
        w2 = tf.get_variable("w2", shape=[n_input, n_output], initializer=initializer_w) * para
        w3 = tf.get_variable("w3", shape=[n_input, n_output], initializer=initializer_w) * para
        w4 = tf.get_variable("w4", shape=[n_input, n_output], initializer=initializer_w) * para
        w5 = tf.get_variable("w5", shape=[n_input, n_output], initializer=initializer_w) * para
        w6 = tf.get_variable("w6", shape=[n_input, n_output], initializer=initializer_w) * para
        w7 = tf.get_variable("w7", shape=[n_input, n_output], initializer=initializer_w) * para
        w8 = tf.get_variable("w8", shape=[n_input, n_output], initializer=initializer_w)
        b = tf.get_variable("b", shape=[n_output], initializer=initializer_b) * para

        u1 = tf.matmul(x, w1)
        u2 = tf.matmul(x, w2)
        u3 = tf.matmul(x, w3)
        u4 = tf.matmul(x, w4)
        u5 = tf.matmul(x, w5)
        u6 = tf.matmul(x, w6)
        u7 = tf.matmul(x, w7)
        u8 = tf.matmul(x, w8)

        v1 = tf.tile(tf.reduce_sum(u1, axis=3, keepdims=True), [1, 1, 1, d3, 1]) - u1 + u2
        v2 = tf.tile(tf.reduce_sum(u3, axis=3, keepdims=True), [1, 1, 1, d3, 1]) - u3 + u4
        v3 = tf.tile(tf.reduce_sum(u5, axis=3, keepdims=True), [1, 1, 1, d3, 1]) - u5 + u6
        v4 = tf.tile(tf.reduce_sum(u7, axis=3, keepdims=True), [1, 1, 1, d3, 1]) - u7 + u8

        t1 = tf.tile(tf.reduce_sum(v1, axis=2, keepdims=True), [1, 1, d2, 1, 1]) - v1 + v2
        t2 = tf.tile(tf.reduce_sum(v3, axis=2, keepdims=True), [1, 1, d2, 1, 1]) - v3 + v4

        output = tf.tile(tf.reduce_sum(t1, axis=1, keepdims=True), [1, d1, 1, 1, 1]) - t1 + t2 + b
        if activation_fn == '(tanh+1)*0.5':
            if bn is True:
                output = tf.layers.batch_normalization(output, training=training, momentum=0.99, epsilon=0.00001)
            output = 0.5 * (tf.tanh(output) + 1)
        elif activation_fn is not None:
            if bn is True:
                output = tf.layers.batch_normalization(output, training=training, momentum=0.99, epsilon=0.00001)
            output = activation_fn(output)
    return output


def max_pool(x, k_sz=np.array([2, 2]), padding='SAME'):
    """max pooling layer wrapper
    Args
      x:      4d tensor [batch, height, width, channels]
      k_sz:   The size of the window for each dimension of the input tensor
    Returns
      a max pooling layer
    """
    return tf.nn.max_pool(x, ksize=[1, k_sz[0], k_sz[1], 1], strides=[1, k_sz[0], k_sz[1], 1], padding=padding)


def conv2d(x, n_filter, k_sz, bn, activation_fn, scope="con", stride=np.array([1, 1]), training=False,
           initializer_w=tf.keras.initializers.VarianceScaling(mode="fan_in"),
           initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    """convolutional layer with relu activation wrapper
    Args:
      x:          4d tensor [batch, height, width, channels]
      n_filter:   number of kernels (output size)
      k_sz:       2d array, kernel size. e.g. [8,8]
      scope:
      training:
      stride:     stride
    Returns
      a conv2d layer
    """
    with tf.variable_scope(scope):
        W = tf.get_variable("W", shape=[k_sz[0], k_sz[1], int(x.get_shape()[3]), n_filter], initializer=initializer_w)
        # W = tf.to_double(W)
        b = tf.get_variable("b", shape=[n_filter], initializer=initializer_b)
        # b = tf.to_double(b)
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, b)  # add bias term
        if activation_fn is not None:
            if bn is True:
                conv = tf.layers.batch_normalization(conv, training=training, momentum=0.99, epsilon=0.00001)
            conv = activation_fn(conv)
    return conv


def flatten(x):
    """flatten a 4d tensor into 2d
    Args
      x:          4d tensor [batch, height, width, channels]
    Returns a flattened 2d tensor
    """
    sh = x.shape
    # return tf.reshape(x, [-1, sh[1] * sh[2] * sh[3]])
    return tf.layers.flatten(x)


def FNN(x, hidden_size, bn=False, scope='FNN', training=False, activation_h=tf.nn.softmax,
        initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
        initializer_b=tf.zeros_initializer()):
    # x: [batch, number, number, dim]
    # hidden_size: number of outputs
    # return: [batch, number, number, out_dim]
    for i, h in enumerate(hidden_size[:]):
        x = fc(x, h, scope=scope + str(i), initializer_w=initializer_w, initializer_b=initializer_b)
        if activation_h is not None:
            if bn:
                x = tf.layers.batch_normalization(x, training=training)
            x = activation_h(x)
    return x


def vertex_gnn_layer(user_rf, user_an, bs_rf, bs_an, H, FNN_hidden, user_num_an, bs_num_an, bs_num_rf,
                     num_users, bn, training, out_dim, process_norm, hidden_activation, scope,
                     initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                     initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        batch = tf.shape(user_rf)[0]
        pre_dim = user_rf.get_shape().as_list()[3]
        user_an_com = tf.get_variable("user_an_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        user_rf_com = tf.get_variable("user_rf_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        user_data_com = tf.get_variable("user_data_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        bs_rf_com = tf.get_variable("bs_rf_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        bs_an_com = tf.get_variable("bs_an_com", shape=[pre_dim, out_dim], initializer=initializer_w)

        user_an_pro_w = tf.get_variable("user_an_pro_w", shape=[out_dim, out_dim], initializer=initializer_w)
        user_rf_pro_w = tf.get_variable("user_rf_pro_w", shape=[out_dim, out_dim], initializer=initializer_w)
        bs_rf_pro_w = tf.get_variable("bs_rf_pro_w", shape=[out_dim, out_dim], initializer=initializer_w)
        bs_an_pro_w = tf.get_variable("bs_an_pro_w", shape=[out_dim, out_dim], initializer=initializer_w)

        user_an_pro = FNN(user_an, FNN_hidden + [out_dim], bn, training=training, scope='user_an_pro',
                          activation_h=hidden_activation)

        user_an_edge = tf.concat([tf.transpose(tf.tile(user_an, [1, bs_num_an, 1, 1]), [0, 2, 1, 3]), H], axis=3)
        user_an_edge_pro = FNN(user_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='user_an_edge_pro',
                               activation_h=hidden_activation)

        bs_an_pro = FNN(bs_an, FNN_hidden + [out_dim], bn, training=training, scope='bs_an_pro',
                        activation_h=hidden_activation)

        bs_an_edge = tf.concat([tf.tile(bs_an, [1, num_users * user_num_an, 1, 1]), H], axis=3)
        bs_an_edge_pro = FNN(bs_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='bs_an_edge_pro',
                             activation_h=hidden_activation)

        user_rf_pro_1 = FNN(user_rf, FNN_hidden + [out_dim], bn, training=training, scope='user_rf_pro_1',
                            activation_h=hidden_activation)
        user_rf_pro_2 = FNN(user_rf, FNN_hidden + [out_dim], bn, training=training, scope='user_rf_pro_2',
                            activation_h=hidden_activation)

        bs_rf_pro_1 = FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='bs_rf_pro_1',
                          activation_h=hidden_activation)
        bs_rf_pro_2 = FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='bs_rf_pro_2',
                          activation_h=hidden_activation)

        # Update user_an
        pro_inf = tf.reshape(tf.tile(user_rf_pro_1, [1, 1, 1, user_num_an]),
                             [batch, 1, num_users * user_num_an, out_dim]) + \
                  tf.transpose(tf.reduce_sum(bs_an_edge_pro, axis=2, keepdims=True), [0, 2, 1, 3])
        user_an_new = tf.matmul(user_an, user_an_com) + process_norm * tf.matmul(pro_inf, user_an_pro_w)

        # Update bs_an
        pro_inf = tf.reduce_sum(tf.tile(tf.transpose(bs_rf_pro_1, [0, 2, 1, 3]), [1, 1, bs_num_an, 1]), axis=1,
                                keepdims=True) + \
                  tf.reduce_sum(user_an_edge_pro, axis=1, keepdims=True)
        bs_an_new = tf.matmul(bs_an, bs_an_com) + process_norm * tf.matmul(pro_inf, bs_an_pro_w)

        # Update user_rf
        pro_inf = tf.transpose(tf.reduce_sum(tf.reshape(user_an_pro, [batch, num_users, user_num_an, out_dim]),
                                             axis=2, keepdims=True), [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_sum(bs_rf_pro_2, axis=2, keepdims=True), [1, 1, num_users, 1])
        user_rf_new = tf.matmul(user_rf, user_rf_com) + process_norm * tf.matmul(pro_inf, user_rf_pro_w)

        # Update bs_rf
        pro_inf = tf.transpose(tf.reduce_sum(tf.tile(bs_an_pro, [1, bs_num_rf, 1, 1]), axis=2, keepdims=True),
                               [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_sum(user_rf_pro_2, axis=2, keepdims=True), [1, 1, bs_num_rf, 1])
        bs_rf_new = tf.matmul(bs_rf, bs_rf_com) + process_norm * tf.matmul(pro_inf, bs_rf_pro_w)

        return user_rf_new, user_an_new, bs_rf_new, bs_an_new


def vertex_gnn_layer_1(user_rf, user_an, bs_rf, bs_an, H, FNN_hidden, user_num_an, bs_num_an, bs_num_rf,
                       num_users, bn, training, out_dim, process_norm, hidden_activation, scope,
                       initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                       initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        batch = tf.shape(user_rf)[0]
        pre_dim = user_rf.get_shape().as_list()[3]
        user_an_com = tf.get_variable("user_an_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        user_rf_com = tf.get_variable("user_rf_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        user_data_com = tf.get_variable("user_data_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        bs_rf_com = tf.get_variable("bs_rf_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        bs_an_com = tf.get_variable("bs_an_com", shape=[pre_dim, out_dim], initializer=initializer_w)

        user_an_pro = FNN(user_an, FNN_hidden + [out_dim], bn, training=training, scope='user_an_pro',
                          activation_h=hidden_activation)

        user_an_edge = tf.concat([tf.transpose(tf.tile(user_an, [1, bs_num_an, 1, 1]), [0, 2, 1, 3]), H], axis=3)
        user_an_edge_pro = FNN(user_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='user_an_edge_pro',
                               activation_h=hidden_activation)

        bs_an_pro = FNN(bs_an, FNN_hidden + [out_dim], bn, training=training, scope='bs_an_pro',
                        activation_h=hidden_activation)

        bs_an_edge = tf.concat([tf.tile(bs_an, [1, num_users * user_num_an, 1, 1]), H], axis=3)
        bs_an_edge_pro = FNN(bs_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='bs_an_edge_pro',
                             activation_h=hidden_activation)

        user_rf_pro_1 = FNN(user_rf, FNN_hidden + [out_dim], bn, training=training, scope='user_rf_pro_1',
                            activation_h=hidden_activation)
        user_rf_pro_2 = FNN(user_rf, FNN_hidden + [out_dim], bn, training=training, scope='user_rf_pro_2',
                            activation_h=hidden_activation)

        bs_rf_pro_1 = FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='bs_rf_pro_1',
                          activation_h=hidden_activation)
        bs_rf_pro_2 = FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='bs_rf_pro_2',
                          activation_h=hidden_activation)

        # Update user_an
        pro_inf = tf.reshape(tf.tile(user_rf_pro_1, [1, 1, 1, user_num_an]),
                             [batch, 1, num_users * user_num_an, out_dim]) + \
                  tf.transpose(tf.reduce_mean(bs_an_edge_pro, axis=2, keepdims=True), [0, 2, 1, 3])
        user_an_new = tf.matmul(user_an, user_an_com) + process_norm * pro_inf

        # Update bs_an
        pro_inf = tf.reduce_mean(tf.tile(tf.transpose(bs_rf_pro_1, [0, 2, 1, 3]), [1, 1, bs_num_an, 1]), axis=1,
                                 keepdims=True) + \
                  tf.reduce_mean(user_an_edge_pro, axis=1, keepdims=True)
        bs_an_new = tf.matmul(bs_an, bs_an_com) + process_norm * pro_inf

        # Update user_rf
        pro_inf = tf.transpose(tf.reduce_mean(tf.reshape(user_an_pro, [batch, num_users, user_num_an, out_dim]),
                                              axis=2, keepdims=True), [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_mean(bs_rf_pro_2, axis=2, keepdims=True), [1, 1, num_users, 1])
        user_rf_new = tf.matmul(user_rf, user_rf_com) + process_norm * pro_inf

        # Update bs_rf
        pro_inf = tf.transpose(tf.reduce_mean(tf.tile(bs_an_pro, [1, bs_num_rf, 1, 1]), axis=2, keepdims=True),
                               [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_mean(user_rf_pro_2, axis=2, keepdims=True), [1, 1, bs_num_rf, 1])
        bs_rf_new = tf.matmul(bs_rf, bs_rf_com) + process_norm * pro_inf

        return user_rf_new, user_an_new, bs_rf_new, bs_an_new


def edge_gnn_layer(edge_wrf, edge_h, edge_frf, edge_fbb, H, FNN_hidden, user_num_an, bs_num_an, bs_num_rf,
                   num_users, bn, training, out_dim, process_norm, FNN_type, hidden_activation, scope,
                   initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                   initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        if FNN_type == 'linear':
            activation_fnn = None
        else:
            activation_fnn = hidden_activation
        batch = tf.shape(edge_wrf)[0]
        pre_dim = edge_wrf.get_shape().as_list()[3]

        edge_wrf_com = tf.get_variable("edge_wrf_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_h_com = tf.get_variable("edge_h_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_frf_com = tf.get_variable("edge_frf_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_fbb_com = tf.get_variable("edge_fbb_com", shape=[pre_dim, out_dim], initializer=initializer_w)

        edge_wrf_pro_1 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='edge_wrf_pro_1',
                             activation_h=activation_fnn)
        edge_wrf_pro_2 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='edge_wrf_pro_2',
                             activation_h=activation_fnn)
        edge_wrf_pro_3 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='edge_wrf_pro_3',
                             activation_h=activation_fnn)
        edge_wrf_pro_2 = tf.reduce_mean(edge_wrf_pro_2, axis=2, keepdims=True)

        edge_h_pro_1 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_1',
                           activation_h=activation_fnn)
        edge_h_pro_2 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_2',
                           activation_h=activation_fnn)
        edge_h_pro_3 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_3',
                           activation_h=activation_fnn)
        edge_h_pro_4 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_4',
                           activation_h=activation_fnn)
        edge_h_pro_5 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_5',
                           activation_h=activation_fnn)

        edge_h_pro_4 = tf.reshape(edge_h_pro_4, [batch, num_users, user_num_an, bs_num_an, out_dim])
        edge_h_pro_5 = tf.reshape(edge_h_pro_5, [batch, num_users, user_num_an, bs_num_an, out_dim])
        edge_h_pro_5 = tf.reduce_sum(edge_h_pro_5, axis=2, keepdims=True)

        edge_frf_pro_1 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_1',
                             activation_h=activation_fnn)
        edge_frf_pro_2 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_2',
                             activation_h=activation_fnn)
        edge_frf_pro_3 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_3',
                             activation_h=activation_fnn)
        edge_frf_pro_4 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_4',
                             activation_h=activation_fnn)

        edge_fbb_pro_1 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_1',
                             activation_h=activation_fnn)
        edge_fbb_pro_2 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_2',
                             activation_h=activation_fnn)
        edge_fbb_pro_3 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_3',
                             activation_h=activation_fnn)
        edge_fbb_pro_4 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_4',
                             activation_h=activation_fnn)

        edge_fbb_pro_2 = tf.reduce_mean(edge_fbb_pro_2, axis=2, keepdims=True)

        # Update edge_wrf
        pro_inf = tf.reduce_mean(tf.reshape(edge_h_pro_1, [batch, num_users, user_num_an, bs_num_an, out_dim]), axis=3) \
                  + (tf.tile(tf.reduce_sum(edge_wrf_pro_3, axis=2, keepdims=True),
                             [1, 1, user_num_an, 1]) - edge_wrf_pro_3) / (user_num_an - 1) \
                  + edge_fbb_pro_2
        edge_wrf_new = tf.matmul(edge_wrf, edge_wrf_com) + process_norm * pro_inf

        # Update edge_h
        pro_inf = tf.tile(tf.reshape(edge_wrf_pro_1, [batch, num_users * user_num_an, 1, out_dim]),
                          [1, 1, bs_num_an, 1]) + \
                  tf.tile(tf.transpose(tf.reduce_mean(edge_frf_pro_1, axis=2, keepdims=True), [0, 2, 1, 3]),
                          [1, num_users * user_num_an, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_h_pro_3, axis=2, keepdims=True), [1, 1, bs_num_an, 1]) - edge_h_pro_3) / (
                              bs_num_an - 1) + \
                  tf.reshape((tf.tile(tf.reduce_sum(edge_h_pro_4, axis=2, keepdims=True),
                                      [1, 1, user_num_an, 1, 1]) - edge_h_pro_4 +
                              tf.tile(tf.tile(tf.reduce_sum(edge_h_pro_5, axis=1, keepdims=True),
                                              [1, num_users, 1, 1, 1]) - edge_h_pro_5,
                                      [1, 1, user_num_an, 1, 1])) / (num_users * user_num_an - 1),
                             [batch, num_users * user_num_an, bs_num_an, out_dim])
        edge_h_new = tf.matmul(edge_h, edge_h_com) + process_norm * pro_inf

        # Update edge_frf
        pro_inf = tf.tile(tf.transpose(tf.reduce_mean(edge_h_pro_2, axis=1, keepdims=True), [0, 2, 1, 3]),
                          [1, 1, bs_num_rf, 1]) + \
                  tf.tile(tf.reduce_mean(edge_fbb_pro_1, axis=1, keepdims=True), [1, bs_num_an, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_3, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_frf_pro_3) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_4, axis=1, keepdims=True),
                           [1, bs_num_an, 1, 1]) - edge_frf_pro_4) / (bs_num_an - 1)
        edge_frf_new = tf.matmul(edge_frf, edge_frf_com) + process_norm * pro_inf

        # Update edge_fbb
        pro_inf = tf.tile(tf.reduce_mean(edge_frf_pro_2, axis=1, keepdims=True), [1, num_users, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_3, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_fbb_pro_3) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_4, axis=1, keepdims=True),
                           [1, num_users, 1, 1]) - edge_fbb_pro_4) / (num_users - 1) \
                  + edge_wrf_pro_2
        edge_fbb_new = tf.matmul(edge_fbb, edge_fbb_com) + process_norm * pro_inf

        return edge_wrf_new, edge_h_new, edge_frf_new, edge_fbb_new


def edge_gnn_layer_1(edge_wrf, edge_h, edge_frf, edge_fbb, H, FNN_hidden, user_num_an, bs_num_an, bs_num_rf,
                     num_users, bn, training, out_dim, process_norm, FNN_type, hidden_activation, scope,
                     initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                     initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        if FNN_type == 'linear':
            activation_fnn = None
        else:
            activation_fnn = hidden_activation
        batch = tf.shape(edge_wrf)[0]
        pre_dim = edge_wrf.get_shape().as_list()[3]

        edge_wrf_com = tf.get_variable("edge_wrf_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_h_com = tf.get_variable("edge_h_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_frf_com = tf.get_variable("edge_frf_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_fbb_com = tf.get_variable("edge_fbb_com", shape=[pre_dim, out_dim], initializer=initializer_w)

        edge_wrf_pro_1 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='edge_wrf_pro_1',
                             activation_h=activation_fnn)
        edge_wrf_pro_2 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='edge_wrf_pro_2',
                             activation_h=activation_fnn)
        edge_wrf_pro_3 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='edge_wrf_pro_3',
                             activation_h=activation_fnn)
        edge_wrf_pro_2 = tf.reduce_mean(edge_wrf_pro_2, axis=2, keepdims=True)

        edge_h_pro_1 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_1',
                           activation_h=activation_fnn)
        edge_h_pro_2 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_2',
                           activation_h=activation_fnn)
        edge_h_pro_3 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_3',
                           activation_h=activation_fnn)
        edge_h_pro_4 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_4',
                           activation_h=activation_fnn)

        edge_frf_pro_1 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_1',
                             activation_h=activation_fnn)
        edge_frf_pro_2 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_2',
                             activation_h=activation_fnn)
        edge_frf_pro_3 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_3',
                             activation_h=activation_fnn)
        edge_frf_pro_4 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_4',
                             activation_h=activation_fnn)

        edge_fbb_pro_1 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_1',
                             activation_h=activation_fnn)
        edge_fbb_pro_2 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_2',
                             activation_h=activation_fnn)
        edge_fbb_pro_3 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_3',
                             activation_h=activation_fnn)
        edge_fbb_pro_4 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_4',
                             activation_h=activation_fnn)
        edge_fbb_pro_2 = tf.reduce_mean(edge_fbb_pro_2, axis=2, keepdims=True)

        # Update edge_wrf
        pro_inf = tf.reduce_mean(tf.reshape(edge_h_pro_1, [batch, num_users, user_num_an, bs_num_an, out_dim]), axis=3) \
                  + (tf.tile(tf.reduce_sum(edge_wrf_pro_3, axis=2, keepdims=True),
                             [1, 1, user_num_an, 1]) - edge_wrf_pro_3) / (user_num_an - 1) \
                  + edge_fbb_pro_2
        edge_wrf_new = tf.matmul(edge_wrf, edge_wrf_com) + process_norm * pro_inf
        # process_norm * tf.matmul(pro_inf, edge_frf_pro)

        # Update edge_h
        pro_inf = tf.tile(tf.reshape(edge_wrf_pro_1, [batch, num_users * user_num_an, 1, out_dim]),
                          [1, 1, bs_num_an, 1]) + \
                  tf.tile(tf.transpose(tf.reduce_mean(edge_frf_pro_1, axis=2, keepdims=True), [0, 2, 1, 3]),
                          [1, num_users * user_num_an, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_h_pro_3, axis=2, keepdims=True), [1, 1, bs_num_an, 1]) - edge_h_pro_3) / (
                              bs_num_an - 1) + \
                  (tf.tile(tf.reduce_sum(edge_h_pro_4, axis=1, keepdims=True),
                           [1, num_users * user_num_an, 1, 1]) - edge_h_pro_4) / (num_users * user_num_an - 1)
        edge_h_new = tf.matmul(edge_h, edge_h_com) + process_norm * pro_inf

        # Update edge_frf
        pro_inf = tf.tile(tf.transpose(tf.reduce_mean(edge_h_pro_2, axis=1, keepdims=True), [0, 2, 1, 3]),
                          [1, 1, bs_num_rf, 1]) + \
                  tf.tile(tf.reduce_mean(edge_fbb_pro_1, axis=1, keepdims=True), [1, bs_num_an, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_3, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_frf_pro_3) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_4, axis=1, keepdims=True),
                           [1, bs_num_an, 1, 1]) - edge_frf_pro_4) / (bs_num_an - 1)
        edge_frf_new = tf.matmul(edge_frf, edge_frf_com) + process_norm * pro_inf

        # Update edge_fbb
        pro_inf = tf.tile(tf.reduce_mean(edge_frf_pro_2, axis=1, keepdims=True), [1, num_users, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_3, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_fbb_pro_3) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_4, axis=1, keepdims=True),
                           [1, num_users, 1, 1]) - edge_fbb_pro_4) / (num_users - 1) \
                  + edge_wrf_pro_2
        edge_fbb_new = tf.matmul(edge_fbb, edge_fbb_com) + process_norm * pro_inf

        return edge_wrf_new, edge_h_new, edge_frf_new, edge_fbb_new


def mdgnn_layer(real_H, im_H, user_data, user_rf, bs_rf, bn, train_ph,
                out_shape, hidden_activation, process_norm=1.0, scope="gnn",
                initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform')):
    with tf.variable_scope(scope):
        # update real_H
        real_H_new = mdgnn_layer_channel(real_H, out_shape, hidden_activation, bn, train_ph, scope='real_H-real_H') + \
                     mdgnn_layer_channel(im_H, out_shape, hidden_activation, bn, train_ph, scope='im_H-real_H') + \
                     mdgnn_layer_channel(user_data, out_shape, hidden_activation, bn, train_ph,
                                         scope='user_data-real_H') + \
                     mdgnn_layer_channel(user_rf, out_shape, hidden_activation, bn, train_ph, scope='user_rf-real_H') + \
                     mdgnn_layer_channel(bs_rf, out_shape, hidden_activation, bn, train_ph, scope='bs_rf-real_H')
        if hidden_activation is not None:
            if bn:
                real_H_new = tf.layers.batch_normalization(real_H_new, training=train_ph)
            real_H_new = hidden_activation(real_H_new)

        # update im_H
        im_H_new = mdgnn_layer_channel(real_H, out_shape, hidden_activation, bn, train_ph, scope='real_H-im_H') + \
                   mdgnn_layer_channel(im_H, out_shape, hidden_activation, bn, train_ph, scope='im_H-im_H') + \
                   mdgnn_layer_channel(user_data, out_shape, hidden_activation, bn, train_ph, scope='user_data-im_H') + \
                   mdgnn_layer_channel(user_rf, out_shape, hidden_activation, bn, train_ph, scope='user_rf-im_H') + \
                   mdgnn_layer_channel(bs_rf, out_shape, hidden_activation, bn, train_ph, scope='bs_rf-im_H')
        if hidden_activation is not None:
            if bn:
                im_H_new = tf.layers.batch_normalization(im_H_new, training=train_ph)
            im_H_new = hidden_activation(im_H_new)

        # update user_data
        user_data_new = mdgnn_layer_channel(real_H, out_shape, hidden_activation, bn, train_ph,
                                            scope='real_H-user_data') + \
                        mdgnn_layer_channel(im_H, out_shape, hidden_activation, bn, train_ph, scope='im_H-user_data') + \
                        mdgnn_layer_channel(user_data, out_shape, hidden_activation, bn, train_ph,
                                            scope='user_data-user_data') + \
                        mdgnn_layer_channel(user_rf, out_shape, hidden_activation, bn, train_ph,
                                            scope='user_rf-user_data') + \
                        mdgnn_layer_channel(bs_rf, out_shape, hidden_activation, bn, train_ph, scope='bs_rf-user_data')
        if hidden_activation is not None:
            if bn:
                user_data_new = tf.layers.batch_normalization(user_data_new, training=train_ph)
            user_data_new = hidden_activation(user_data_new)

        # update user_rf
        user_rf_new = mdgnn_layer_channel(real_H, out_shape, hidden_activation, bn, train_ph, scope='real_H-user_rf') + \
                      mdgnn_layer_channel(im_H, out_shape, hidden_activation, bn, train_ph, scope='im_H-user_rf') + \
                      mdgnn_layer_channel(user_data, out_shape, hidden_activation, bn, train_ph,
                                          scope='user_data-user_rf') + \
                      mdgnn_layer_channel(user_rf, out_shape, hidden_activation, bn, train_ph,
                                          scope='user_rf-user_rf') + \
                      mdgnn_layer_channel(bs_rf, out_shape, hidden_activation, bn, train_ph, scope='bs_rf-user_rf')
        if hidden_activation is not None:
            if bn:
                user_rf_new = tf.layers.batch_normalization(user_rf_new, training=train_ph)
            user_rf_new = hidden_activation(user_rf_new)

        # update bs_rf
        bs_rf_new = mdgnn_layer_channel(real_H, out_shape, hidden_activation, bn, train_ph, scope='real_H-bs_rf') + \
                    mdgnn_layer_channel(im_H, out_shape, hidden_activation, bn, train_ph, scope='im_H-bs_rf') + \
                    mdgnn_layer_channel(user_data, out_shape, hidden_activation, bn, train_ph,
                                        scope='user_data-bs_rf') + \
                    mdgnn_layer_channel(user_rf, out_shape, hidden_activation, bn, train_ph, scope='user_rf-bs_rf') + \
                    mdgnn_layer_channel(bs_rf, out_shape, hidden_activation, bn, train_ph, scope='bs_rf-bs_rf')
        if hidden_activation is not None:
            if bn:
                bs_rf_new = tf.layers.batch_normalization(bs_rf_new, training=train_ph)
            bs_rf_new = hidden_activation(bs_rf_new)

    return real_H_new, im_H_new, user_data_new, user_rf_new, bs_rf_new


def mdgnn_layer_channel(input, out_shape, hidden_activation, bn, train_ph, process_norm=1.0, scope="mdgnn",
                        initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform')):
    input_shape = input.get_shape().as_list()
    in_shape = input_shape[-1]
    with tf.variable_scope(scope):
        w1 = tf.get_variable("w1", shape=[in_shape, out_shape], initializer=initializer_w)
        w2 = tf.get_variable("w2", shape=[in_shape, out_shape], initializer=initializer_w)
        w3 = tf.get_variable("w3", shape=[in_shape, out_shape], initializer=initializer_w)
        w4 = tf.get_variable("w4", shape=[in_shape, out_shape], initializer=initializer_w)
        w5 = tf.get_variable("w5", shape=[in_shape, out_shape], initializer=initializer_w)
        w6 = tf.get_variable("w6", shape=[in_shape, out_shape], initializer=initializer_w)

        pro_1 = tf.matmul(input, w1)
        pro_2 = tf.matmul(input, w2)
        pro_3 = tf.matmul(input, w3)
        pro_4 = tf.matmul(input, w4)
        pro_5 = tf.matmul(input, w5)
        pro_6 = tf.matmul(input, w6)

        output = tf.tile(tf.reduce_sum(pro_1, axis=[2, 3, 4, 5], keepdims=True),
                         [1, 1, input_shape[2], input_shape[3], input_shape[4], input_shape[5], 1]) - pro_1 + \
                 tf.tile(tf.reduce_sum(pro_2, axis=[1, 3, 4, 5], keepdims=True),
                         [1, input_shape[1], 1, input_shape[3], input_shape[4], input_shape[5], 1]) - pro_2 + \
                 tf.tile(tf.reduce_sum(pro_3, axis=[1, 2, 4, 5], keepdims=True),
                         [1, input_shape[1], input_shape[2], 1, input_shape[4], input_shape[5], 1]) - pro_3 + \
                 tf.tile(tf.reduce_sum(pro_4, axis=[1, 2, 3, 5], keepdims=True),
                         [1, input_shape[1], input_shape[2], input_shape[3], 1, input_shape[5], 1]) - pro_4 + \
                 tf.tile(tf.reduce_sum(pro_5, axis=[1, 2, 3, 4], keepdims=True),
                         [1, input_shape[1], input_shape[2], input_shape[3], input_shape[4], 1, 1]) - pro_5 + \
                 pro_6

        return output


def f_fc(x, n_output, bn=False, training=False, scope="fc", activation_fn=None,
         initializer_w=tf.keras.initializers.VarianceScaling(mode="fan_in"),
         initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    """fully connected layer with relu activation wrapper
    Args
      x:          2d tensor [batch, n_input]
      n_output    output size
    """
    batch = tf.shape(x)[0]
    x_shape = np.shape(x).as_list()
    x = tf.expand_dims(x, axis=-2)
    with tf.variable_scope(scope):
        W = tf.tile(tf.get_variable("W", shape=[1, x_shape[1], x_shape[2], x_shape[3], n_output],
                                    initializer=initializer_w), [batch, 1, 1, 1, 1])
        b = tf.tile(tf.get_variable("b", shape=[1, x_shape[1], x_shape[2], 1, n_output], initializer=initializer_b),
                    [batch, 1, 1, 1, 1])
        fc = tf.add(tf.matmul(x, W), b)
        fc = tf.reshape(fc, [batch, x_shape[1], x_shape[2], n_output])
        if activation_fn == '(tanh+1)*0.5':
            if bn is True:
                fc = tf.layers.batch_normalization(fc, training=training, momentum=0.99, epsilon=0.00001)
            fc = 0.5 * (tf.tanh(fc) + 1)
        elif activation_fn is not None:
            if bn is True:
                fc = tf.layers.batch_normalization(fc, training=training, momentum=0.99, epsilon=0.00001)
            fc = activation_fn(fc)
    return fc


def f_FNN(x, hidden_size, bn=False, scope='FNN', training=False, activation_h=tf.nn.softmax,
          initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
          initializer_b=tf.zeros_initializer()):
    # x: [batch, number, number, dim]
    # hidden_size: number of outputs
    # return: [batch, number, number, out_dim]
    for i, h in enumerate(hidden_size[:]):
        x = f_fc(x, h, scope=scope + str(i), initializer_w=initializer_w, initializer_b=initializer_b)
        if activation_h is not None:
            if bn:
                x = tf.layers.batch_normalization(x, training=training)
            x = activation_h(x)
    return x


def f_vertex_gnn_layer(user_rf, user_an, bs_rf, bs_an, H, FNN_hidden, user_num_an, bs_num_an, bs_num_rf,
                       num_users, bn, training, out_dim, process_norm, hidden_activation, scope,
                       initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                       initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        batch = tf.shape(user_rf)[0]
        pre_dim = user_rf.get_shape().as_list()[3]

        user_an_pro_w = tf.get_variable("user_an_pro_w", shape=[out_dim, out_dim], initializer=initializer_w)
        user_rf_pro_w = tf.get_variable("user_rf_pro_w", shape=[out_dim, out_dim], initializer=initializer_w)
        bs_rf_pro_w = tf.get_variable("bs_rf_pro_w", shape=[out_dim, out_dim], initializer=initializer_w)
        bs_an_pro_w = tf.get_variable("bs_an_pro_w", shape=[out_dim, out_dim], initializer=initializer_w)

        user_an_pro = f_FNN(user_an, FNN_hidden + [out_dim], bn, training=training, scope='user_an_pro',
                            activation_h=hidden_activation)

        user_an_edge = tf.concat([tf.transpose(tf.tile(user_an, [1, bs_num_an, 1, 1]), [0, 2, 1, 3]), H], axis=3)
        user_an_edge_pro = f_FNN(user_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='user_an_edge_pro',
                                 activation_h=hidden_activation)

        bs_an_pro = f_FNN(bs_an, FNN_hidden + [out_dim], bn, training=training, scope='bs_an_pro',
                          activation_h=hidden_activation)

        bs_an_edge = tf.concat([tf.tile(bs_an, [1, num_users * user_num_an, 1, 1]), H], axis=3)
        bs_an_edge_pro = f_FNN(bs_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='bs_an_edge_pro',
                               activation_h=hidden_activation)

        user_rf_pro_1 = f_FNN(user_rf, FNN_hidden + [out_dim], bn, training=training, scope='user_rf_pro_1',
                              activation_h=hidden_activation)
        user_rf_pro_2 = f_FNN(user_rf, FNN_hidden + [out_dim], bn, training=training, scope='user_rf_pro_2',
                              activation_h=hidden_activation)

        bs_rf_pro_1 = f_FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='bs_rf_pro_1',
                            activation_h=hidden_activation)
        bs_rf_pro_2 = f_FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='bs_rf_pro_2',
                            activation_h=hidden_activation)

        # Combine
        user_an_shape = user_an.get_shape().as_list()
        user_rf_shape = user_rf.get_shape().as_list()
        bs_rf_shape = bs_rf.get_shape().as_list()
        bs_an_shape = bs_an.get_shape().as_list()

        user_an_com = tf.tile(
            tf.get_variable("user_an_com", shape=[1, user_an_shape[1], user_an_shape[2], pre_dim, out_dim],
                            initializer=initializer_w), [batch, 1, 1, 1, 1])
        user_rf_com = tf.tile(
            tf.get_variable("user_rf_com", shape=[1, user_rf_shape[1], user_rf_shape[2], pre_dim, out_dim],
                            initializer=initializer_w), [batch, 1, 1, 1, 1])
        bs_rf_com = tf.tile(tf.get_variable("bs_rf_com", shape=[1, bs_rf_shape[1], bs_rf_shape[2], pre_dim, out_dim],
                                            initializer=initializer_w), [batch, 1, 1, 1, 1])
        bs_an_com = tf.tile(tf.get_variable("bs_an_com", shape=[1, bs_an_shape[1], bs_an_shape[2], pre_dim, out_dim],
                                            initializer=initializer_w), [batch, 1, 1, 1, 1])

        user_rf = tf.expand_dims(user_rf, axis=-2)
        user_an = tf.expand_dims(user_an, axis=-2)
        bs_rf = tf.expand_dims(bs_rf, axis=-2)
        bs_an = tf.expand_dims(bs_an, axis=-2)

        user_rf_combine = tf.reshape(tf.matmul(user_rf, user_rf_com),
                                     [batch, user_rf_shape[1], user_rf_shape[2], out_dim])
        user_an_combine = tf.reshape(tf.matmul(user_an, user_an_com),
                                     [batch, user_an_shape[1], user_an_shape[2], out_dim])
        bs_rf_combine = tf.reshape(tf.matmul(bs_rf, bs_rf_com),
                                   [batch, bs_rf_shape[1], bs_rf_shape[2], out_dim])
        bs_an_combine = tf.reshape(tf.matmul(bs_an, bs_an_com),
                                   [batch, bs_an_shape[1], bs_an_shape[2], out_dim])

        # Update user_an
        pro_inf = tf.reshape(tf.tile(user_rf_pro_1, [1, 1, 1, user_num_an]),
                             [batch, 1, num_users * user_num_an, out_dim]) + \
                  tf.transpose(tf.reduce_mean(bs_an_edge_pro, axis=2, keepdims=True), [0, 2, 1, 3])
        user_an_new = user_an_combine + process_norm * tf.matmul(pro_inf, user_an_pro_w)

        # Update bs_an
        pro_inf = tf.reduce_mean(tf.tile(tf.transpose(bs_rf_pro_1, [0, 2, 1, 3]), [1, 1, bs_num_an, 1]), axis=1,
                                 keepdims=True) + \
                  tf.reduce_mean(user_an_edge_pro, axis=1, keepdims=True)
        bs_an_new = bs_an_combine + process_norm * tf.matmul(pro_inf, bs_an_pro_w)

        # Update user_rf
        pro_inf = tf.transpose(tf.reduce_mean(tf.reshape(user_an_pro, [batch, num_users, user_num_an, out_dim]),
                                              axis=2, keepdims=True), [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_mean(bs_rf_pro_2, axis=2, keepdims=True), [1, 1, num_users, 1])
        user_rf_new = user_rf_combine + process_norm * tf.matmul(pro_inf, user_rf_pro_w)

        # Update bs_rf
        pro_inf = tf.transpose(tf.reduce_mean(tf.tile(bs_an_pro, [1, bs_num_rf, 1, 1]), axis=2, keepdims=True),
                               [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_mean(user_rf_pro_2, axis=2, keepdims=True), [1, 1, bs_num_rf, 1])

        bs_rf_new = bs_rf_combine + process_norm * tf.matmul(pro_inf, bs_rf_pro_w)

        return user_rf_new, user_an_new, bs_rf_new, bs_an_new


def REGNN_layer(input, adj_matrix, num_filter, scope,
                initializer=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform')):
    with tf.variable_scope(scope):
        alpha = tf.get_variable(name='alpha', shape=[num_filter], initializer=initializer)
        batch_size = tf.shape(adj_matrix)[0]
        n_UE = adj_matrix.get_shape().as_list()[1]
        eye_matrix = tf.eye(n_UE, dtype=tf.float32)
        A_k = tf.tile(tf.expand_dims(eye_matrix, axis=0), [batch_size, 1, 1])

        output = 0.0
        for k in range(num_filter):
            output = output + alpha[k] * tf.linalg.matmul(A_k, input)
            A_k = tf.linalg.matmul(A_k, adj_matrix)

        return output


def homo_vertex_gnn_layer(user_rf, user_an, bs_rf, bs_an, H, FNN_hidden, user_num_an, bs_num_an, bs_num_rf,
                          num_users, bn, training, out_dim, process_norm, hidden_activation, scope,
                          initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1),
                                                                              distribution='uniform'),
                          initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        batch = tf.shape(user_rf)[0]
        pre_dim = user_rf.get_shape().as_list()[3]
        combiner = tf.get_variable("combiner", shape=[pre_dim, out_dim], initializer=initializer_w)

        pro_w = tf.get_variable("pro_w", shape=[out_dim, out_dim], initializer=initializer_w)

        user_an_pro = FNN(user_an, FNN_hidden + [out_dim], bn, training=training, scope='processor',
                          activation_h=hidden_activation)

        user_an_edge = tf.concat([tf.transpose(tf.tile(user_an, [1, bs_num_an, 1, 1]), [0, 2, 1, 3]), H], axis=3)
        user_an_edge_pro = FNN(user_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='processor_H',
                               activation_h=hidden_activation)

        bs_an_pro = FNN(bs_an, FNN_hidden + [out_dim], bn, training=training, scope='processor',
                        activation_h=hidden_activation)

        bs_an_edge = tf.concat([tf.tile(bs_an, [1, num_users * user_num_an, 1, 1]), H], axis=3)
        bs_an_edge_pro = FNN(bs_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='processor_H',
                             activation_h=hidden_activation)

        user_rf_pro_1 = FNN(user_rf, FNN_hidden + [out_dim], bn, training=training, scope='processor',
                            activation_h=hidden_activation)
        user_rf_pro_2 = FNN(user_rf, FNN_hidden + [out_dim], bn, training=training, scope='processor',
                            activation_h=hidden_activation)

        bs_rf_pro_1 = FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='processor',
                          activation_h=hidden_activation)
        bs_rf_pro_2 = FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='processor',
                          activation_h=hidden_activation)

        # Update user_an
        pro_inf = tf.reshape(tf.tile(user_rf_pro_1, [1, 1, 1, user_num_an]),
                             [batch, 1, num_users * user_num_an, out_dim]) + \
                  tf.transpose(tf.reduce_mean(bs_an_edge_pro, axis=2, keepdims=True), [0, 2, 1, 3])
        user_an_new = tf.matmul(user_an, combiner) + process_norm * tf.matmul(pro_inf, pro_w)

        # Update bs_an
        pro_inf = tf.reduce_mean(tf.tile(tf.transpose(bs_rf_pro_1, [0, 2, 1, 3]), [1, 1, bs_num_an, 1]), axis=1,
                                 keepdims=True) + \
                  tf.reduce_mean(user_an_edge_pro, axis=1, keepdims=True)
        bs_an_new = tf.matmul(bs_an, combiner) + process_norm * tf.matmul(pro_inf, pro_w)

        # Update user_rf
        pro_inf = tf.transpose(tf.reduce_mean(tf.reshape(user_an_pro, [batch, num_users, user_num_an, out_dim]),
                                              axis=2, keepdims=True), [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_mean(bs_rf_pro_2, axis=2, keepdims=True), [1, 1, num_users, 1])
        user_rf_new = tf.matmul(user_rf, combiner) + process_norm * tf.matmul(pro_inf, pro_w)

        # Update bs_rf
        pro_inf = tf.transpose(tf.reduce_mean(tf.tile(bs_an_pro, [1, bs_num_rf, 1, 1]), axis=2, keepdims=True),
                               [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_mean(user_rf_pro_2, axis=2, keepdims=True), [1, 1, bs_num_rf, 1])
        bs_rf_new = tf.matmul(bs_rf, combiner) + process_norm * tf.matmul(pro_inf, pro_w)

        return user_rf_new, user_an_new, bs_rf_new, bs_an_new


def pgnn_layer(user_rf, user_an, bs_rf, bs_an, H, FNN_hidden, user_num_an, bs_num_an, bs_num_rf, num_users, bn,
               training, out_dim, process_norm, hidden_activation, scope,
               initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
               initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        batch = tf.shape(user_rf)[0]
        pre_dim = user_rf.get_shape().as_list()[3]
        com_1 = tf.get_variable("com_1", shape=[pre_dim, out_dim], initializer=initializer_w)
        com_2 = tf.get_variable("com_2", shape=[pre_dim, out_dim], initializer=initializer_w)

        pro_w_1 = tf.get_variable("pro_w_1", shape=[out_dim, out_dim], initializer=initializer_w)
        pro_w_2 = tf.get_variable("pro_w_2", shape=[out_dim, out_dim], initializer=initializer_w)

        user_an_pro = FNN(user_an, FNN_hidden + [out_dim], bn, training=training, scope='agg_1',
                          activation_h=hidden_activation)

        user_an_edge = tf.concat([tf.transpose(tf.tile(user_an, [1, bs_num_an, 1, 1]), [0, 2, 1, 3]), H], axis=3)
        user_an_edge_pro = FNN(user_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='agg_2',
                               activation_h=hidden_activation)

        bs_an_pro = FNN(bs_an, FNN_hidden + [out_dim], bn, training=training, scope='agg_3',
                        activation_h=hidden_activation)

        bs_an_edge = tf.concat([tf.tile(bs_an, [1, num_users * user_num_an, 1, 1]), H], axis=3)
        bs_an_edge_pro = FNN(bs_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='agg_4',
                             activation_h=hidden_activation)

        user_rf_pro_1 = FNN(user_rf, FNN_hidden + [out_dim], bn, training=training,
                            scope='agg_3', activation_h=hidden_activation)

        ones_1 = tf.ones([batch, 1, num_users, 2])
        user_rf_pro_2 = FNN(tf.concat([user_rf, ones_1], axis=-1), FNN_hidden + [out_dim], bn, training=training,
                            scope='agg_4', activation_h=hidden_activation)

        bs_rf_pro_1 = FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training,
                          scope='agg_1',
                          activation_h=hidden_activation)

        ones_2 = tf.ones([batch, 1, bs_num_rf, 2])
        bs_rf_pro_2 = FNN(tf.concat([bs_rf, ones_2], axis=-1), FNN_hidden + [out_dim], bn, training=training,
                          scope='agg_2', activation_h=hidden_activation)

        # Update user_an
        pro_inf = tf.reshape(tf.tile(user_rf_pro_1, [1, 1, 1, user_num_an]),
                             [batch, 1, num_users * user_num_an, out_dim]) + \
                  tf.transpose(tf.reduce_mean(bs_an_edge_pro, axis=2, keepdims=True), [0, 2, 1, 3])
        user_an_new = tf.matmul(user_an, com_2) + process_norm * tf.matmul(pro_inf, pro_w_2)

        # Update bs_an
        pro_inf = tf.reduce_mean(tf.tile(tf.transpose(bs_rf_pro_1, [0, 2, 1, 3]), [1, 1, bs_num_an, 1]), axis=1,
                                 keepdims=True) + tf.reduce_mean(user_an_edge_pro, axis=1, keepdims=True)
        bs_an_new = tf.matmul(bs_an, com_1) + process_norm * tf.matmul(pro_inf, pro_w_1)

        # Update user_rf
        pro_inf = tf.transpose(tf.reduce_mean(tf.reshape(user_an_pro, [batch, num_users, user_num_an, out_dim]),
                                              axis=2, keepdims=True), [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_mean(bs_rf_pro_2, axis=2, keepdims=True), [1, 1, num_users, 1])
        user_rf_new = tf.matmul(user_rf, com_1) + process_norm * tf.matmul(pro_inf, pro_w_1)

        # Update bs_rf
        pro_inf = tf.transpose(tf.reduce_mean(tf.tile(bs_an_pro, [1, bs_num_rf, 1, 1]), axis=2, keepdims=True),
                               [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_mean(user_rf_pro_2, axis=2, keepdims=True), [1, 1, bs_num_rf, 1])
        bs_rf_new = tf.matmul(bs_rf, com_2) + process_norm * tf.matmul(pro_inf, pro_w_2)

        return user_rf_new, user_an_new, bs_rf_new, bs_an_new


def vanilla_het_vertex_gnn_layer(user_rf, user_an, bs_rf, bs_an, H, FNN_hidden, user_num_an, bs_num_an, bs_num_rf,
                                 num_users, bn, training, out_dim, process_norm, hidden_activation, scope,
                                 initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1),
                                                                                     distribution='uniform'),
                                 initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        batch = tf.shape(user_rf)[0]
        pre_dim = user_rf.get_shape().as_list()[3]
        com_1 = tf.get_variable("com_1", shape=[pre_dim, out_dim], initializer=initializer_w)
        com_2 = tf.get_variable("com_2", shape=[pre_dim, out_dim], initializer=initializer_w)

        pro_w_1 = tf.get_variable("pro_w_1", shape=[out_dim, out_dim], initializer=initializer_w)
        pro_w_2 = tf.get_variable("pro_w_2", shape=[out_dim, out_dim], initializer=initializer_w)

        ones_0 = tf.ones([batch, 1, num_users * user_num_an, 2])
        user_an_pro = FNN(tf.concat([user_an, ones_0], axis=-1), FNN_hidden + [out_dim], bn, training=training,
                          scope='agg_1', activation_h=hidden_activation)

        user_an_edge = tf.concat([tf.transpose(tf.tile(user_an, [1, bs_num_an, 1, 1]), [0, 2, 1, 3]), H], axis=3)
        user_an_edge_pro = FNN(user_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='agg_1',
                               activation_h=hidden_activation)

        ones_3 = tf.ones([batch, 1, bs_num_an, 2])
        bs_an_pro = FNN(tf.concat([bs_an, ones_3], axis=-1), FNN_hidden + [out_dim], bn, training=training,
                        scope='agg_2', activation_h=hidden_activation)

        bs_an_edge = tf.concat([tf.tile(bs_an, [1, num_users * user_num_an, 1, 1]), H], axis=3)
        bs_an_edge_pro = FNN(bs_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='agg_2',
                             activation_h=hidden_activation)

        ones_1 = tf.ones([batch, 1, num_users, 2])
        user_rf_pro_1 = FNN(tf.concat([user_rf, ones_1], axis=-1), FNN_hidden + [out_dim], bn, training=training,
                            scope='agg_2', activation_h=hidden_activation)

        user_rf_pro_2 = FNN(tf.concat([user_rf, ones_1], axis=-1), FNN_hidden + [out_dim], bn, training=training,
                            scope='agg_2', activation_h=hidden_activation)

        ones_2 = tf.ones([batch, 1, bs_num_rf, 2])
        bs_rf_pro_1 = FNN(tf.concat([bs_rf, ones_2], axis=-1), FNN_hidden + [out_dim], bn, training=training,
                          scope='agg_1',
                          activation_h=hidden_activation)
        bs_rf_pro_2 = FNN(tf.concat([bs_rf, ones_2], axis=-1), FNN_hidden + [out_dim], bn, training=training,
                          scope='agg_1', activation_h=hidden_activation)

        # Update user_an
        pro_inf = tf.reshape(tf.tile(user_rf_pro_1, [1, 1, 1, user_num_an]),
                             [batch, 1, num_users * user_num_an, out_dim]) + \
                  tf.transpose(tf.reduce_mean(bs_an_edge_pro, axis=2, keepdims=True), [0, 2, 1, 3])
        user_an_new = tf.matmul(user_an, com_2) + process_norm * tf.matmul(pro_inf, pro_w_2)

        # Update bs_an
        pro_inf = tf.reduce_mean(tf.tile(tf.transpose(bs_rf_pro_1, [0, 2, 1, 3]), [1, 1, bs_num_an, 1]), axis=1,
                                 keepdims=True) + tf.reduce_mean(user_an_edge_pro, axis=1, keepdims=True)
        bs_an_new = tf.matmul(bs_an, com_1) + process_norm * tf.matmul(pro_inf, pro_w_1)

        # Update user_rf
        pro_inf = tf.transpose(tf.reduce_mean(tf.reshape(user_an_pro, [batch, num_users, user_num_an, out_dim]),
                                              axis=2, keepdims=True), [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_mean(bs_rf_pro_2, axis=2, keepdims=True), [1, 1, num_users, 1])
        user_rf_new = tf.matmul(user_rf, com_1) + process_norm * tf.matmul(pro_inf, pro_w_1)

        # Update bs_rf
        pro_inf = tf.transpose(tf.reduce_mean(tf.tile(bs_an_pro, [1, bs_num_rf, 1, 1]), axis=2, keepdims=True),
                               [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_mean(user_rf_pro_2, axis=2, keepdims=True), [1, 1, bs_num_rf, 1])
        bs_rf_new = tf.matmul(bs_rf, com_2) + process_norm * tf.matmul(pro_inf, pro_w_2)

        return user_rf_new, user_an_new, bs_rf_new, bs_an_new


def vertex_gnn_layer_edge_type(user_rf, user_an, bs_rf, bs_an, H, FNN_hidden, user_num_an, bs_num_an, bs_num_rf,
                               num_users, bn, training, out_dim, process_norm, hidden_activation, scope,
                               initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1),
                                                                                   distribution='uniform'),
                               initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        batch = tf.shape(user_rf)[0]
        pre_dim = user_rf.get_shape().as_list()[3]
        vertex_com = tf.get_variable("user_an_com", shape=[pre_dim, out_dim], initializer=initializer_w)

        user_an_pro_w = tf.get_variable("user_an_pro_w", shape=[out_dim, out_dim], initializer=initializer_w)
        user_rf_pro_w = tf.get_variable("user_rf_pro_w", shape=[out_dim, out_dim], initializer=initializer_w)
        bs_rf_pro_w = tf.get_variable("bs_rf_pro_w", shape=[out_dim, out_dim], initializer=initializer_w)
        bs_an_pro_w = tf.get_variable("bs_an_pro_w", shape=[out_dim, out_dim], initializer=initializer_w)

        user_an_pro = FNN(user_an, FNN_hidden + [out_dim], bn, training=training, scope='user_an_user_rf',
                          activation_h=hidden_activation)

        user_an_edge = tf.concat([tf.transpose(tf.tile(user_an, [1, bs_num_an, 1, 1]), [0, 2, 1, 3]), H], axis=3)
        user_an_edge_pro = FNN(user_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='user_an_bs_an',
                               activation_h=hidden_activation)

        bs_an_pro = FNN(bs_an, FNN_hidden + [out_dim], bn, training=training, scope='bs_rf_bs_an',
                        activation_h=hidden_activation)

        bs_an_edge = tf.concat([tf.tile(bs_an, [1, num_users * user_num_an, 1, 1]), H], axis=3)
        bs_an_edge_pro = FNN(bs_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='user_an_bs_an',
                             activation_h=hidden_activation)

        user_rf_pro_1 = FNN(user_rf, FNN_hidden + [out_dim], bn, training=training, scope='user_an_user_rf',
                            activation_h=hidden_activation)
        user_rf_pro_2 = FNN(user_rf, FNN_hidden + [out_dim], bn, training=training, scope='user_rf_bs_rf',
                            activation_h=hidden_activation)

        bs_rf_pro_1 = FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='bs_rf_bs_an',
                          activation_h=hidden_activation)
        bs_rf_pro_2 = FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='user_rf_bs_rf',
                          activation_h=hidden_activation)

        # Update user_an
        pro_inf = tf.reshape(tf.tile(user_rf_pro_1, [1, 1, 1, user_num_an]),
                             [batch, 1, num_users * user_num_an, out_dim]) + \
                  tf.transpose(tf.reduce_mean(bs_an_edge_pro, axis=2, keepdims=True), [0, 2, 1, 3])
        user_an_new = tf.matmul(user_an, vertex_com) + process_norm * tf.matmul(pro_inf, user_an_pro_w)

        # Update bs_an
        pro_inf = tf.reduce_mean(tf.tile(tf.transpose(bs_rf_pro_1, [0, 2, 1, 3]), [1, 1, bs_num_an, 1]), axis=1,
                                 keepdims=True) + \
                  tf.reduce_mean(user_an_edge_pro, axis=1, keepdims=True)
        bs_an_new = tf.matmul(bs_an, vertex_com) + process_norm * tf.matmul(pro_inf, bs_an_pro_w)

        # Update user_rf
        pro_inf = tf.transpose(tf.reduce_mean(tf.reshape(user_an_pro, [batch, num_users, user_num_an, out_dim]),
                                              axis=2, keepdims=True), [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_mean(bs_rf_pro_2, axis=2, keepdims=True), [1, 1, num_users, 1])
        user_rf_new = tf.matmul(user_rf, vertex_com) + process_norm * tf.matmul(pro_inf, user_rf_pro_w)

        # Update bs_rf
        pro_inf = tf.transpose(tf.reduce_mean(tf.tile(bs_an_pro, [1, bs_num_rf, 1, 1]), axis=2, keepdims=True),
                               [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_mean(user_rf_pro_2, axis=2, keepdims=True), [1, 1, bs_num_rf, 1])
        bs_rf_new = tf.matmul(bs_rf, vertex_com) + process_norm * tf.matmul(pro_inf, bs_rf_pro_w)

        return user_rf_new, user_an_new, bs_rf_new, bs_an_new


def edge_gnn_layer_vetex_type(edge_wrf, edge_h, edge_frf, edge_fbb, H, FNN_hidden, user_num_an, bs_num_an, bs_num_rf,
                              num_users, bn, training, out_dim, process_norm, FNN_type, hidden_activation, scope,
                              initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1),
                                                                                  distribution='uniform'),
                              initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        if FNN_type == 'linear':
            activation_fnn = None
        else:
            activation_fnn = hidden_activation
        batch = tf.shape(edge_wrf)[0]
        pre_dim = edge_wrf.get_shape().as_list()[3]

        edge_com = tf.get_variable("edge_com", shape=[pre_dim, out_dim], initializer=initializer_w)

        edge_wrf_pro_1 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='w2',
                             activation_h=activation_fnn)
        edge_wrf_pro_2 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='w1',
                             activation_h=activation_fnn)
        edge_wrf_pro_3 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='w1',
                             activation_h=activation_fnn)
        edge_wrf_pro_2 = tf.reduce_mean(edge_wrf_pro_2, axis=2, keepdims=True)

        edge_h_pro_1 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='w2',
                           activation_h=activation_fnn)
        edge_h_pro_2 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='w3',
                           activation_h=activation_fnn)
        edge_h_pro_3 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='w2',
                           activation_h=activation_fnn)
        edge_h_pro_4 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='w3',
                           activation_h=activation_fnn)

        edge_frf_pro_1 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='w3',
                             activation_h=activation_fnn)
        edge_frf_pro_2 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='w4',
                             activation_h=activation_fnn)
        edge_frf_pro_3 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='w3',
                             activation_h=activation_fnn)
        edge_frf_pro_4 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='w4',
                             activation_h=activation_fnn)

        edge_fbb_pro_1 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='w4',
                             activation_h=activation_fnn)
        edge_fbb_pro_2 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='w1',
                             activation_h=activation_fnn)
        edge_fbb_pro_3 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='w1',
                             activation_h=activation_fnn)
        edge_fbb_pro_4 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='w4',
                             activation_h=activation_fnn)
        edge_fbb_pro_2 = tf.reduce_mean(edge_fbb_pro_2, axis=2, keepdims=True)

        # Update edge_wrf
        pro_inf = tf.reduce_mean(tf.reshape(edge_h_pro_1, [batch, num_users, user_num_an, bs_num_an, out_dim]), axis=3) \
                  + (tf.tile(tf.reduce_sum(edge_wrf_pro_3, axis=2, keepdims=True),
                             [1, 1, user_num_an, 1]) - edge_wrf_pro_3) / (user_num_an - 1) \
                  + edge_fbb_pro_2
        edge_wrf_new = tf.matmul(edge_wrf, edge_com) + process_norm * pro_inf
        # process_norm * tf.matmul(pro_inf, edge_frf_pro)

        # Update edge_h
        pro_inf = tf.tile(tf.reshape(edge_wrf_pro_1, [batch, num_users * user_num_an, 1, out_dim]),
                          [1, 1, bs_num_an, 1]) + \
                  tf.tile(tf.transpose(tf.reduce_mean(edge_frf_pro_1, axis=2, keepdims=True), [0, 2, 1, 3]),
                          [1, num_users * user_num_an, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_h_pro_3, axis=2, keepdims=True), [1, 1, bs_num_an, 1]) - edge_h_pro_3) / (
                              bs_num_an - 1) + \
                  (tf.tile(tf.reduce_sum(edge_h_pro_4, axis=1, keepdims=True),
                           [1, num_users * user_num_an, 1, 1]) - edge_h_pro_4) / (num_users * user_num_an - 1)
        edge_h_new = tf.matmul(edge_h, edge_com) + process_norm * pro_inf

        # Update edge_frf
        pro_inf = tf.tile(tf.transpose(tf.reduce_mean(edge_h_pro_2, axis=1, keepdims=True), [0, 2, 1, 3]),
                          [1, 1, bs_num_rf, 1]) + \
                  tf.tile(tf.reduce_mean(edge_fbb_pro_1, axis=1, keepdims=True), [1, bs_num_an, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_3, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_frf_pro_3) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_4, axis=1, keepdims=True),
                           [1, bs_num_an, 1, 1]) - edge_frf_pro_4) / (bs_num_an - 1)
        edge_frf_new = tf.matmul(edge_frf, edge_com) + process_norm * pro_inf

        # Update edge_fbb
        pro_inf = tf.tile(tf.reduce_mean(edge_frf_pro_2, axis=1, keepdims=True), [1, num_users, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_3, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_fbb_pro_3) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_4, axis=1, keepdims=True),
                           [1, num_users, 1, 1]) - edge_fbb_pro_4) / (num_users - 1) \
                  + edge_wrf_pro_2
        edge_fbb_new = tf.matmul(edge_fbb, edge_com) + process_norm * pro_inf

        return edge_wrf_new, edge_h_new, edge_frf_new, edge_fbb_new


def edge_gnn_layer_vetex_type_new(edge_wrf, edge_h, edge_frf, edge_fbb, H, FNN_hidden, user_num_an, bs_num_an,
                                  bs_num_rf, num_users, bn, training, out_dim, process_norm, FNN_type,
                                  hidden_activation, scope,
                                  initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1),
                                                                                      distribution='uniform'),
                                  initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        if FNN_type == 'linear':
            activation_fnn = None
        else:
            activation_fnn = hidden_activation
        batch = tf.shape(edge_wrf)[0]
        pre_dim = edge_wrf.get_shape().as_list()[3]

        edge_com = tf.get_variable("edge_com", shape=[pre_dim, out_dim], initializer=initializer_w)

        edge_wrf_pro_1 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='w2',
                             activation_h=activation_fnn)
        edge_wrf_pro_2 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='w1',
                             activation_h=activation_fnn)
        edge_wrf_pro_2 = tf.reduce_mean(edge_wrf_pro_2, axis=2, keepdims=True)

        edge_h_pro_1 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='w2',
                           activation_h=activation_fnn)
        edge_h_pro_2 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='w3',
                           activation_h=activation_fnn)

        edge_frf_pro_1 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='w3',
                             activation_h=activation_fnn)
        edge_frf_pro_2 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='w4',
                             activation_h=activation_fnn)

        edge_fbb_pro_1 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='w4',
                             activation_h=activation_fnn)
        edge_fbb_pro_2 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='w1',
                             activation_h=activation_fnn)
        edge_fbb_pro_2 = tf.reduce_mean(edge_fbb_pro_2, axis=2, keepdims=True)

        # Update edge_wrf
        pro_inf = tf.reduce_mean(tf.reshape(edge_h_pro_1, [batch, num_users, user_num_an, bs_num_an, out_dim]), axis=3) \
                  + (tf.tile(tf.reduce_sum(edge_wrf_pro_2, axis=2, keepdims=True),
                             [1, 1, user_num_an, 1]) - edge_wrf_pro_2) / (user_num_an - 1) \
                  + edge_fbb_pro_2
        edge_wrf_new = tf.matmul(edge_wrf, edge_com) + process_norm * pro_inf
        # process_norm * tf.matmul(pro_inf, edge_frf_pro)

        # Update edge_h
        pro_inf = tf.tile(tf.reshape(edge_wrf_pro_1, [batch, num_users * user_num_an, 1, out_dim]),
                          [1, 1, bs_num_an, 1]) + \
                  tf.tile(tf.transpose(tf.reduce_mean(edge_frf_pro_1, axis=2, keepdims=True), [0, 2, 1, 3]),
                          [1, num_users * user_num_an, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_h_pro_1, axis=2, keepdims=True), [1, 1, bs_num_an, 1]) - edge_h_pro_1) / (
                              bs_num_an - 1) + \
                  (tf.tile(tf.reduce_sum(edge_h_pro_2, axis=1, keepdims=True),
                           [1, num_users * user_num_an, 1, 1]) - edge_h_pro_2) / (num_users * user_num_an - 1)
        edge_h_new = tf.matmul(edge_h, edge_com) + process_norm * pro_inf

        # Update edge_frf
        pro_inf = tf.tile(tf.transpose(tf.reduce_mean(edge_h_pro_2, axis=1, keepdims=True), [0, 2, 1, 3]),
                          [1, 1, bs_num_rf, 1]) + \
                  tf.tile(tf.reduce_mean(edge_fbb_pro_1, axis=1, keepdims=True), [1, bs_num_an, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_1, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_frf_pro_1) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_2, axis=1, keepdims=True),
                           [1, bs_num_an, 1, 1]) - edge_frf_pro_2) / (bs_num_an - 1)
        edge_frf_new = tf.matmul(edge_frf, edge_com) + process_norm * pro_inf

        # Update edge_fbb
        pro_inf = tf.tile(tf.reduce_mean(edge_frf_pro_2, axis=1, keepdims=True), [1, num_users, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_2, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_fbb_pro_2) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_1, axis=1, keepdims=True),
                           [1, num_users, 1, 1]) - edge_fbb_pro_1) / (num_users - 1) \
                  + edge_wrf_pro_2
        edge_fbb_new = tf.matmul(edge_fbb, edge_com) + process_norm * pro_inf

        return edge_wrf_new, edge_h_new, edge_frf_new, edge_fbb_new


def edge_gnn_layer_edge_type(edge_wrf, edge_h, edge_frf, edge_fbb, H, FNN_hidden, user_num_an, bs_num_an, bs_num_rf,
                             num_users, bn, training, out_dim, process_norm, FNN_type, hidden_activation, scope,
                             initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1),
                                                                                 distribution='uniform'),
                             initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        if FNN_type == 'linear':
            activation_fnn = None
        else:
            activation_fnn = hidden_activation
        batch = tf.shape(edge_wrf)[0]
        pre_dim = edge_wrf.get_shape().as_list()[3]

        edge_wrf_com = tf.get_variable("edge_wrf_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_h_com = tf.get_variable("edge_h_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_frf_com = tf.get_variable("edge_frf_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_fbb_com = tf.get_variable("edge_fbb_com", shape=[pre_dim, out_dim], initializer=initializer_w)

        edge_wrf_pro_1 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='edge_wrf_pro_1',
                             activation_h=activation_fnn)
        edge_wrf_pro_2 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='edge_wrf_pro_2',
                             activation_h=activation_fnn)
        edge_wrf_pro_3 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='edge_wrf_pro_3',
                             activation_h=activation_fnn)
        edge_wrf_pro_2 = tf.reduce_mean(edge_wrf_pro_2, axis=2, keepdims=True)

        edge_h_pro_1 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_1',
                           activation_h=activation_fnn)
        edge_h_pro_2 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_2',
                           activation_h=activation_fnn)
        edge_h_pro_3 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro',
                           activation_h=activation_fnn)

        edge_frf_pro_1 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_1',
                             activation_h=activation_fnn)
        edge_frf_pro_2 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_2',
                             activation_h=activation_fnn)
        edge_frf_pro_3 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro',
                             activation_h=activation_fnn)

        edge_fbb_pro_1 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_1',
                             activation_h=activation_fnn)
        edge_fbb_pro_2 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_2',
                             activation_h=activation_fnn)
        edge_fbb_pro_3 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro',
                             activation_h=activation_fnn)
        edge_fbb_pro_2 = tf.reduce_mean(edge_fbb_pro_2, axis=2, keepdims=True)

        # Update edge_wrf
        pro_inf = tf.reduce_mean(tf.reshape(edge_h_pro_1, [batch, num_users, user_num_an, bs_num_an, out_dim]), axis=3) \
                  + (tf.tile(tf.reduce_sum(edge_wrf_pro_3, axis=2, keepdims=True),
                             [1, 1, user_num_an, 1]) - edge_wrf_pro_3) / (user_num_an - 1) \
                  + edge_fbb_pro_2
        edge_wrf_new = tf.matmul(edge_wrf, edge_wrf_com) + process_norm * pro_inf
        # process_norm * tf.matmul(pro_inf, edge_frf_pro)

        # Update edge_h
        pro_inf = tf.tile(tf.reshape(edge_wrf_pro_1, [batch, num_users * user_num_an, 1, out_dim]),
                          [1, 1, bs_num_an, 1]) + \
                  tf.tile(tf.transpose(tf.reduce_mean(edge_frf_pro_1, axis=2, keepdims=True), [0, 2, 1, 3]),
                          [1, num_users * user_num_an, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_h_pro_3, axis=2, keepdims=True), [1, 1, bs_num_an, 1]) - edge_h_pro_3) / (
                              bs_num_an - 1) + \
                  (tf.tile(tf.reduce_sum(edge_h_pro_3, axis=1, keepdims=True),
                           [1, num_users * user_num_an, 1, 1]) - edge_h_pro_3) / (num_users * user_num_an - 1)
        edge_h_new = tf.matmul(edge_h, edge_h_com) + process_norm * pro_inf

        # Update edge_frf
        pro_inf = tf.tile(tf.transpose(tf.reduce_mean(edge_h_pro_2, axis=1, keepdims=True), [0, 2, 1, 3]),
                          [1, 1, bs_num_rf, 1]) + \
                  tf.tile(tf.reduce_mean(edge_fbb_pro_1, axis=1, keepdims=True), [1, bs_num_an, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_3, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_frf_pro_3) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_3, axis=1, keepdims=True),
                           [1, bs_num_an, 1, 1]) - edge_frf_pro_3) / (bs_num_an - 1)
        edge_frf_new = tf.matmul(edge_frf, edge_frf_com) + process_norm * pro_inf

        # Update edge_fbb
        pro_inf = tf.tile(tf.reduce_mean(edge_frf_pro_2, axis=1, keepdims=True), [1, num_users, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_3, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_fbb_pro_3) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_3, axis=1, keepdims=True),
                           [1, num_users, 1, 1]) - edge_fbb_pro_3) / (num_users - 1) \
                  + edge_wrf_pro_2
        edge_fbb_new = tf.matmul(edge_fbb, edge_fbb_com) + process_norm * pro_inf

        return edge_wrf_new, edge_h_new, edge_frf_new, edge_fbb_new


def model_edge_gnn_layer(edge_wrf, edge_h, edge_frf, edge_fbb, H, FNN_hidden, user_num_an, bs_num_an, bs_num_rf,
                         num_users, bn, training, out_dim, process_norm, FNN_type, hidden_activation, scope,
                         initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                         initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        if FNN_type == 'linear':
            activation_fnn = None
        else:
            activation_fnn = hidden_activation
        batch = tf.shape(edge_wrf)[0]
        pre_dim = edge_wrf.get_shape().as_list()[3]

        edge_wrf_com = tf.get_variable("edge_wrf_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_h_com = tf.get_variable("edge_h_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_frf_com = tf.get_variable("edge_frf_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_fbb_com = tf.get_variable("edge_fbb_com", shape=[pre_dim, out_dim], initializer=initializer_w)

        edge_wrf_pro_1 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='edge_wrf_pro_1',
                             activation_h=activation_fnn)
        edge_wrf_pro_2 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='edge_wrf_pro_2',
                             activation_h=activation_fnn)
        edge_wrf_pro_3 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='edge_wrf_pro_3',
                             activation_h=activation_fnn)
        edge_wrf_pro_2 = tf.reduce_mean(edge_wrf_pro_2, axis=2, keepdims=True)

        # Model
        c_model = model_compute(H, edge_h)
        edge_h_pro_1 = FNN(c_model, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_1',
                           activation_h=activation_fnn)
        edge_h_pro_2 = FNN(c_model, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_2',
                           activation_h=activation_fnn)
        edge_h_pro_3 = FNN(c_model, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_3',
                           activation_h=activation_fnn)
        edge_h_pro_4 = FNN(c_model, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_4',
                           activation_h=activation_fnn)

        edge_frf_pro_1 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_1',
                             activation_h=activation_fnn)
        edge_frf_pro_2 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_2',
                             activation_h=activation_fnn)
        edge_frf_pro_3 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_3',
                             activation_h=activation_fnn)
        edge_frf_pro_4 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_4',
                             activation_h=activation_fnn)

        edge_fbb_pro_1 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_1',
                             activation_h=activation_fnn)
        edge_fbb_pro_2 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_2',
                             activation_h=activation_fnn)
        edge_fbb_pro_3 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_3',
                             activation_h=activation_fnn)
        edge_fbb_pro_4 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_4',
                             activation_h=activation_fnn)
        edge_fbb_pro_2 = tf.reduce_mean(edge_fbb_pro_2, axis=2, keepdims=True)

        # Update edge_wrf
        pro_inf = tf.reduce_mean(tf.reshape(edge_h_pro_1, [batch, num_users, user_num_an, bs_num_an, out_dim]), axis=3) \
                  + (tf.tile(tf.reduce_sum(edge_wrf_pro_3, axis=2, keepdims=True),
                             [1, 1, user_num_an, 1]) - edge_wrf_pro_3) / (user_num_an - 1) \
                  + edge_fbb_pro_2
        edge_wrf_new = tf.matmul(edge_wrf, edge_wrf_com) + process_norm * pro_inf
        # process_norm * tf.matmul(pro_inf, edge_frf_pro)

        # Update edge_h
        pro_inf = tf.tile(tf.reshape(edge_wrf_pro_1, [batch, num_users * user_num_an, 1, out_dim]),
                          [1, 1, bs_num_an, 1]) + \
                  tf.tile(tf.transpose(tf.reduce_mean(edge_frf_pro_1, axis=2, keepdims=True), [0, 2, 1, 3]),
                          [1, num_users * user_num_an, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_h_pro_3, axis=2, keepdims=True), [1, 1, bs_num_an, 1]) - edge_h_pro_3) / (
                              bs_num_an - 1) + \
                  (tf.tile(tf.reduce_sum(edge_h_pro_4, axis=1, keepdims=True),
                           [1, num_users * user_num_an, 1, 1]) - edge_h_pro_4) / (num_users * user_num_an - 1)
        edge_h_new = tf.matmul(edge_h, edge_h_com) + process_norm * pro_inf

        # Update edge_frf
        pro_inf = tf.tile(tf.transpose(tf.reduce_mean(edge_h_pro_2, axis=1, keepdims=True), [0, 2, 1, 3]),
                          [1, 1, bs_num_rf, 1]) + \
                  tf.tile(tf.reduce_mean(edge_fbb_pro_1, axis=1, keepdims=True), [1, bs_num_an, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_3, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_frf_pro_3) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_4, axis=1, keepdims=True),
                           [1, bs_num_an, 1, 1]) - edge_frf_pro_4) / (bs_num_an - 1)
        edge_frf_new = tf.matmul(edge_frf, edge_frf_com) + process_norm * pro_inf

        # Update edge_fbb
        pro_inf = tf.tile(tf.reduce_mean(edge_frf_pro_2, axis=1, keepdims=True), [1, num_users, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_3, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_fbb_pro_3) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_4, axis=1, keepdims=True),
                           [1, num_users, 1, 1]) - edge_fbb_pro_4) / (num_users - 1) \
                  + edge_wrf_pro_2
        edge_fbb_new = tf.matmul(edge_fbb, edge_fbb_com) + process_norm * pro_inf

        return edge_wrf_new, edge_h_new, edge_frf_new, edge_fbb_new


def complex_matmul_tf(X, Y):
    X_Re = X[..., 0];
    X_Im = X[..., 1]
    Y_Re = Y[..., 0];
    Y_Im = Y[..., 1]

    XY_Re = tf.matmul(X_Re, Y_Re) - tf.matmul(X_Im, Y_Im)
    XY_Im = tf.matmul(X_Re, Y_Im) + tf.matmul(X_Im, Y_Re)

    XY = tf.concat((tf.expand_dims(XY_Re, axis=-1), tf.expand_dims(XY_Im, axis=-1)), axis=-1)

    return XY


def model_compute(feature, hidden):
    # feature: [batch, num_obj1, num_obj2, D]
    # hidden: [batch, num_obj1, num_obj2, D]
    batch = tf.shape(feature)[0]
    num_obj1 = feature.get_shape().as_list()[1]
    num_obj2 = feature.get_shape().as_list()[2]
    d_feature = feature.get_shape().as_list()[-1]
    d_hidden = hidden.get_shape().as_list()[-1]

    feature = tf.transpose(tf.reshape(feature, [batch, num_obj1, num_obj2, int(d_feature / 2), 2]), [0, 3, 1, 2, 4])
    hidden = tf.transpose(tf.reshape(hidden, [batch, num_obj1, num_obj2, int(d_hidden / 2), 2]), [0, 3, 2, 1, 4])

    Alpha = complex_matmul_tf(feature * tf.ones([1, hidden.get_shape().as_list()[1], 1, 1, 1]), hidden)
    B = complex_matmul_tf(hidden, Alpha)
    X = tf.concat((hidden, B), axis=-1)
    X = tf.reshape(tf.transpose(X, [0, 3, 2, 1, 4]), [-1, num_obj1, num_obj2, int(X.shape[1] * X.shape[-1])])

    return X


def CNN(A, CNN_dim, activation_h, scope, bn):
    batch_size = tf.shape(A)[0]
    N1 = tf.shape(A)[1]
    N2 = tf.shape(A)[2]
    D = tf.shape(A)[3]

    A_reshaped = tf.reshape(A, [-1, D, 1])

    conv_out = A_reshaped

    for l, h in enumerate(CNN_dim):
        with tf.variable_scope(scope + str(l), reuse=tf.AUTO_REUSE):
            conv_out = tf.layers.conv1d(
                inputs=conv_out,
                filters=h,
                kernel_size=3,
                padding='same',
                activation=None
            )

    conv_out_pooled = tf.reduce_mean(conv_out, axis=1)
    output = tf.reshape(conv_out_pooled, [batch_size, N1, N2, CNN_dim[-1]])

    if activation_h is not None:
        if bn is True:
            output = tf.layers.batch_normalization(output, training=True)
        output = activation_h(output)

    return output


def homo_CNN_gnn_layer(user_rf, user_an, bs_rf, bs_an, H, CNN_hidden, FNN_hidden, user_num_an, bs_num_an, bs_num_rf,
                       num_users, bn, training, out_dim, process_norm, hidden_activation, scope,
                       initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                       initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        batch = tf.shape(user_rf)[0]
        pre_dim = user_rf.get_shape().as_list()[3]
        combiner = tf.get_variable("combiner", shape=[pre_dim, out_dim], initializer=initializer_w)

        pro_w = tf.get_variable("pro_w", shape=[out_dim, out_dim], initializer=initializer_w)

        user_an_pro = FNN(user_an, FNN_hidden + [out_dim], bn, training=training, scope='processor',
                          activation_h=hidden_activation)

        user_an_edge = tf.concat([tf.transpose(tf.tile(user_an, [1, bs_num_an, 1, 1]), [0, 2, 1, 3]), H], axis=3)
        user_an_edge_pro = FNN(user_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='processor_H',
                               activation_h=hidden_activation)

        bs_an_pro = FNN(bs_an, FNN_hidden + [out_dim], bn, training=training, scope='processor',
                        activation_h=hidden_activation)

        bs_an_edge = tf.concat([tf.tile(bs_an, [1, num_users * user_num_an, 1, 1]), H], axis=3)
        bs_an_edge_pro = FNN(bs_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='processor_H',
                             activation_h=hidden_activation)

        user_rf_pro_1 = FNN(user_rf, FNN_hidden + [out_dim], bn, training=training, scope='processor',
                            activation_h=hidden_activation)
        user_rf_pro_2 = FNN(user_rf, FNN_hidden + [out_dim], bn, training=training, scope='processor',
                            activation_h=hidden_activation)

        bs_rf_pro_1 = FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='processor',
                          activation_h=hidden_activation)
        bs_rf_pro_2 = FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='processor',
                          activation_h=hidden_activation)

        # Update user_an
        pro_inf = tf.reshape(tf.tile(user_rf_pro_1, [1, 1, 1, user_num_an]),
                             [batch, 1, num_users * user_num_an, out_dim]) + \
                  tf.transpose(tf.reduce_mean(bs_an_edge_pro, axis=2, keepdims=True), [0, 2, 1, 3])
        user_an_new = tf.matmul(user_an, combiner) + process_norm * tf.matmul(pro_inf, pro_w)

        # Update bs_an
        pro_inf = tf.reduce_mean(tf.tile(tf.transpose(bs_rf_pro_1, [0, 2, 1, 3]), [1, 1, bs_num_an, 1]), axis=1,
                                 keepdims=True) + \
                  tf.reduce_mean(user_an_edge_pro, axis=1, keepdims=True)
        bs_an_new = tf.matmul(bs_an, combiner) + process_norm * tf.matmul(pro_inf, pro_w)

        # Update user_rf
        pro_inf = tf.transpose(tf.reduce_mean(tf.reshape(user_an_pro, [batch, num_users, user_num_an, out_dim]),
                                              axis=2, keepdims=True), [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_mean(bs_rf_pro_2, axis=2, keepdims=True), [1, 1, num_users, 1])
        user_rf_new = tf.matmul(user_rf, combiner) + process_norm * tf.matmul(pro_inf, pro_w)

        # Update bs_rf
        pro_inf = tf.transpose(tf.reduce_mean(tf.tile(bs_an_pro, [1, bs_num_rf, 1, 1]), axis=2, keepdims=True),
                               [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_mean(user_rf_pro_2, axis=2, keepdims=True), [1, 1, bs_num_rf, 1])
        bs_rf_new = tf.matmul(bs_rf, combiner) + process_norm * tf.matmul(pro_inf, pro_w)

        return user_rf_new, user_an_new, bs_rf_new, bs_an_new


def fdgnn_layer(user_rf, user_an, bs_rf, bs_an, H, FNN_hidden_1, FNN_hidden_2, user_num_an, bs_num_an, bs_num_rf,
                num_users, bn, training, out_dim, process_norm, hidden_activation_1, hidden_activation_2, scope,
                initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        batch = tf.shape(user_rf)[0]

        ones_1 = tf.ones([batch, 1, user_num_an * num_users, 2])
        user_an_pro = FNN(tf.concat([user_an, ones_1], axis=-1), FNN_hidden_1, bn, training=training,
                          scope='processor', activation_h=hidden_activation_1)

        user_an_edge = tf.concat([tf.transpose(tf.tile(user_an, [1, bs_num_an, 1, 1]), [0, 2, 1, 3]), H], axis=3)
        user_an_edge_pro = FNN(user_an_edge, FNN_hidden_1, bn, training=training, scope='processor',
                               activation_h=hidden_activation_1)

        ones_2 = tf.ones([batch, 1, bs_num_an, 2])
        bs_an_pro = FNN(tf.concat([bs_an, ones_2], axis=-1), FNN_hidden_1, bn, training=training,
                        scope='processor', activation_h=hidden_activation_1)

        bs_an_edge = tf.concat([tf.tile(bs_an, [1, num_users * user_num_an, 1, 1]), H], axis=3)
        bs_an_edge_pro = FNN(bs_an_edge, FNN_hidden_1, bn, training=training, scope='processor',
                             activation_h=hidden_activation_1)

        ones_3 = tf.ones([batch, 1, num_users, 2])
        user_rf_pro_1 = FNN(tf.concat([user_rf, ones_3], axis=-1), FNN_hidden_1, bn, training=training,
                            scope='processor', activation_h=hidden_activation_1)
        user_rf_pro_2 = FNN(tf.concat([user_rf, ones_3], axis=-1), FNN_hidden_1, bn, training=training,
                            scope='processor', activation_h=hidden_activation_1)

        ones_4 = tf.ones([batch, 1, bs_num_rf, 2])
        bs_rf_pro_1 = FNN(tf.concat([bs_rf, ones_4], axis=-1), FNN_hidden_1, bn, training=training,
                          scope='processor', activation_h=hidden_activation_1)
        bs_rf_pro_2 = FNN(tf.concat([bs_rf, ones_4], axis=-1), FNN_hidden_1, bn, training=training,
                          scope='processor', activation_h=hidden_activation_1)

        # Update user_an
        pro_inf = tf.reshape(tf.tile(user_rf_pro_1, [1, 1, 1, user_num_an]), [batch, 1, num_users * user_num_an,
                                                                              FNN_hidden_1[-1]]) + \
                  tf.transpose(tf.reduce_mean(bs_an_edge_pro, axis=2, keepdims=True), [0, 2, 1, 3])
        user_an_new = FNN(tf.concat([user_an, pro_inf], axis=-1), FNN_hidden_2, bn, training=training,
                          scope='combiner', activation_h=hidden_activation_2)

        # Update bs_an
        pro_inf = tf.reduce_mean(tf.tile(tf.transpose(bs_rf_pro_1, [0, 2, 1, 3]), [1, 1, bs_num_an, 1]), axis=1,
                                 keepdims=True) + \
                  tf.reduce_mean(user_an_edge_pro, axis=1, keepdims=True)
        bs_an_new = FNN(tf.concat([bs_an, pro_inf], axis=-1), FNN_hidden_2, bn, training=training,
                        scope='combiner', activation_h=hidden_activation_2)

        # Update user_rf
        pro_inf = tf.transpose(
            tf.reduce_mean(tf.reshape(user_an_pro, [batch, num_users, user_num_an, FNN_hidden_1[-1]]),
                           axis=2, keepdims=True), [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_mean(bs_rf_pro_2, axis=2, keepdims=True), [1, 1, num_users, 1])
        user_rf_new = FNN(tf.concat([user_rf, pro_inf], axis=-1), FNN_hidden_2, bn, training=training,
                          scope='combiner', activation_h=hidden_activation_2)

        # Update bs_rf
        pro_inf = tf.transpose(tf.reduce_mean(tf.tile(bs_an_pro, [1, bs_num_rf, 1, 1]), axis=2, keepdims=True),
                               [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_mean(user_rf_pro_2, axis=2, keepdims=True), [1, 1, bs_num_rf, 1])
        bs_rf_new = FNN(tf.concat([bs_rf, pro_inf], axis=-1), FNN_hidden_2, bn, training=training,
                        scope='combiner', activation_h=hidden_activation_2)

        return user_rf_new, user_an_new, bs_rf_new, bs_an_new


def CNN(A, stride, filters, CNN_dim, activation_h, scope, bn, training,
        initializer_w=tf.keras.initializers.VarianceScaling(mode="fan_in")):
    batch_size = tf.shape(A)[0]
    N1 = A.get_shape()[1]
    N2 = A.get_shape()[2]
    D = A.get_shape()[-1]

    A_reshaped = tf.reshape(A, [batch_size * N1 * N2, D, 1])

    conv_out = A_reshaped

    for l, h in enumerate(CNN_dim):
        with tf.variable_scope(scope + str(l), reuse=tf.AUTO_REUSE):
            if l == 0:
                W = tf.get_variable('W' + str(l), shape=[filters[l], 1, h], initializer=initializer_w)
            else:
                W = tf.get_variable('W' + str(l), shape=[filters[l], CNN_dim[l - 1], h], initializer=initializer_w)
            conv_out = tf.nn.conv1d(conv_out, W, stride=stride[l], padding='VALID', dilations=1)
            if activation_h is not None:
                if bn is True:
                    conv_out = tf.layers.batch_normalization(conv_out, training=training)
                conv_out = activation_h(conv_out)

    output = tf.reshape(conv_out, [batch_size, N1, N2, D * CNN_dim[-1]])

    return output


def cnn_gnn_layer(user_rf, user_an, bs_rf, bs_an, H, stride, filters, CNN_dim, FNN_hidden_2, user_num_an, bs_num_an,
                  bs_num_rf, num_users, bn, training, out_dim, process_norm, hidden_activation_1, hidden_activation_2,
                  scope):
    with tf.variable_scope(scope):
        batch = tf.shape(user_rf)[0]

        ones_1 = tf.ones([batch, 1, user_num_an * num_users, 2])
        user_an_pro = CNN(tf.concat([user_an, ones_1], axis=-1), stride, filters, CNN_dim, bn=bn, training=training,
                          scope='processor', activation_h=hidden_activation_1)

        user_an_edge = tf.concat([tf.transpose(tf.tile(user_an, [1, bs_num_an, 1, 1]), [0, 2, 1, 3]), H], axis=3)
        user_an_edge_pro = CNN(user_an_edge, stride, filters, CNN_dim, bn=bn, training=training,
                               scope='processor', activation_h=hidden_activation_1)

        ones_2 = tf.ones([batch, 1, bs_num_an, 2])
        bs_an_pro = CNN(tf.concat([bs_an, ones_2], axis=-1), stride, filters, CNN_dim, bn=bn, training=training,
                        scope='processor', activation_h=hidden_activation_1)

        bs_an_edge = tf.concat([tf.tile(bs_an, [1, num_users * user_num_an, 1, 1]), H], axis=3)
        bs_an_edge_pro = CNN(bs_an_edge, stride, filters, CNN_dim, bn=bn, training=training,
                             scope='processor', activation_h=hidden_activation_1)

        ones_3 = tf.ones([batch, 1, num_users, 2])
        user_rf_pro_1 = CNN(tf.concat([user_rf, ones_3], axis=-1), stride, filters, CNN_dim, bn=bn, training=training,
                            scope='processor', activation_h=hidden_activation_1)
        user_rf_pro_2 = CNN(tf.concat([user_rf, ones_3], axis=-1), stride, filters, CNN_dim, bn=bn, training=training,
                            scope='processor', activation_h=hidden_activation_1)

        ones_4 = tf.ones([batch, 1, bs_num_rf, 2])
        bs_rf_pro_1 = CNN(tf.concat([bs_rf, ones_4], axis=-1), stride, filters, CNN_dim, bn=bn, training=training,
                          scope='processor', activation_h=hidden_activation_1)
        bs_rf_pro_2 = CNN(tf.concat([bs_rf, ones_4], axis=-1), stride, filters, CNN_dim, bn=bn, training=training,
                          scope='processor', activation_h=hidden_activation_1)

        # Update user_an
        pro_inf = tf.reshape(tf.tile(user_rf_pro_1, [1, 1, 1, user_num_an]), [batch, 1, num_users * user_num_an,
                                                                              (2 + user_rf.get_shape()[-1]) * CNN_dim[
                                                                                  -1]]) + \
                  tf.transpose(tf.reduce_mean(bs_an_edge_pro, axis=2, keepdims=True), [0, 2, 1, 3])
        user_an_new = FNN(tf.concat([user_an, pro_inf], axis=-1), FNN_hidden_2, bn, training=training,
                          scope='combiner', activation_h=hidden_activation_2)

        # Update bs_an
        pro_inf = tf.reduce_mean(tf.tile(tf.transpose(bs_rf_pro_1, [0, 2, 1, 3]), [1, 1, bs_num_an, 1]), axis=1,
                                 keepdims=True) + \
                  tf.reduce_mean(user_an_edge_pro, axis=1, keepdims=True)
        bs_an_new = FNN(tf.concat([bs_an, pro_inf], axis=-1), FNN_hidden_2, bn, training=training,
                        scope='combiner', activation_h=hidden_activation_2)

        # Update user_rf
        pro_inf = tf.transpose(tf.reduce_mean(tf.reshape(user_an_pro, [batch, num_users, user_num_an,
                                                                       (2 + user_an.get_shape()[-1]) * CNN_dim[-1]]),
                                              axis=2, keepdims=True), [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_mean(bs_rf_pro_2, axis=2, keepdims=True), [1, 1, num_users, 1])
        user_rf_new = FNN(tf.concat([user_rf, pro_inf], axis=-1), FNN_hidden_2, bn, training=training,
                          scope='combiner', activation_h=hidden_activation_2)

        # Update bs_rf
        pro_inf = tf.transpose(tf.reduce_mean(tf.tile(bs_an_pro, [1, bs_num_rf, 1, 1]), axis=2, keepdims=True),
                               [0, 2, 1, 3]) \
                  + tf.tile(tf.reduce_mean(user_rf_pro_2, axis=2, keepdims=True), [1, 1, bs_num_rf, 1])
        bs_rf_new = FNN(tf.concat([bs_rf, pro_inf], axis=-1), FNN_hidden_2, bn, training=training,
                        scope='combiner', activation_h=hidden_activation_2)

        return user_rf_new, user_an_new, bs_rf_new, bs_an_new


def edge_gnn_layer_three_edge_type(edge_wrf, edge_h, edge_frf, edge_fbb, H, FNN_hidden, user_num_an, bs_num_an,
                                   bs_num_rf,
                                   num_users, bn, training, out_dim, process_norm, FNN_type, hidden_activation, scope,
                                   initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1),
                                                                                       distribution='uniform'),
                                   initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        if FNN_type == 'linear':
            activation_fnn = None
        else:
            activation_fnn = hidden_activation
        batch = tf.shape(edge_wrf)[0]
        pre_dim = edge_wrf.get_shape().as_list()[3]

        edge_wrf_com = tf.get_variable("edge_wrf_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_h_com = tf.get_variable("edge_h_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_fbb_com = tf.get_variable("edge_fbb_com", shape=[pre_dim, out_dim], initializer=initializer_w)

        edge_wrf_pro_1 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='edge_wrf_pro_1',
                             activation_h=activation_fnn)
        edge_wrf_pro_2 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='edge_wrf_pro_2',
                             activation_h=activation_fnn)
        edge_wrf_pro_3 = FNN(edge_wrf, FNN_hidden + [out_dim], bn, training=training, scope='edge_wrf_pro_3',
                             activation_h=activation_fnn)
        edge_wrf_pro_2 = tf.reduce_mean(edge_wrf_pro_2, axis=2, keepdims=True)

        edge_h_pro_1 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_1',
                           activation_h=activation_fnn)
        edge_h_pro_2 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_rf_h_pro',
                           activation_h=activation_fnn)
        edge_h_pro_3 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_3',
                           activation_h=activation_fnn)
        edge_h_pro_4 = FNN(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_rf_h_pro',
                           activation_h=activation_fnn)

        edge_frf_pro_1 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_rf_h_pro',
                             activation_h=activation_fnn)
        edge_frf_pro_2 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_2',
                             activation_h=activation_fnn)
        edge_frf_pro_3 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_rf_h_pro',
                             activation_h=activation_fnn)
        edge_frf_pro_4 = FNN(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_4',
                             activation_h=activation_fnn)

        edge_fbb_pro_1 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_1',
                             activation_h=activation_fnn)
        edge_fbb_pro_2 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_2',
                             activation_h=activation_fnn)
        edge_fbb_pro_3 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_3',
                             activation_h=activation_fnn)
        edge_fbb_pro_4 = FNN(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_4',
                             activation_h=activation_fnn)
        edge_fbb_pro_2 = tf.reduce_mean(edge_fbb_pro_2, axis=2, keepdims=True)

        # Update edge_wrf
        pro_inf = tf.reduce_mean(tf.reshape(edge_h_pro_1, [batch, num_users, user_num_an, bs_num_an, out_dim]), axis=3) \
                  + (tf.tile(tf.reduce_sum(edge_wrf_pro_3, axis=2, keepdims=True),
                             [1, 1, user_num_an, 1]) - edge_wrf_pro_3) / (user_num_an - 1) \
                  + edge_fbb_pro_2
        edge_wrf_new = tf.matmul(edge_wrf, edge_wrf_com) + process_norm * pro_inf
        # process_norm * tf.matmul(pro_inf, edge_frf_pro)

        # Update edge_h
        pro_inf = tf.tile(tf.reshape(edge_wrf_pro_1, [batch, num_users * user_num_an, 1, out_dim]),
                          [1, 1, bs_num_an, 1]) + \
                  tf.tile(tf.transpose(tf.reduce_mean(edge_frf_pro_1, axis=2, keepdims=True), [0, 2, 1, 3]),
                          [1, num_users * user_num_an, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_h_pro_3, axis=2, keepdims=True), [1, 1, bs_num_an, 1]) - edge_h_pro_3) / (
                              bs_num_an - 1) + \
                  (tf.tile(tf.reduce_sum(edge_h_pro_4, axis=1, keepdims=True),
                           [1, num_users * user_num_an, 1, 1]) - edge_h_pro_4) / (num_users * user_num_an - 1)
        edge_h_new = tf.matmul(edge_h, edge_h_com) + process_norm * pro_inf

        # Update edge_frf
        pro_inf = tf.tile(tf.transpose(tf.reduce_mean(edge_h_pro_2, axis=1, keepdims=True), [0, 2, 1, 3]),
                          [1, 1, bs_num_rf, 1]) + \
                  tf.tile(tf.reduce_mean(edge_fbb_pro_1, axis=1, keepdims=True), [1, bs_num_an, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_3, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_frf_pro_3) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_4, axis=1, keepdims=True),
                           [1, bs_num_an, 1, 1]) - edge_frf_pro_4) / (bs_num_an - 1)
        edge_frf_new = tf.matmul(edge_frf, edge_h_com) + process_norm * pro_inf

        # Update edge_fbb
        pro_inf = tf.tile(tf.reduce_mean(edge_frf_pro_2, axis=1, keepdims=True), [1, num_users, 1, 1]) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_3, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_fbb_pro_3) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_4, axis=1, keepdims=True),
                           [1, num_users, 1, 1]) - edge_fbb_pro_4) / (num_users - 1) \
                  + edge_wrf_pro_2
        edge_fbb_new = tf.matmul(edge_fbb, edge_fbb_com) + process_norm * pro_inf

        return edge_wrf_new, edge_h_new, edge_frf_new, edge_fbb_new
