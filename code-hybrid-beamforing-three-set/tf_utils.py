import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()


def complex_modulus(x):
    output = tf.reduce_sum(tf.square(x), axis=3, keepdims=True)
    return output


def complex_modulus_all(x):
    output = tf.reduce_sum(tf.square(x), axis=[1, 2, 3], keepdims=True)
    return output


def complex_multiply(x, y):
    output = tf.stack([tf.linalg.matmul(x[:, :, :, 0], y[:, :, :, 0]) - tf.linalg.matmul(x[:, :, :, 1], y[:, :, :, 1]),
                       tf.linalg.matmul(x[:, :, :, 0], y[:, :, :, 1]) + tf.linalg.matmul(x[:, :, :, 1], y[:, :, :, 0])],
                      axis=3)
    return output


def complex_multiply_high(x, y):
    output = tf.stack([tf.linalg.matmul(x[:, :, :, :, 0], y[:, :, :, :, 0]) -
                       tf.linalg.matmul(x[:, :, :, :, 1], y[:, :, :, :, 1]),
                       tf.linalg.matmul(x[:, :, :, :, 0], y[:, :, :, :, 1]) +
                       tf.linalg.matmul(x[:, :, :, :, 1], y[:, :, :, :, 0])],
                      axis=4)
    return output


def complex_H(x):
    x_1, x_2 = tf.split(x, [1, 1], axis=3)
    x = tf.concat([x_1, -1 * x_2], axis=3)
    return x


def select_top_diag_columns(A, M):
    import tensorflow as tf
    B = tf.reduce_sum(A, axis=3)
    diag_elements = tf.linalg.diag_part(B)

    _, topk_indices = tf.nn.top_k(diag_elements, k=M)

    def gather_columns(a_sample, indices_sample):
        return tf.gather(a_sample, indices_sample, axis=1)

    selected_columns = tf.map_fn(
        lambda x: gather_columns(x[0], x[1]),
        (A, topk_indices),
        dtype=tf.float32
    )

    return selected_columns


def fc(x, n_output, bn=False, training=False, scope="fc", activation_fn=None,
       initializer_w=tf.keras.initializers.VarianceScaling(mode="fan_in"),
       initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
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


def fc_no_bias(x, n_output, bn=False, training=False, scope="fc", activation_fn=None,
               initializer_w=tf.keras.initializers.VarianceScaling(mode="fan_in"),
               initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        W = tf.get_variable("W", shape=[x.get_shape()[-1], n_output], initializer=initializer_w)
        fc = tf.matmul(x, W)
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
    x_shape = x.get_shape()
    batch = tf.shape(x)[0]
    d1 = x_shape[1]
    d2 = x_shape[2]
    n_input = int(x.get_shape()[-1])
    with tf.variable_scope(scope):
        w1 = tf.get_variable("w1", shape=[n_input, n_output], initializer=initializer_w) * para
        w2 = tf.get_variable("w2", shape=[n_input, n_output], initializer=initializer_w) * para
        w3 = tf.get_variable("w3", shape=[n_input, n_output], initializer=initializer_w) * para
        w4 = tf.get_variable("w4", shape=[n_input, n_output], initializer=initializer_w)
        b = tf.get_variable("b", shape=[n_output], initializer=initializer_b) * para

        u1 = tf.matmul(x, w1)
        u2 = tf.matmul(x, w2)
        u3 = tf.matmul(x, w3)
        u4 = tf.matmul(x, w4)

        v1 = tf.tile(tf.reduce_sum(u1, axis=2, keepdims=True), [1, 1, d2, 1]) - u1 + u2
        v2 = tf.tile(tf.reduce_sum(u3, axis=2, keepdims=True), [1, 1, d2, 1]) - u3 + u4

        output = tf.tile(tf.reduce_sum(v1, axis=1, keepdims=True), [1, d1, 1, 1]) - v1 + v2 + b
        if activation_fn == '(tanh+1)*0.5':
            if bn is True:
                output = tf.layers.batch_normalization(output, training=training, momentum=0.99, epsilon=0.00001)
            output = 0.5 * (tf.tanh(output) + 1)
        elif activation_fn is not None:
            if bn is True:
                output = tf.layers.batch_normalization(output, training=training, momentum=0.99, epsilon=0.00001)
            output = activation_fn(output)
    return output


def penn_layer_new(x, shade_mat, n_output, para, bs_num_an, user_dim, bn=False, training=False, scope="fc",
                   activation_fn=None,
                   initializer_w=tf.keras.initializers.VarianceScaling(mode="fan_in"),
                   initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    x_shape = x.get_shape()
    batch = tf.shape(x)[0]
    d1 = x_shape[1]
    d2 = x_shape[2]
    n_input = int(x.get_shape()[-1])
    with tf.variable_scope(scope):
        w1 = tf.get_variable("w1", shape=[n_input, n_output], initializer=initializer_w) * para
        w2 = tf.get_variable("w2", shape=[n_input, n_output], initializer=initializer_w) * para
        w3 = tf.get_variable("w3", shape=[n_input, n_output], initializer=initializer_w) * para
        w4 = tf.get_variable("w4", shape=[n_input, n_output], initializer=initializer_w)
        b = tf.get_variable("b", shape=[n_output], initializer=initializer_b) * para

        u1 = tf.matmul(x, w1)
        u2 = tf.matmul(x, w2)
        u3 = tf.matmul(x, w3)
        u4 = tf.matmul(x, w4)

        if user_dim == 2:
            shade = tf.reduce_sum(shade_mat, axis=1, keepdims=True) + 1e-5
            shade = tf.transpose(shade, [0, 2, 1, 3])
            shade = tf.tile(shade, [1, 1, d2, 1])
            v1 = (tf.tile(tf.reduce_sum(u1, axis=2, keepdims=True), [1, 1, d2, 1]) - u1) / shade + u2
            v2 = (tf.tile(tf.reduce_sum(u3, axis=2, keepdims=True), [1, 1, d2, 1]) - u3) / shade + u4

            output = (tf.tile(tf.reduce_sum(v1, axis=1, keepdims=True), [1, d1, 1, 1]) - v1) / (bs_num_an - 1) + v2 + b
        elif user_dim == 1:
            shade = tf.reduce_sum(shade_mat, axis=1, keepdims=True) + 1e-5
            shade = tf.tile(shade, [1, d1, 1, 1])
            v1 = (tf.tile(tf.reduce_sum(u1, axis=2, keepdims=True), [1, 1, d2, 1]) - u1) / (bs_num_an - 1) + u2
            v2 = (tf.tile(tf.reduce_sum(u3, axis=2, keepdims=True), [1, 1, d2, 1]) - u3) / (bs_num_an - 1) + u4

            output = (tf.tile(tf.reduce_sum(v1, axis=1, keepdims=True), [1, d1, 1, 1]) - v1) / shade + v2 + b

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
    return tf.nn.max_pool(x, ksize=[1, k_sz[0], k_sz[1], 1], strides=[1, k_sz[0], k_sz[1], 1], padding=padding)


def conv2d(x, n_filter, k_sz, bn, activation_fn, scope="con", stride=np.array([1, 1]), training=False,
           initializer_w=tf.keras.initializers.VarianceScaling(mode="fan_in"),
           initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
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
    sh = x.shape
    # return tf.reshape(x, [-1, sh[1] * sh[2] * sh[3]])
    return tf.layers.flatten(x)


def FNN(x, hidden_size, bn=False, scope='FNN', training=False, activation_h=tf.nn.softmax, output_h=tf.nn.softmax,
        initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
        initializer_b=tf.zeros_initializer()):
    for i, h in enumerate(hidden_size[:-1]):
        x = fc(x, h, scope=scope + str(i), initializer_w=initializer_w, initializer_b=initializer_b)
        if activation_h is not None:
            if bn:
                x = tf.layers.batch_normalization(x, training=training)
            x = activation_h(x)
    x = fc(x, hidden_size[-1], scope=scope + '_out', initializer_w=initializer_w, initializer_b=initializer_b)
    if output_h is not None:
        if bn:
            x = tf.layers.batch_normalization(x, training=training)
        x = output_h(x)
    return x


def FNN_no_bias(x, hidden_size, bn=False, scope='FNN', training=False, activation_h=tf.nn.softmax,
                initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                initializer_b=tf.zeros_initializer()):
    for i, h in enumerate(hidden_size[:]):
        x = fc_no_bias(x, h, scope=scope + str(i), initializer_w=initializer_w, initializer_b=initializer_b)
        if activation_h is not None:
            if bn:
                x = tf.layers.batch_normalization(x, training=training)
            x = activation_h(x)
    return x


def vertex_gnn_layer(user, bs_rf, bs_an, H, FNN_hidden, bs_num_an, bs_num_rf, num_users, bn, training, out_dim,
                     process_norm, hidden_activation, scope,
                     initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                     initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        batch = tf.shape(user)[0]
        pre_dim = user.get_shape().as_list()[3]
        user_com = tf.get_variable("user_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        bs_rf_com = tf.get_variable("bs_rf_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        bs_an_com = tf.get_variable("bs_an_com", shape=[pre_dim, out_dim], initializer=initializer_w)

        user_pro_w = tf.get_variable("user_pro_w", shape=[out_dim, out_dim], initializer=initializer_w)
        bs_rf_pro_w = tf.get_variable("bs_rf_pro_w", shape=[out_dim, out_dim], initializer=initializer_w)
        bs_an_pro_w = tf.get_variable("bs_an_pro_w", shape=[out_dim, out_dim], initializer=initializer_w)

        user_pro = FNN(user, FNN_hidden + [out_dim], bn, training=training, scope='user_pro',
                       activation_h=hidden_activation)

        user_edge = tf.concat([tf.transpose(tf.tile(user, [1, bs_num_an, 1, 1]), [0, 2, 1, 3]), H], axis=3)
        user_edge_pro = FNN(user_edge, FNN_hidden + [out_dim], bn, training=training, scope='user_edge_pro',
                            activation_h=hidden_activation)

        bs_an_pro = FNN(bs_an, FNN_hidden + [out_dim], bn, training=training, scope='bs_an_pro',
                        activation_h=hidden_activation)

        bs_an_edge = tf.concat([tf.tile(bs_an, [1, num_users, 1, 1]), H], axis=3)
        bs_an_edge_pro = FNN(bs_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='bs_an_edge_pro',
                             activation_h=hidden_activation)

        bs_rf_pro_1 = FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='bs_rf_pro_1',
                          activation_h=hidden_activation)
        bs_rf_pro_2 = FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='bs_rf_pro_2',
                          activation_h=hidden_activation)

        # Update user
        pro_inf = tf.reduce_mean(bs_rf_pro_2, axis=2, keepdims=True) + \
                  tf.transpose(tf.reduce_mean(bs_an_edge_pro, axis=2, keepdims=True), [0, 2, 1, 3])
        user_new = tf.matmul(user, user_com) + process_norm * tf.matmul(pro_inf, user_pro_w)

        # Update bs_an
        pro_inf = tf.reduce_mean(tf.tile(tf.transpose(bs_rf_pro_1, [0, 2, 1, 3]), [1, 1, bs_num_an, 1]),
                                 axis=1, keepdims=True) + tf.reduce_mean(user_edge_pro, axis=1, keepdims=True)
        bs_an_new = tf.matmul(bs_an, bs_an_com) + process_norm * tf.matmul(pro_inf, bs_an_pro_w)

        # Update bs_rf
        pro_inf = tf.reduce_mean(user_pro, axis=2, keepdims=True) + \
                  tf.transpose(tf.reduce_mean(tf.tile(bs_an_pro, [1, bs_num_rf, 1, 1]), axis=2, keepdims=True),
                               [0, 2, 1, 3])
        bs_rf_new = tf.matmul(bs_rf, bs_rf_com) + process_norm * tf.matmul(pro_inf, bs_rf_pro_w)

        return user_new, bs_rf_new, bs_an_new


def vertex_gnn_layer_1(user, bs_rf, bs_an, H, FNN_hidden, bs_num_an, bs_num_rf, num_users, bn, training, out_dim,
                       process_norm, hidden_activation, scope,
                       initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                       initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        user_pro = FNN(user, FNN_hidden + [out_dim], bn, training=training, scope='user_pro',
                       activation_h=hidden_activation)

        user_edge = tf.concat([tf.transpose(tf.tile(user, [1, bs_num_an, 1, 1]), [0, 2, 1, 3]), H], axis=3)
        user_edge_pro = FNN(user_edge, FNN_hidden + [out_dim], bn, training=training, scope='user_edge_pro',
                            activation_h=hidden_activation)

        bs_an_pro = FNN(bs_an, FNN_hidden + [out_dim], bn, training=training, scope='bs_an_pro',
                        activation_h=hidden_activation)

        bs_an_edge = tf.concat([tf.tile(bs_an, [1, num_users, 1, 1]), H], axis=3)
        bs_an_edge_pro = FNN(bs_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='bs_an_edge_pro',
                             activation_h=hidden_activation)

        bs_rf_pro_1 = FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='bs_rf_pro_1',
                          activation_h=hidden_activation)
        bs_rf_pro_2 = FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='bs_rf_pro_2',
                          activation_h=hidden_activation)

        # Update user
        pro_inf = tf.concat([user, tf.tile(tf.reduce_mean(bs_rf_pro_2, axis=2, keepdims=True), [1, 1, num_users, 1]),
                             tf.transpose(tf.reduce_mean(bs_an_edge_pro, axis=2, keepdims=True), [0, 2, 1, 3])],
                            axis=-1)
        user_new = FNN(pro_inf, FNN_hidden + [out_dim], bn, training=training, scope='user_com',
                       activation_h=hidden_activation)

        # Update bs_an
        pro_inf = tf.concat([bs_an,
                             tf.reduce_mean(tf.tile(tf.transpose(bs_rf_pro_1, [0, 2, 1, 3]), [1, 1, bs_num_an, 1]),
                                            axis=1, keepdims=True),
                             tf.reduce_mean(user_edge_pro, axis=1, keepdims=True)], axis=-1)
        bs_an_new = FNN(pro_inf, FNN_hidden + [out_dim], bn, training=training, scope='bs_an_com',
                        activation_h=hidden_activation)

        # Update bs_rf
        pro_inf = tf.concat([bs_rf,
                             tf.tile(tf.reduce_mean(user_pro, axis=2, keepdims=True), [1, 1, bs_num_rf, 1]),
                             tf.transpose(tf.reduce_mean(tf.tile(bs_an_pro, [1, bs_num_rf, 1, 1]), axis=2,
                                                         keepdims=True), [0, 2, 1, 3])],
                            axis=-1)
        bs_rf_new = FNN(pro_inf, FNN_hidden + [out_dim], bn, training=training, scope='bs_rf_com',
                        activation_h=hidden_activation)

        return user_new, bs_rf_new, bs_an_new


def vertex_gnn_layer_2(user, bs_rf, bs_an, H, FNN_hidden, bs_num_an, bs_num_rf, num_users, bn, training, out_dim,
                       process_norm, hidden_activation, scope,
                       initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                       initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        batch = tf.shape(user)[0]
        pre_dim = user.get_shape().as_list()[3]
        user_com = tf.get_variable("user_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        bs_rf_com = tf.get_variable("bs_rf_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        bs_an_com = tf.get_variable("bs_an_com", shape=[pre_dim, out_dim], initializer=initializer_w)

        user_pro_w_1 = tf.get_variable("user_pro_w_1", shape=[out_dim, out_dim], initializer=initializer_w)
        user_pro_w_2 = tf.get_variable("user_pro_w_2", shape=[out_dim, out_dim], initializer=initializer_w)
        bs_rf_pro_w_1 = tf.get_variable("bs_rf_pro_w_1", shape=[out_dim, out_dim], initializer=initializer_w)
        bs_rf_pro_w_2 = tf.get_variable("bs_rf_pro_w_2", shape=[out_dim, out_dim], initializer=initializer_w)
        bs_an_pro_w_1 = tf.get_variable("bs_an_pro_w_1", shape=[out_dim, out_dim], initializer=initializer_w)
        bs_an_pro_w_2 = tf.get_variable("bs_an_pro_w_2", shape=[out_dim, out_dim], initializer=initializer_w)

        user_pro = FNN(user, FNN_hidden + [out_dim], bn, training=training, scope='user_pro',
                       activation_h=hidden_activation)

        user_edge = tf.concat([tf.transpose(tf.tile(user, [1, bs_num_an, 1, 1]), [0, 2, 1, 3]), H], axis=3)
        user_edge_pro = FNN(user_edge, FNN_hidden + [out_dim], bn, training=training, scope='user_edge_pro',
                            activation_h=hidden_activation)

        bs_an_pro = FNN(bs_an, FNN_hidden + [out_dim], bn, training=training, scope='bs_an_pro',
                        activation_h=hidden_activation)

        bs_an_edge = tf.concat([tf.tile(bs_an, [1, num_users, 1, 1]), H], axis=3)
        bs_an_edge_pro = FNN(bs_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='bs_an_edge_pro',
                             activation_h=hidden_activation)

        bs_rf_pro_1 = FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='bs_rf_pro_1',
                          activation_h=hidden_activation)
        bs_rf_pro_2 = FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='bs_rf_pro_2',
                          activation_h=hidden_activation)

        # Update user
        pro_inf = tf.matmul(tf.reduce_mean(bs_rf_pro_2, axis=2, keepdims=True), user_pro_w_1) + \
                  tf.matmul(tf.transpose(tf.reduce_mean(bs_an_edge_pro, axis=2, keepdims=True), [0, 2, 1, 3]),
                            user_pro_w_2)
        user_new = tf.matmul(user, user_com) + process_norm * pro_inf

        # Update bs_an
        pro_inf = tf.matmul(tf.reduce_mean(tf.tile(tf.transpose(bs_rf_pro_1, [0, 2, 1, 3]), [1, 1, bs_num_an, 1]),
                                           axis=1, keepdims=True), bs_rf_pro_w_1) + \
                  tf.matmul(tf.reduce_mean(user_edge_pro, axis=1, keepdims=True), bs_rf_pro_w_2)
        bs_an_new = tf.matmul(bs_an, bs_an_com) + process_norm * pro_inf

        # Update bs_rf
        pro_inf = tf.matmul(tf.reduce_mean(user_pro, axis=2, keepdims=True), bs_an_pro_w_1) + \
                  tf.matmul(tf.transpose(tf.reduce_mean(tf.tile(bs_an_pro, [1, bs_num_rf, 1, 1]),
                                                        axis=2, keepdims=True), [0, 2, 1, 3]),
                            bs_an_pro_w_2)
        bs_rf_new = tf.matmul(bs_rf, bs_rf_com) + process_norm * pro_inf

        return user_new, bs_rf_new, bs_an_new


def vanilla_het_vertex_gnn_layer(user, bs_rf, bs_an, H, FNN_hidden, bs_num_an, bs_num_rf, num_users, bn, training,
                                 out_dim, hidden_activation, scope,
                                 initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1),
                                                                                     distribution='uniform'),
                                 initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        batch = tf.shape(user)[0]
        pre_dim = user.get_shape().as_list()[3]
        com_1 = tf.get_variable("com_1", shape=[pre_dim, out_dim], initializer=initializer_w)
        com_2 = tf.get_variable("com_2", shape=[pre_dim, out_dim], initializer=initializer_w)

        pro_w_1 = tf.get_variable("pro_w_1", shape=[out_dim, out_dim], initializer=initializer_w)
        pro_w_2 = tf.get_variable("pro_w_2", shape=[out_dim, out_dim], initializer=initializer_w)

        ones_1 = tf.ones([batch, 1, num_users, 2])
        ones_2 = tf.ones([batch, 1, bs_num_an, 2])
        ones_3 = tf.ones([batch, 1, bs_num_rf, 2])

        user_pro = FNN(tf.concat([user, ones_1], axis=-1), FNN_hidden + [out_dim], bn, training=training, scope='agg_1',
                       activation_h=hidden_activation)

        user_edge = tf.concat([tf.transpose(tf.tile(user, [1, bs_num_an, 1, 1]), [0, 2, 1, 3]), H], axis=3)
        user_edge_pro = FNN(user_edge, FNN_hidden + [out_dim], bn, training=training, scope='agg_1',
                            activation_h=hidden_activation)

        bs_an_pro = FNN(tf.concat([bs_an, ones_2], axis=-1), FNN_hidden + [out_dim], bn, training=training,
                        scope='agg_1', activation_h=hidden_activation)

        bs_an_edge = tf.concat([tf.tile(bs_an, [1, num_users, 1, 1]), H], axis=3)
        bs_an_edge_pro = FNN(bs_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='agg_2',
                             activation_h=hidden_activation)

        bs_rf_pro_1 = FNN(tf.concat([bs_rf, ones_3], axis=-1), FNN_hidden + [out_dim], bn, training=training,
                          scope='agg_1', activation_h=hidden_activation)
        bs_rf_pro_2 = FNN(tf.concat([bs_rf, ones_3], axis=-1), FNN_hidden + [out_dim], bn, training=training,
                          scope='agg_2', activation_h=hidden_activation)

        # Update user
        pro_inf = tf.matmul(tf.reduce_mean(bs_rf_pro_2, axis=2, keepdims=True), pro_w_2) + \
                  tf.matmul(tf.transpose(tf.reduce_mean(bs_an_edge_pro, axis=2, keepdims=True), [0, 2, 1, 3]),
                            pro_w_2)
        user_new = tf.matmul(user, com_2) + pro_inf

        # Update bs_an
        pro_inf = tf.matmul(tf.reduce_mean(tf.tile(tf.transpose(bs_rf_pro_1, [0, 2, 1, 3]), [1, 1, bs_num_an, 1]),
                                           axis=1, keepdims=True), pro_w_1) + \
                  tf.matmul(tf.reduce_mean(user_edge_pro, axis=1, keepdims=True), pro_w_1)
        bs_an_new = tf.matmul(bs_an, com_1) + pro_inf

        # Update bs_rf
        pro_inf = tf.matmul(tf.reduce_mean(user_pro, axis=2, keepdims=True), pro_w_1) + \
                  tf.matmul(tf.transpose(tf.reduce_mean(tf.tile(bs_an_pro, [1, bs_num_rf, 1, 1]),
                                                        axis=2, keepdims=True), [0, 2, 1, 3]),
                            pro_w_1)
        bs_rf_new = tf.matmul(bs_rf, com_1) + pro_inf

        return user_new, bs_rf_new, bs_an_new


def pgnn_layer(user, bs_rf, bs_an, H, FNN_hidden, bs_num_an, bs_num_rf, num_users, bn, training,
               out_dim, hidden_activation, scope,
               initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
               initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        batch = tf.shape(user)[0]
        pre_dim = user.get_shape().as_list()[3]
        com_1 = tf.get_variable("com_1", shape=[pre_dim, out_dim], initializer=initializer_w)
        com_2 = tf.get_variable("com_2", shape=[pre_dim, out_dim], initializer=initializer_w)

        pro_w_1 = tf.get_variable("pro_w_1", shape=[out_dim, out_dim], initializer=initializer_w)
        pro_w_2 = tf.get_variable("pro_w_2", shape=[out_dim, out_dim], initializer=initializer_w)

        ones_1 = tf.ones([batch, 1, num_users, 2])
        ones_2 = tf.ones([batch, 1, bs_num_an, 2])
        ones_3 = tf.ones([batch, 1, bs_num_rf, 2])

        user_pro = FNN(tf.concat([user, ones_1], axis=-1), FNN_hidden + [out_dim], bn, training=training, scope='agg_2',
                       activation_h=hidden_activation)

        user_edge = tf.concat([tf.transpose(tf.tile(user, [1, bs_num_an, 1, 1]), [0, 2, 1, 3]), H], axis=3)
        user_edge_pro = FNN(user_edge, FNN_hidden + [out_dim], bn, training=training, scope='agg_1',
                            activation_h=hidden_activation)

        bs_an_pro = FNN(tf.concat([bs_an, ones_2], axis=-1), FNN_hidden + [out_dim], bn, training=training,
                        scope='agg_1', activation_h=hidden_activation)

        bs_an_edge = tf.concat([tf.tile(bs_an, [1, num_users, 1, 1]), H], axis=3)
        bs_an_edge_pro = FNN(bs_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='agg_3',
                             activation_h=hidden_activation)

        bs_rf_pro_1 = FNN(tf.concat([bs_rf, ones_3], axis=-1), FNN_hidden + [out_dim], bn, training=training,
                          scope='agg_2', activation_h=hidden_activation)
        bs_rf_pro_2 = FNN(tf.concat([bs_rf, ones_3], axis=-1), FNN_hidden + [out_dim], bn, training=training,
                          scope='agg_4', activation_h=hidden_activation)

        # Update user
        pro_inf = tf.matmul(tf.reduce_mean(bs_rf_pro_2, axis=2, keepdims=True), pro_w_2) + \
                  tf.matmul(tf.transpose(tf.reduce_mean(bs_an_edge_pro, axis=2, keepdims=True), [0, 2, 1, 3]),
                            pro_w_2)
        user_new = tf.matmul(user, com_2) + pro_inf

        # Update bs_an
        pro_inf = tf.matmul(tf.reduce_mean(tf.tile(tf.transpose(bs_rf_pro_1, [0, 2, 1, 3]), [1, 1, bs_num_an, 1]),
                                           axis=1, keepdims=True), pro_w_1) + \
                  tf.matmul(tf.reduce_mean(user_edge_pro, axis=1, keepdims=True), pro_w_1)
        bs_an_new = tf.matmul(bs_an, com_1) + pro_inf

        # Update bs_rf
        pro_inf = tf.matmul(tf.reduce_mean(user_pro, axis=2, keepdims=True), pro_w_1) + \
                  tf.matmul(tf.transpose(tf.reduce_mean(tf.tile(bs_an_pro, [1, bs_num_rf, 1, 1]),
                                                        axis=2, keepdims=True), [0, 2, 1, 3]),
                            pro_w_1)
        bs_rf_new = tf.matmul(bs_rf, com_1) + pro_inf

        return user_new, bs_rf_new, bs_an_new


def edge_gnn_layer_1(edge_h, edge_frf, edge_fbb, H, FNN_hidden, bs_num_an, bs_num_rf, num_users, bn, training,
                     out_dim, process_norm, FNN_type, hidden_activation, scope,
                     initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                     initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        if FNN_type == 'linear':
            activation_fnn = None
        else:
            activation_fnn = hidden_activation
        batch = tf.shape(edge_h)[0]
        pre_dim = edge_h.get_shape().as_list()[3]

        edge_h_com = tf.get_variable("edge_h_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_frf_com = tf.get_variable("edge_frf_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_fbb_com = tf.get_variable("edge_fbb_com", shape=[pre_dim, out_dim], initializer=initializer_w)

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

        # Update edge_h
        pro_inf = tf.reduce_mean(edge_fbb_pro_2, axis=2, keepdims=True) + \
                  tf.transpose(tf.reduce_mean(edge_frf_pro_1, axis=2, keepdims=True), [0, 2, 1, 3]) + \
                  (tf.tile(tf.reduce_sum(edge_h_pro_3, axis=2, keepdims=True), [1, 1, bs_num_an, 1]) - edge_h_pro_3) / (
                          bs_num_an - 1) + \
                  (tf.tile(tf.reduce_sum(edge_h_pro_4, axis=1, keepdims=True), [1, num_users, 1, 1]) - edge_h_pro_4) / (
                          num_users - 1)
        edge_h_new = tf.matmul(edge_h, edge_h_com) + process_norm * pro_inf

        # Update edge_frf
        pro_inf = tf.transpose(tf.reduce_mean(edge_h_pro_2, axis=1, keepdims=True), [0, 2, 1, 3]) + \
                  tf.reduce_mean(edge_fbb_pro_1, axis=1, keepdims=True) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_3, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_frf_pro_3) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_4, axis=1, keepdims=True),
                           [1, bs_num_an, 1, 1]) - edge_frf_pro_4) / (bs_num_an - 1)
        edge_frf_new = tf.matmul(edge_frf, edge_frf_com) + process_norm * pro_inf

        # Update edge_fbb
        pro_inf = tf.reduce_mean(edge_frf_pro_2, axis=1, keepdims=True) + \
                  tf.reduce_mean(edge_h_pro_1, axis=2, keepdims=True) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_3, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_fbb_pro_3) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_4, axis=1, keepdims=True),
                           [1, num_users, 1, 1]) - edge_fbb_pro_4) / (num_users - 1)
        edge_fbb_new = tf.matmul(edge_fbb, edge_fbb_com) + process_norm * pro_inf

        return edge_h_new, edge_frf_new, edge_fbb_new


def edge_gnn_layer_2(edge_h, edge_frf, edge_fbb, H, FNN_hidden, bs_num_an, bs_num_rf, num_users, bn, training,
                     out_dim, process_norm, FNN_type, hidden_activation, scope,
                     initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                     initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        if FNN_type == 'linear':
            activation_fnn = None
        else:
            activation_fnn = hidden_activation
        batch = tf.shape(edge_h)[0]
        pre_dim = edge_h.get_shape().as_list()[3]

        edge_h_com = tf.get_variable("edge_h_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_frf_com = tf.get_variable("edge_frf_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_fbb_com = tf.get_variable("edge_fbb_com", shape=[pre_dim, out_dim], initializer=initializer_w)

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

        # Update edge_h
        pro_inf = tf.reduce_mean(edge_fbb_pro_2, axis=2, keepdims=True) + \
                  tf.transpose(tf.reduce_mean(edge_frf_pro_1, axis=2, keepdims=True), [0, 2, 1, 3])
        pro_inf_1 = (tf.tile(tf.reduce_sum(edge_h_pro_3, axis=2, keepdims=True),
                             [1, 1, bs_num_an, 1]) - edge_h_pro_3) / (bs_num_an - 1) + \
                    (tf.tile(tf.reduce_sum(edge_h_pro_4, axis=1, keepdims=True),
                             [1, num_users, 1, 1]) - edge_h_pro_4) / (num_users - 1)
        edge_h_new = tf.matmul(edge_h, edge_h_com) + pro_inf_1 + process_norm * pro_inf

        # Update edge_frf
        pro_inf = tf.transpose(tf.reduce_mean(edge_h_pro_2, axis=1, keepdims=True), [0, 2, 1, 3]) + \
                  tf.reduce_mean(edge_fbb_pro_1, axis=1, keepdims=True)
        pro_inf_1 = (tf.tile(tf.reduce_sum(edge_frf_pro_3, axis=2, keepdims=True),
                             [1, 1, bs_num_rf, 1]) - edge_frf_pro_3) / (bs_num_rf - 1) + \
                    (tf.tile(tf.reduce_sum(edge_frf_pro_4, axis=1, keepdims=True),
                             [1, bs_num_an, 1, 1]) - edge_frf_pro_4) / (bs_num_an - 1)
        edge_frf_new = tf.matmul(edge_frf, edge_frf_com) + pro_inf_1 + process_norm * pro_inf

        # Update edge_fbb
        pro_inf = tf.reduce_mean(edge_frf_pro_2, axis=1, keepdims=True) + \
                  tf.reduce_mean(edge_h_pro_1, axis=2, keepdims=True)
        pro_inf_1 = (tf.tile(tf.reduce_sum(edge_fbb_pro_3, axis=2, keepdims=True),
                             [1, 1, bs_num_rf, 1]) - edge_fbb_pro_3) / (bs_num_rf - 1) + \
                    (tf.tile(tf.reduce_sum(edge_fbb_pro_4, axis=1, keepdims=True),
                             [1, num_users, 1, 1]) - edge_fbb_pro_4) / (num_users - 1)
        edge_fbb_new = tf.matmul(edge_fbb, edge_fbb_com) + pro_inf_1 + process_norm * pro_inf

        return edge_h_new, edge_frf_new, edge_fbb_new


def edge_gnn_layer_3(edge_h, edge_frf, edge_fbb, H, FNN_hidden, bs_num_an, bs_num_rf, num_users, bn, training,
                     out_dim, process_norm, FNN_type, hidden_activation, scope,
                     initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                     initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        if FNN_type == 'linear':
            activation_fnn = None
        else:
            activation_fnn = hidden_activation
        batch = tf.shape(edge_h)[0]
        pre_dim = edge_h.get_shape().as_list()[3]

        edge_h_com = tf.get_variable("edge_h_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_frf_com = tf.get_variable("edge_frf_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_fbb_com = tf.get_variable("edge_fbb_com", shape=[pre_dim, out_dim], initializer=initializer_w)

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

        # Update edge_h
        pro_inf = tf.reduce_mean(edge_fbb_pro_2, axis=2, keepdims=True) + \
                  tf.transpose(tf.reduce_mean(edge_frf_pro_1, axis=2, keepdims=True), [0, 2, 1, 3]) + \
                  (tf.tile(tf.reduce_sum(edge_h_pro_3, axis=2, keepdims=True), [1, 1, bs_num_an, 1]) - edge_h_pro_3) / (
                          bs_num_an - 1) + \
                  (tf.tile(tf.reduce_sum(edge_h_pro_4, axis=1, keepdims=True), [1, num_users, 1, 1]) - edge_h_pro_4) / (
                          num_users - 1)
        pro_H = FNN(H, FNN_hidden + [out_dim], bn, training=training, scope='H_pro',
                    activation_h=activation_fnn)
        edge_h_new = tf.matmul(edge_h, edge_h_com) + process_norm * pro_inf + pro_H

        # Update edge_frf
        pro_inf = tf.transpose(tf.reduce_mean(edge_h_pro_2, axis=1, keepdims=True), [0, 2, 1, 3]) + \
                  tf.reduce_mean(edge_fbb_pro_1, axis=1, keepdims=True) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_3, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_frf_pro_3) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_4, axis=1, keepdims=True),
                           [1, bs_num_an, 1, 1]) - edge_frf_pro_4) / (bs_num_an - 1)
        edge_frf_new = tf.matmul(edge_frf, edge_frf_com) + process_norm * pro_inf

        # Update edge_fbb
        pro_inf = tf.reduce_mean(edge_frf_pro_2, axis=1, keepdims=True) + \
                  tf.reduce_mean(edge_h_pro_1, axis=2, keepdims=True) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_3, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_fbb_pro_3) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_4, axis=1, keepdims=True),
                           [1, num_users, 1, 1]) - edge_fbb_pro_4) / (num_users - 1)
        edge_fbb_new = tf.matmul(edge_fbb, edge_fbb_com) + process_norm * pro_inf

        return edge_h_new, edge_frf_new, edge_fbb_new


def edge_gnn_layer_4(edge_h, edge_frf, edge_fbb, H, FNN_hidden, bs_num_an, bs_num_rf, num_users, bn, training,
                     out_dim, process_norm, FNN_type, hidden_activation, scope,
                     initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                     initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        if FNN_type == 'linear':
            activation_fnn = None
        else:
            activation_fnn = hidden_activation
        batch = tf.shape(edge_h)[0]
        pre_dim = edge_h.get_shape().as_list()[3]

        edge_h_com = tf.get_variable("edge_h_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_frf_com = tf.get_variable("edge_frf_com", shape=[pre_dim, out_dim], initializer=initializer_w)
        edge_fbb_com = tf.get_variable("edge_fbb_com", shape=[pre_dim, out_dim], initializer=initializer_w)

        edge_h_pro_1 = FNN_no_bias(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_1',
                                   activation_h=activation_fnn)
        edge_h_pro_2 = FNN_no_bias(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_2',
                                   activation_h=activation_fnn)
        edge_h_pro_3 = FNN_no_bias(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_3',
                                   activation_h=activation_fnn)
        edge_h_pro_4 = FNN_no_bias(edge_h, FNN_hidden + [out_dim], bn, training=training, scope='edge_h_pro_4',
                                   activation_h=activation_fnn)

        edge_frf_pro_1 = FNN_no_bias(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_1',
                                     activation_h=activation_fnn)
        edge_frf_pro_2 = FNN_no_bias(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_2',
                                     activation_h=activation_fnn)
        edge_frf_pro_3 = FNN_no_bias(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_3',
                                     activation_h=activation_fnn)
        edge_frf_pro_4 = FNN_no_bias(edge_frf, FNN_hidden + [out_dim], bn, training=training, scope='edge_frf_pro_4',
                                     activation_h=activation_fnn)

        edge_fbb_pro_1 = FNN_no_bias(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_1',
                                     activation_h=activation_fnn)
        edge_fbb_pro_2 = FNN_no_bias(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_2',
                                     activation_h=activation_fnn)
        edge_fbb_pro_3 = FNN_no_bias(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_3',
                                     activation_h=activation_fnn)
        edge_fbb_pro_4 = FNN_no_bias(edge_fbb, FNN_hidden + [out_dim], bn, training=training, scope='edge_fbb_pro_4',
                                     activation_h=activation_fnn)

        # Update edge_h
        pro_inf = tf.reduce_mean(edge_fbb_pro_2, axis=2, keepdims=True) + \
                  tf.transpose(tf.reduce_mean(edge_frf_pro_1, axis=2, keepdims=True), [0, 2, 1, 3]) + \
                  (tf.tile(tf.reduce_sum(edge_h_pro_3, axis=2, keepdims=True), [1, 1, bs_num_an, 1]) - edge_h_pro_3) / (
                          bs_num_an - 1) + \
                  (tf.tile(tf.reduce_sum(edge_h_pro_4, axis=1, keepdims=True), [1, num_users, 1, 1]) - edge_h_pro_4) / (
                          num_users - 1)
        edge_h_new = tf.matmul(edge_h, edge_h_com) + process_norm * pro_inf

        # Update edge_frf
        pro_inf = tf.transpose(tf.reduce_mean(edge_h_pro_2, axis=1, keepdims=True), [0, 2, 1, 3]) + \
                  tf.reduce_mean(edge_fbb_pro_1, axis=1, keepdims=True) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_3, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_frf_pro_3) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_frf_pro_4, axis=1, keepdims=True),
                           [1, bs_num_an, 1, 1]) - edge_frf_pro_4) / (bs_num_an - 1)
        edge_frf_new = tf.matmul(edge_frf, edge_frf_com) + process_norm * pro_inf

        # Update edge_fbb
        pro_inf = tf.reduce_mean(edge_frf_pro_2, axis=1, keepdims=True) + \
                  tf.reduce_mean(edge_h_pro_1, axis=2, keepdims=True) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_3, axis=2, keepdims=True),
                           [1, 1, bs_num_rf, 1]) - edge_fbb_pro_3) / (bs_num_rf - 1) + \
                  (tf.tile(tf.reduce_sum(edge_fbb_pro_4, axis=1, keepdims=True),
                           [1, num_users, 1, 1]) - edge_fbb_pro_4) / (num_users - 1)
        edge_fbb_new = tf.matmul(edge_fbb, edge_fbb_com) + process_norm * pro_inf

        return edge_h_new, edge_frf_new, edge_fbb_new


def mdgnn_layer_new(H, out_shape, scope="gnn",
                    initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform')):
    input_shape = H.get_shape().as_list()
    in_shape = input_shape[-1]
    with tf.variable_scope(scope):
        w1 = tf.get_variable("w1", shape=[in_shape, out_shape], initializer=initializer_w)
        w2 = tf.get_variable("w2", shape=[in_shape, out_shape], initializer=initializer_w)
        w3 = tf.get_variable("w3", shape=[in_shape, out_shape], initializer=initializer_w)
        w4 = tf.get_variable("w4", shape=[in_shape, out_shape], initializer=initializer_w)

        H_1 = tf.matmul(H, w1)
        H_2 = tf.matmul(H, w2)
        H_3 = tf.matmul(H, w3)
        H_4 = tf.matmul(H, w4)

        output = tf.reduce_mean(H_1, axis=1, keepdims=True) + tf.reduce_mean(H_2, axis=2, keepdims=True) + \
                 tf.reduce_mean(H_3, axis=3, keepdims=True) + H_4

        return output


def mdgnn_layer(real_H, im_H, bs_rf, bn, train_ph, out_shape, hidden_activation, process_norm=1.0, scope="gnn"):
    with tf.variable_scope(scope):
        # update real_H
        real_H_new = mdgnn_layer_channel(real_H, out_shape, hidden_activation, bn, train_ph, scope='real_H-real_H') + \
                     mdgnn_layer_channel(im_H, out_shape, hidden_activation, bn, train_ph, scope='im_H-real_H') + \
                     mdgnn_layer_channel(bs_rf, out_shape, hidden_activation, bn, train_ph, scope='bs_rf-real_H')
        if hidden_activation is not None:
            if bn:
                real_H_new = tf.layers.batch_normalization(real_H_new, training=train_ph)
            real_H_new = hidden_activation(real_H_new)

        # update im_H
        im_H_new = mdgnn_layer_channel(real_H, out_shape, hidden_activation, bn, train_ph, scope='real_H-im_H') + \
                   mdgnn_layer_channel(im_H, out_shape, hidden_activation, bn, train_ph, scope='im_H-im_H') + \
                   mdgnn_layer_channel(bs_rf, out_shape, hidden_activation, bn, train_ph, scope='bs_rf-im_H')
        if hidden_activation is not None:
            if bn:
                im_H_new = tf.layers.batch_normalization(im_H_new, training=train_ph)
            im_H_new = hidden_activation(im_H_new)

        # update bs_rf
        bs_rf_new = mdgnn_layer_channel(real_H, out_shape, hidden_activation, bn, train_ph, scope='real_H-bs_rf') + \
                    mdgnn_layer_channel(im_H, out_shape, hidden_activation, bn, train_ph, scope='im_H-bs_rf') + \
                    mdgnn_layer_channel(bs_rf, out_shape, hidden_activation, bn, train_ph, scope='bs_rf-bs_rf')
        if hidden_activation is not None:
            if bn:
                bs_rf_new = tf.layers.batch_normalization(bs_rf_new, training=train_ph)
            bs_rf_new = hidden_activation(bs_rf_new)

    return real_H_new, im_H_new, bs_rf_new


def mdgnn_layer_channel(input, out_shape, hidden_activation, bn, train_ph, process_norm=1.0, scope="mdgnn",
                        initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform')):
    input_shape = input.get_shape().as_list()
    in_shape = input_shape[-1]
    with tf.variable_scope(scope):
        w1 = tf.get_variable("w1", shape=[in_shape, out_shape], initializer=initializer_w)
        w2 = tf.get_variable("w2", shape=[in_shape, out_shape], initializer=initializer_w)
        w3 = tf.get_variable("w3", shape=[in_shape, out_shape], initializer=initializer_w)
        w4 = tf.get_variable("w4", shape=[in_shape, out_shape], initializer=initializer_w)

        pro_1 = tf.matmul(input, w1)
        pro_2 = tf.matmul(input, w2)
        pro_3 = tf.matmul(input, w3)
        pro_4 = tf.matmul(input, w4)

        output = tf.reduce_mean(pro_1, axis=1, keepdims=True) + \
                 tf.reduce_mean(pro_2, axis=2, keepdims=True) + \
                 tf.reduce_mean(pro_3, axis=3, keepdims=True) + \
                 pro_4

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


def f_vertex_gnn_layer(user, bs_rf, bs_an, H, FNN_hidden, bs_num_an, bs_num_rf, num_users, bn, training, out_dim,
                       process_norm, hidden_activation, scope,
                       initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                       initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        batch = tf.shape(user)[0]
        pre_dim = user.get_shape().as_list()[3]

        user_pro_w = tf.get_variable("user_pro_w", shape=[out_dim, out_dim], initializer=initializer_w)
        bs_rf_pro_w = tf.get_variable("bs_rf_pro_w", shape=[out_dim, out_dim], initializer=initializer_w)
        bs_an_pro_w = tf.get_variable("bs_an_pro_w", shape=[out_dim, out_dim], initializer=initializer_w)

        user_pro = f_FNN(user, FNN_hidden + [out_dim], bn, training=training, scope='user_pro',
                         activation_h=hidden_activation)

        user_edge = tf.concat([tf.transpose(tf.tile(user, [1, bs_num_an, 1, 1]), [0, 2, 1, 3]), H], axis=3)
        user_edge_pro = f_FNN(user_edge, FNN_hidden + [out_dim], bn, training=training, scope='user_edge_pro',
                              activation_h=hidden_activation)

        bs_an_pro = f_FNN(bs_an, FNN_hidden + [out_dim], bn, training=training, scope='bs_an_pro',
                          activation_h=hidden_activation)

        bs_an_edge = tf.concat([tf.tile(bs_an, [1, num_users, 1, 1]), H], axis=3)
        bs_an_edge_pro = f_FNN(bs_an_edge, FNN_hidden + [out_dim], bn, training=training, scope='bs_an_edge_pro',
                               activation_h=hidden_activation)

        bs_rf_pro_1 = f_FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='bs_rf_pro_1',
                            activation_h=hidden_activation)
        bs_rf_pro_2 = f_FNN(bs_rf, FNN_hidden + [out_dim], bn, training=training, scope='bs_rf_pro_2',
                            activation_h=hidden_activation)

        # Combine
        user_shape = user.get_shape().as_list()
        bs_rf_shape = bs_rf.get_shape().as_list()
        bs_an_shape = bs_an.get_shape().as_list()

        user = tf.expand_dims(user, axis=-2)
        bs_rf = tf.expand_dims(bs_rf, axis=-2)
        bs_an = tf.expand_dims(bs_an, axis=-2)

        user_com = tf.tile(tf.get_variable("user_com", shape=[1, user_shape[1], user_shape[2], pre_dim, out_dim],
                                           initializer=initializer_w), [batch, 1, 1, 1, 1])
        bs_rf_com = tf.tile(tf.get_variable("bs_rf_com", shape=[1, bs_rf_shape[1], bs_rf_shape[2], pre_dim, out_dim],
                                            initializer=initializer_w), [batch, 1, 1, 1, 1])
        bs_an_com = tf.tile(tf.get_variable("bs_an_com", shape=[1, bs_an_shape[1], bs_an_shape[2], pre_dim, out_dim],
                                            initializer=initializer_w), [batch, 1, 1, 1, 1])

        user_combine = tf.reshape(tf.matmul(user, user_com), [batch, user_shape[1], user_shape[2], out_dim])
        bs_rf_combine = tf.reshape(tf.matmul(bs_rf, bs_rf_com), [batch, bs_rf_shape[1], bs_rf_shape[2], out_dim])
        bs_an_combine = tf.reshape(tf.matmul(bs_an, bs_an_com), [batch, bs_an_shape[1], bs_an_shape[2], out_dim])

        # Update user
        pro_inf = tf.reduce_mean(bs_rf_pro_2, axis=2, keepdims=True) + \
                  tf.transpose(tf.reduce_mean(bs_an_edge_pro, axis=2, keepdims=True), [0, 2, 1, 3])
        user_new = user_combine + process_norm * tf.matmul(pro_inf, user_pro_w)

        # Update bs_an
        pro_inf = tf.reduce_mean(tf.tile(tf.transpose(bs_rf_pro_1, [0, 2, 1, 3]), [1, 1, bs_num_an, 1]),
                                 axis=1, keepdims=True) + tf.reduce_mean(user_edge_pro, axis=1, keepdims=True)
        bs_an_new = bs_an_combine + process_norm * tf.matmul(pro_inf, bs_an_pro_w)

        # Update bs_rf
        pro_inf = tf.reduce_mean(user_pro, axis=2, keepdims=True) + \
                  tf.transpose(tf.reduce_mean(tf.tile(bs_an_pro, [1, bs_num_rf, 1, 1]), axis=2, keepdims=True),
                               [0, 2, 1, 3])
        bs_rf_new = bs_rf_combine + process_norm * tf.matmul(pro_inf, bs_rf_pro_w)

        return user_new, bs_rf_new, bs_an_new


def ini_variable(shape, name, is_train=False, initializer=tf.constant_initializer(0.1)):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('bns', reuse=tf.AUTO_REUSE):
            variable = tf.get_variable(name='bias', shape=shape,
                                       initializer=initializer, trainable=is_train)
    return variable


def ini_bias(out_shape, name, is_train=True, initializer=tf.constant_initializer(0.1)):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('biases', reuse=tf.AUTO_REUSE):
            bias_vec = tf.get_variable(name='bias', shape=[out_shape],
                                       initializer=initializer, trainable=is_train)
    return bias_vec


def batch_normalization_size_gen(X, A, name, is_train=True, decay=0.9):
    """
    X: 输入张量，形状：(None, ..., object特征数)
    A: shade_mat，形状：(None, ..., object特征数)
    name: batch_normalization 变量名
    """
    axis_list = [i for i in range(len(X.shape) - 1)]

    X_mean = tf.reduce_sum(X, axis=axis_list, keepdims=True) / (tf.reduce_sum(A, axis=axis_list, keepdims=True) + 1e-6)
    X_std = tf.math.sqrt(tf.reduce_sum(tf.pow(X - X_mean, 2), axis=axis_list, keepdims=True) /
                         (tf.reduce_sum(A, axis=axis_list, keepdims=True) + 1e-6) + 1e-6)
    pop_mean = ini_variable(X_mean.shape, name=name + 'meanBN', is_train=False, initializer=tf.zeros_initializer())
    pop_std = ini_variable(X_std.shape, name=name + 'stdBN', is_train=False, initializer=tf.ones_initializer())
    beta = ini_bias(int(X.shape[-1]), name=name + '_betaBN', initializer=tf.zeros_initializer())
    gamma = ini_bias(int(X.shape[-1]), name=name + '_gammaBN', initializer=tf.ones_initializer())

    def f1(X):
        X_mean1 = tf.assign(pop_mean, pop_mean * decay + X_mean * (1 - decay))
        X_std1 = tf.assign(pop_std, pop_std * decay + X_std * (1 - decay))
        with tf.control_dependencies([X_mean1, X_std1]):
            X = (X - X_mean) / X_std * gamma + beta
            X = X * A
        return X

    def f2(X):
        X_mean = pop_mean;
        X_std = pop_std
        X = (X - X_mean) / X_std * gamma + beta
        X = X * A
        return X

    X = tf.cond(is_train, lambda: f1(X), lambda: f2(X))

    return X


def GNN_2d_layer(X, A, output_shape, scope, is_train=True,
                 initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform')):
    # X: [batch, N1, N2, d]: feature
    # A: [batch, N1, N2, 1]: edges in graph
    input_shape = X.get_shape().as_list()[-1]
    N1 = X.get_shape().as_list()[1]
    N2 = X.get_shape().as_list()[2]

    with tf.variable_scope(scope):
        W_com = tf.get_variable("W_com", shape=[input_shape, output_shape], initializer=initializer_w)
        W_pro_1 = tf.get_variable("W_pro_1", shape=[input_shape, output_shape], initializer=initializer_w)
        W_pro_2 = tf.get_variable("W_pro_2", shape=[input_shape, output_shape], initializer=initializer_w)

        X_com = tf.matmul(X, W_com)
        X_1 = tf.matmul(X, W_pro_1)
        X_2 = tf.matmul(X, W_pro_2)

        pro_1 = tf.divide(tf.reduce_sum(X_1, axis=1, keepdims=True) - X_1,
                          tf.tile(tf.reduce_sum(A, axis=1, keepdims=True) - 1 + 1e-6, [1, N1, 1, output_shape]))

        pro_2 = tf.divide(tf.reduce_sum(X_2, axis=2, keepdims=True) - X_2,
                          tf.tile(tf.reduce_sum(A, axis=2, keepdims=True) - 1 + 1e-6, [1, 1, N2, output_shape]))

        # pro_1 = tf.reduce_max(X_1, axis=1, keepdims=True)
        #
        # pro_2 = tf.reduce_max(X_2, axis=2, keepdims=True)

        X = tf.multiply(X_com + pro_1 + pro_2, tf.tile(A, [1, 1, 1, output_shape]))

        return X


def model_GNN_2d_layer(X, A, output_shape, scope, K_factor,
                       initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                       initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    # X: [batch, N1, N2, d]: feature
    # A: [batch, N1, N2, 1]: edges in graph
    input_shape = X.get_shape().as_list()[-1]
    N1 = X.get_shape().as_list()[1]
    N2 = X.get_shape().as_list()[2]

    with tf.variable_scope(scope):
        W_com = tf.get_variable("W_com", shape=[input_shape, output_shape], initializer=initializer_w)
        W_pro_1 = tf.get_variable("W_pro_1", shape=[input_shape, output_shape], initializer=initializer_w)
        W_pro_2 = tf.get_variable("W_pro_2", shape=[input_shape, output_shape], initializer=initializer_w)
        W_pro_3 = tf.get_variable("W_pro_3", shape=[input_shape, output_shape], initializer=initializer_w)
        b = tf.get_variable("bias", shape=[output_shape], initializer=initializer_b)

        # X_com = tf.matmul(X, W_com)
        # X_1 = tf.matmul(X, W_pro_1) * K_factor
        # X_2 = tf.matmul(X, W_pro_2) * K_factor

        X_com = tf.matmul(X, W_com - W_pro_1 - W_pro_2 + W_pro_3)
        X_1 = tf.matmul(X, W_pro_1 - W_pro_3) * K_factor
        X_2 = tf.matmul(X, W_pro_2 - W_pro_3) * K_factor

        pro_1 = tf.divide(tf.reduce_sum(X_1, axis=1, keepdims=True) - X_1,
                          tf.tile(tf.reduce_sum(A, axis=1, keepdims=True) - 1 + 1e-6, [1, N1, 1, output_shape]))

        pro_2 = tf.divide(tf.reduce_sum(X_2, axis=2, keepdims=True) - X_2,
                          tf.tile(tf.reduce_sum(A, axis=2, keepdims=True) - 1 + 1e-6, [1, 1, N2, output_shape]))

        X = tf.multiply(X_com + pro_1 + pro_2 + b, tf.tile(A, [1, 1, 1, output_shape]))

        return X


def model_GNN_scale(X, A, num_hidden_model_GNN, num_hidden_factor_fnn, hidden_ac_factor_fnn, hidden_ac_model_GNN,
                    output_ac_factor_fnn, output_transfer_model_GNN, scope, is_bn_gnn, is_bn_fnn,
                    is_train, k=1, is_mul_K=True):
    N1 = int(X.shape[1]);
    N2 = int(X.shape[2])
    X_Re = tf.transpose(X, [0, 2, 1, 3])[..., 0:1];
    X_Im = -1 * tf.transpose(X, [0, 2, 1, 3])[..., 1:2]
    Xh = tf.concat((X_Re, X_Im), axis=3)
    V_temp = Xh
    H = tf.expand_dims(X, axis=1);
    V_temp = tf.expand_dims(V_temp, axis=1)
    K_ue = tf.reduce_sum(A[:, :, 0, 0], axis=1, keepdims=True)
    Adj_mat = tf.expand_dims(A, axis=1)  # (N_spl, 1, N1, N2, 1)
    Adj_mat_ue = Adj_mat[:, :, :, 0:1, :] * tf.transpose(Adj_mat[:, :, :, 0:1, :], [0, 1, 3, 2, 4])
    Adj_mat_T = tf.transpose(Adj_mat, [0, 1, 3, 2, 4])  # (N_spl, 1, N2, N1, 1)
    with tf.variable_scope(scope):
        for l, h in enumerate(num_hidden_model_GNN):
            Alpha = complex_matmul_tf(H * tf.ones([1, int(V_temp.shape[1]), 1, 1, 1]), V_temp)
            Alpha = Alpha * Adj_mat_ue
            B = complex_matmul_tf(V_temp, Alpha) * k
            X = tf.concat((V_temp, B), axis=-1)
            X = X * Adj_mat_T
            X = tf.reshape(tf.transpose(X, [0, 2, 3, 1, 4]), [-1, N2, N1, int(X.shape[1] * X.shape[-1])])
            if is_mul_K is True:
                K_factor = FNN(K_ue, num_hidden_factor_fnn, bn=is_bn_fnn, scope='Model_GNN_FNN' + str(l),
                               training=is_train, activation_h=hidden_ac_factor_fnn, output_h=output_ac_factor_fnn)
                K_factor = tf.reshape(K_factor, [-1, 1, 1, 1])
            else:
                K_factor = 1.0

            V_temp = model_GNN_2d_layer(X, Adj_mat_T[:, 0, ...], 2 * h, scope='model_GNN_2d_pe' + str(l),
                                        K_factor=K_factor)
            if (hidden_ac_model_GNN is not None) and (l != (len(num_hidden_model_GNN) - 1)):
                if is_bn_gnn:
                    V_temp = tf.layers.batch_normalization(V_temp, training=is_train, name='model_gnn_bn' + str(l))
                V_temp = hidden_ac_model_GNN(V_temp)
                V_temp = V_temp * Adj_mat_T[:, 0, :, :, :]
                V_temp = tf.transpose(tf.reshape(V_temp, [-1, N2, N1, h, 2]), [0, 3, 1, 2, 4])
            elif (hidden_ac_model_GNN is not None) and (l == (len(num_hidden_model_GNN) - 1)):
                if output_transfer_model_GNN:
                    if is_bn_gnn:
                        V_temp = tf.layers.batch_normalization(V_temp, training=is_train, name='model_gnn_bn' + str(l))
                    V_temp = hidden_ac_model_GNN(V_temp)
                    V_temp = V_temp * Adj_mat_T[:, 0, :, :, :]
                V_temp = tf.transpose(tf.reshape(V_temp, [-1, N2, N1, h, 2]), [0, 3, 1, 2, 4])

            if (hidden_ac_model_GNN is None) and (l != (len(num_hidden_model_GNN) - 1)):
                V_temp = tf.transpose(tf.reshape(V_temp, [-1, N2, N1, h, 2]), [0, 3, 1, 2, 4])
                V_temp = V_temp * Adj_mat_T
                V_temp = V_temp / (tf.sqrt(tf.reduce_sum(tf.pow(V_temp, 2), axis=[-1, -2, -3], keepdims=True)) + 1e-6)
            elif (hidden_ac_model_GNN is None) and (l == (len(num_hidden_model_GNN) - 1)):
                if output_transfer_model_GNN:
                    V_temp = tf.transpose(tf.reshape(V_temp, [-1, N2, N1, h, 2]), [0, 3, 1, 2, 4])
                    V_temp = V_temp * Adj_mat_T
                    V_temp = V_temp / (
                            tf.sqrt(tf.reduce_sum(tf.pow(V_temp, 2), axis=[-1, -2, -3], keepdims=True)) + 1e-6)
                else:
                    V_temp = tf.transpose(tf.reshape(V_temp, [-1, N2, N1, h, 2]), [0, 3, 1, 2, 4])

        V_temp = tf.transpose(V_temp[:, 0, ...], [0, 2, 1, 3])  # (N_spl, K, Ntx, 2)

    return V_temp


def complex_matmul_tf(X, Y):
    X_Re = X[..., 0];
    X_Im = X[..., 1]
    Y_Re = Y[..., 0];
    Y_Im = Y[..., 1]

    XY_Re = tf.matmul(X_Re, Y_Re) - tf.matmul(X_Im, Y_Im)
    XY_Im = tf.matmul(X_Re, Y_Im) + tf.matmul(X_Im, Y_Re)

    XY = tf.concat((tf.expand_dims(XY_Re, axis=-1), tf.expand_dims(XY_Im, axis=-1)), axis=-1)

    return XY


def ini_weights_highdim(shape, name, stddev=0.1, is_train=True, initlize='normal'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights', reuse=tf.AUTO_REUSE):
            if initlize == 'normal':
                weight_mat = tf.get_variable(name='weight', shape=shape,
                                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=stddev),
                                             trainable=is_train)
            else:
                weight_mat = tf.get_variable(name='weight', shape=shape,
                                             initializer=tf.zeros_initializer(), trainable=is_train)
    return weight_mat


def add_2d_pe_layer_scale(input, adj_mat, inshape, outshape, name, stddev=0.1, transfer_function=tf.nn.leaky_relu,
                          is_BN=False, is_transfer=True, k_factor=1.0, is_trainBN=False, is_train=True,
                          aggr_func='mean'):
    '''
    :param adj_mat: adj_mat: 邻接矩阵，形状为（N_spl, n_obj1, n_obj2, 1）
    :return:
    '''
    W = ini_weights_highdim([inshape, outshape, 5], name + '_Wh', stddev, is_train=is_train)
    U = W[:, :, 0];
    V = W[:, :, 1];
    P = W[:, :, 2];
    Q = W[:, :, 3];
    M = W[:, :, 4]

    b = ini_bias(outshape, name + '_bias', is_train=is_train)

    if aggr_func == 'mean':
        input_sum_dim1 = tf.reduce_sum(input, axis=-2, keepdims=True) / (
                tf.reduce_sum(adj_mat, axis=-2, keepdims=True) + 1e-5)
        input_sum_dim2 = tf.reduce_sum(input, axis=-3, keepdims=True) / (
                tf.reduce_sum(adj_mat, axis=-3, keepdims=True) + 1e-5)
        input_sum_dim12 = tf.reduce_sum(input, axis=[-2, -3], keepdims=True) / (
                tf.reduce_sum(adj_mat, axis=[-2, -3], keepdims=True) + 1e-5)
    else:
        input_sum_dim1 = tf.reduce_sum(input, axis=-2, keepdims=True)
        input_sum_dim2 = tf.reduce_sum(input, axis=-3, keepdims=True)
        input_sum_dim12 = tf.reduce_sum(input, axis=[-2, -3], keepdims=True)

    output = tf.matmul(input, U - V - P + Q) + \
             tf.matmul(input_sum_dim1, V - Q) * k_factor + \
             tf.matmul(input_sum_dim2, P - Q) * k_factor + b  # \
    # tf.matmul(input_sum_dim12, Q) + b

    if is_transfer is True:
        output = transfer_function(output)
    if is_BN is True:
        output = tf.layers.batch_normalization(output, training=is_trainBN, name=name + '_BN',
                                               reuse=tf.AUTO_REUSE, axis=[-1, -4])
    output = output * adj_mat
    return output


def ini_weights(in_shape, out_shape, name, stddev=0.1, is_train=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights', reuse=tf.AUTO_REUSE):
            weight_mat = tf.get_variable(name='weight', shape=[in_shape, out_shape],
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=stddev),
                                         trainable=is_train)
    return weight_mat


def ini_bias(out_shape, name, is_train=True, initializer=tf.constant_initializer(0.1)):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('biases', reuse=tf.AUTO_REUSE):
            bias_vec = tf.get_variable(name='bias', shape=[out_shape],
                                       initializer=initializer, trainable=is_train)
    return bias_vec


def add_layer(input, inshape, outshape, name, stddev=0.1, keep_prob=1.0, transfer_function=tf.nn.crelu,
              is_transfer=True,
              is_BN=False, is_trainBN=True, is_train=True):
    inshape = int(input.shape[-1])
    W = ini_weights(inshape, outshape, name, stddev, is_train=is_train)
    b = ini_bias(outshape, name, is_train=is_train)
    output = tf.matmul(input, W) + b
    output = tf.nn.dropout(output, keep_prob=keep_prob)
    if is_BN is True:
        output = tf.layers.batch_normalization(output, training=is_trainBN, name=name + '_BN', reuse=tf.AUTO_REUSE)
    if is_transfer is True:
        output = transfer_function(output)
    return output, W, b


def fcnn(input, layernum, output_activation=tf.nn.sigmoid, is_BN=False, is_trainBN=False, name='',
         is_lastlayer_transfer=True, is_train=True):
    hidden = dict()
    weight = dict()
    bias = dict()
    if len(layernum) >= 2:
        for i in range(len(layernum)):
            if i == 0:
                hidden[str(i)], weight[str(i)], bias[str(i)] = \
                    add_layer(input, int(input.shape[-1]), layernum[i], name + 'layer' + str(i + 1), is_BN=is_BN,
                              is_trainBN=is_trainBN, is_train=is_train)
            elif i != len(layernum) - 1:
                hidden[str(i)], weight[str(i)], bias[str(i)] = \
                    add_layer(hidden[str(i - 1)], layernum[i - 1], layernum[i], name + 'layer' + str(i + 1),
                              is_BN=is_BN,
                              is_trainBN=is_trainBN, is_train=is_train)
            else:
                output, weight[str(i)], bias[str(i)] = \
                    add_layer(hidden[str(i - 1)], layernum[i - 1], layernum[i], name + 'layer' + str(i + 1),
                              is_BN=is_BN,
                              transfer_function=output_activation, is_trainBN=is_trainBN,
                              is_transfer=is_lastlayer_transfer,
                              is_train=is_train)
    else:
        output, weight['0'], bias['0'] = \
            add_layer(input, int(input.shape[-1]), layernum[0], name + 'layer1', is_BN=is_BN,
                      transfer_function=output_activation, is_trainBN=is_trainBN, is_transfer=is_lastlayer_transfer,
                      is_train=is_train)
    return output, hidden, bias


def PENN_BF_scale(H, Adj_mat, layernum, hidden_activation, k=1, is_mul_K=True):
    """
    :param H: 信道矩阵，形状为（N_spl, K, Ntx, 2）
    :param: Adj_mat: 邻接矩阵，形状为（N_spl, K, Ntx, 1）
    :param layernum: 每一层的隐藏层节点数，是一个list
    :return:
    """
    K = int(H.shape[1]);
    Ntx = int(H.shape[2])
    Hh_Re = tf.transpose(H, [0, 2, 1, 3])[..., 0:1];
    Hh_Im = -1 * tf.transpose(H, [0, 2, 1, 3])[..., 1:2]
    Hh = tf.concat((Hh_Re, Hh_Im), axis=3)
    V_temp = Hh
    H = tf.expand_dims(H, axis=1);
    V_temp = tf.expand_dims(V_temp, axis=1)
    K_ue = tf.reduce_sum(Adj_mat[:, :, 0, 0], axis=1, keepdims=True)
    Adj_mat = tf.expand_dims(Adj_mat, axis=1)  # (N_spl, 1, K, Ntx, 1)
    Adj_mat_ue = Adj_mat[:, :, :, 0:1, :] * tf.transpose(Adj_mat[:, :, :, 0:1, :], [0, 1, 3, 2, 4])
    Adj_mat_T = tf.transpose(Adj_mat, [0, 1, 3, 2, 4])  # (N_spl, 1, Ntx, K, 1)
    for i in range(len(layernum)):
        Alpha = complex_matmul_tf(H * tf.ones([1, int(V_temp.shape[1]), 1, 1, 1]), V_temp)
        Alpha = Alpha * Adj_mat_ue
        B = complex_matmul_tf(V_temp, Alpha) * k
        X = tf.concat((V_temp, B), axis=-1)
        X = X * Adj_mat_T
        X = tf.reshape(tf.transpose(X, [0, 2, 3, 1, 4]), [-1, Ntx, K, int(X.shape[1] * X.shape[-1])])
        if is_mul_K is True:
            K_factor, _, _ = fcnn(K_ue, [8, 8, 1], name='factor_net_layer_' + str(i), output_activation=tf.nn.relu)
            K_factor = tf.reshape(K_factor, [-1, 1, 1, 1])
        else:
            K_factor = 1.0
        V_new = add_2d_pe_layer_scale(X, Adj_mat_T[:, 0, ...], int(X.shape[-1]), 2 * layernum[i], k_factor=K_factor,
                                      name='penn_bf_scale_layer_' + str(i), transfer_function=hidden_activation,
                                      is_transfer=False)
        V_temp = tf.transpose(tf.reshape(V_new, [-1, Ntx, K, layernum[i], 2]), [0, 3, 1, 2, 4])
        V_temp = V_temp * Adj_mat_T
        V_temp = V_temp / (tf.sqrt(tf.reduce_sum(tf.pow(V_temp, 2), axis=[-1, -2, -3], keepdims=True)) + 1e-6)
        # V_temp = V_temp * Adj_mat_T

    V_temp = tf.transpose(V_temp[:, 0, ...], [0, 2, 1, 3])  # (N_spl, K, Ntx, 2)

    return V_temp


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


def cnn_gnn_layer(user, bs_rf, bs_an, H, stride, filters, CNN_dim, FNN_hidden_2, bs_num_an, bs_num_rf, num_users, bn,
                  training, out_dim, hidden_activation_1, hidden_activation_2, scope,
                  initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                  initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        batch = tf.shape(user)[0]

        ones_1 = tf.ones([batch, 1, num_users, 2])
        ones_2 = tf.ones([batch, 1, bs_num_an, 2])
        ones_3 = tf.ones([batch, 1, bs_num_rf, 2])

        user_pro = CNN(tf.concat([user, ones_1], axis=-1), stride, filters, CNN_dim, bn=bn, training=training,
                       scope='agg',
                       activation_h=hidden_activation_1)

        user_edge = tf.concat([tf.transpose(tf.tile(user, [1, bs_num_an, 1, 1]), [0, 2, 1, 3]), H], axis=3)
        user_edge_pro = CNN(user_edge, stride, filters, CNN_dim, bn=bn, training=training, scope='agg',
                            activation_h=hidden_activation_1)

        bs_an_pro = CNN(tf.concat([bs_an, ones_2], axis=-1), stride, filters, CNN_dim, bn=bn, training=training,
                        scope='agg',
                        activation_h=hidden_activation_1)

        bs_an_edge = tf.concat([tf.tile(bs_an, [1, num_users, 1, 1]), H], axis=3)
        bs_an_edge_pro = CNN(bs_an_edge, stride, filters, CNN_dim, bn=bn, training=training, scope='agg',
                             activation_h=hidden_activation_1)

        bs_rf_pro_1 = CNN(tf.concat([bs_rf, ones_3], axis=-1), stride, filters, CNN_dim, bn=bn, training=training,
                          scope='agg',
                          activation_h=hidden_activation_1)
        bs_rf_pro_2 = CNN(tf.concat([bs_rf, ones_3], axis=-1), stride, filters, CNN_dim, bn=bn, training=training,
                          scope='agg',
                          activation_h=hidden_activation_1)

        # Update user
        pro_inf = tf.reduce_mean(bs_rf_pro_2, axis=2, keepdims=True) + \
                  tf.transpose(tf.reduce_mean(bs_an_edge_pro, axis=2, keepdims=True), [0, 2, 1, 3])
        user_new = FNN(tf.concat([user, pro_inf], axis=-1), FNN_hidden_2, bn, training=training,
                       scope='com', activation_h=hidden_activation_2, output_h=hidden_activation_2)

        # Update bs_an
        pro_inf = tf.reduce_mean(tf.tile(tf.transpose(bs_rf_pro_1, [0, 2, 1, 3]), [1, 1, bs_num_an, 1]),
                                 axis=1, keepdims=True) + \
                  tf.reduce_mean(user_edge_pro, axis=1, keepdims=True)
        bs_an_new = FNN(tf.concat([bs_an, pro_inf], axis=-1), FNN_hidden_2, bn, training=training,
                        scope='com', activation_h=hidden_activation_2, output_h=hidden_activation_2)

        # Update bs_rf
        pro_inf = tf.reduce_mean(user_pro, axis=2, keepdims=True) + \
                  tf.transpose(tf.reduce_mean(tf.tile(bs_an_pro, [1, bs_num_rf, 1, 1]), axis=2, keepdims=True),
                               [0, 2, 1, 3])
        bs_rf_new = FNN(tf.concat([bs_rf, pro_inf], axis=-1), FNN_hidden_2, bn, training=training,
                        scope='com', activation_h=hidden_activation_2, output_h=hidden_activation_2)

        return user_new, bs_rf_new, bs_an_new


def homo_gnn_layer(user, bs_rf, bs_an, H, FNN_hidden_1, FNN_hidden_2, bs_num_an, bs_num_rf, num_users, bn, training,
                   out_dim, hidden_activation_1, hidden_activation_2, scope,
                   initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
                   initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    with tf.variable_scope(scope):
        batch = tf.shape(user)[0]

        ones_1 = tf.ones([batch, 1, num_users, 2])
        ones_2 = tf.ones([batch, 1, bs_num_an, 2])
        ones_3 = tf.ones([batch, 1, bs_num_rf, 2])

        user_pro = FNN(tf.concat([user, ones_1], axis=-1), FNN_hidden_1, bn, training=training, scope='agg',
                       activation_h=hidden_activation_1, output_h=hidden_activation_1)

        user_edge = tf.concat([tf.transpose(tf.tile(user, [1, bs_num_an, 1, 1]), [0, 2, 1, 3]), H], axis=3)
        user_edge_pro = FNN(user_edge, FNN_hidden_1, bn, training=training, scope='agg',
                            activation_h=hidden_activation_1, output_h=hidden_activation_1)

        bs_an_pro = FNN(tf.concat([bs_an, ones_2], axis=-1), FNN_hidden_1, bn, training=training, scope='agg',
                        activation_h=hidden_activation_1, output_h=hidden_activation_1)

        bs_an_edge = tf.concat([tf.tile(bs_an, [1, num_users, 1, 1]), H], axis=3)
        bs_an_edge_pro = FNN(bs_an_edge, FNN_hidden_1, bn, training=training, scope='agg',
                             activation_h=hidden_activation_1, output_h=hidden_activation_1)

        bs_rf_pro_1 = FNN(tf.concat([bs_rf, ones_3], axis=-1), FNN_hidden_1, bn, training=training,
                          scope='agg', activation_h=hidden_activation_1, output_h=hidden_activation_1)
        bs_rf_pro_2 = FNN(tf.concat([bs_rf, ones_3], axis=-1), FNN_hidden_1, bn, training=training,
                          scope='agg', activation_h=hidden_activation_1, output_h=hidden_activation_1)

        # Update user
        pro_inf = tf.reduce_mean(bs_rf_pro_2, axis=2, keepdims=True) + \
                  tf.transpose(tf.reduce_mean(bs_an_edge_pro, axis=2, keepdims=True), [0, 2, 1, 3])
        user_new = FNN(tf.concat([user, pro_inf], axis=-1), FNN_hidden_2, bn, training=training,
                       scope='com', activation_h=hidden_activation_2, output_h=hidden_activation_2)

        # Update bs_an
        pro_inf = tf.reduce_mean(tf.tile(tf.transpose(bs_rf_pro_1, [0, 2, 1, 3]), [1, 1, bs_num_an, 1]),
                                 axis=1, keepdims=True) + \
                  tf.reduce_mean(user_edge_pro, axis=1, keepdims=True)
        bs_an_new = FNN(tf.concat([bs_an, pro_inf], axis=-1), FNN_hidden_2, bn, training=training,
                        scope='com', activation_h=hidden_activation_2, output_h=hidden_activation_2)

        # Update bs_rf
        pro_inf = tf.reduce_mean(user_pro, axis=2, keepdims=True) + \
                  tf.transpose(tf.reduce_mean(tf.tile(bs_an_pro, [1, bs_num_rf, 1, 1]), axis=2, keepdims=True),
                               [0, 2, 1, 3])
        bs_rf_new = FNN(tf.concat([bs_rf, pro_inf], axis=-1), FNN_hidden_2, bn, training=training,
                        scope='com', activation_h=hidden_activation_2, output_h=hidden_activation_2)

        return user_new, bs_rf_new, bs_an_new
