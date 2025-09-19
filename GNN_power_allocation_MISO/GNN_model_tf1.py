import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


def fc(x, n_output, bn=False, training=False, scope="fc", activation_fn=None,
       initializer_w=tf.keras.initializers.VarianceScaling(mode="fan_in"),
       initializer_b=tf.constant_initializer(.0, dtype=tf.float32)):
    batch = tf.shape(x)[0]
    N1 = x.get_shape()[1]
    D = x.get_shape()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", shape=[x.get_shape()[-1], n_output], initializer=initializer_w)
        # b = tf.get_variable("b", shape=[n_output], initializer=initializer_b)
        x = tf.reshape(x, [batch * N1, D])
        # fc = tf.add(tf.matmul(x, W), b)
        fc = tf.matmul(x, W)
        fc = tf.reshape(fc, [batch, N1, n_output])
        # fc = tf.add(tf.tensordot(x, W, axes=[[-1], [0]]), b)
        if activation_fn == '(tanh+1)*0.5':
            if bn is True:
                fc = tf.layers.batch_normalization(fc, training=training, momentum=0.99, epsilon=0.00001)
            fc = 0.5 * (tf.tanh(fc) + 1)
        elif activation_fn is not None:
            if bn is True:
                fc = tf.layers.batch_normalization(fc, training=training, momentum=0.99, epsilon=0.00001)
            fc = activation_fn(fc)
    return fc


def FNN(x, hidden_size, bn=False, scope='FNN', training=True, activation_h=tf.nn.softmax,
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


def get_random_block_from_data(A, B, C, D, E, F, G, batch_size):
    start_index = np.random.randint(0, len(A) - batch_size)
    return A[start_index: start_index + batch_size], \
           B[start_index: start_index + batch_size], \
           C[start_index: start_index + batch_size], \
           D[start_index: start_index + batch_size], \
           E[start_index: start_index + batch_size], \
           F[start_index: start_index + batch_size], \
           G[start_index: start_index + batch_size]


def get_random_block_from_data1(A, B, C, D, batch_size):
    start_index = np.random.randint(0, len(A) - batch_size)
    return A[start_index: start_index + batch_size], \
           B[start_index: start_index + batch_size], \
           C[start_index: start_index + batch_size], \
           D[start_index: start_index + batch_size],


def get_random_block_from_data2(A, B, C, D, E, batch_size):
    start_index = np.random.randint(0, len(A) - batch_size)
    return A[start_index: start_index + batch_size], \
           B[start_index: start_index + batch_size], \
           C[start_index: start_index + batch_size], \
           D[start_index: start_index + batch_size], \
           E[start_index: start_index + batch_size]


def get_random_block_from_data3(A, B, C, batch_size):
    start_index = np.random.randint(0, len(A) - batch_size)
    return A[start_index: start_index + batch_size], \
           B[start_index: start_index + batch_size], \
           C[start_index: start_index + batch_size]


def CNN(A, stride, filters, CNN_dim, activation_h, scope, bn,
        initializer_w=tf.keras.initializers.VarianceScaling(mode="fan_in")):
    batch_size = tf.shape(A)[0]
    K = A.get_shape()[1]
    D = A.get_shape()[-1]

    A_reshaped = tf.reshape(A, [batch_size*K*K, D, 1])

    conv_out = A_reshaped

    for l, h in enumerate(CNN_dim):
        with tf.variable_scope(scope + str(l), reuse=tf.AUTO_REUSE):
            if l == 0:
                W = tf.get_variable('W' + str(l), shape=[filters[l], 1, h], initializer=initializer_w)
            else:
                W = tf.get_variable('W' + str(l), shape=[filters[l], CNN_dim[l-1], h], initializer=initializer_w)
            conv_out = tf.nn.conv1d(conv_out, W, stride=stride[l], padding='VALID', dilations=1)
            if activation_h is not None:
                if bn is True:
                    conv_out = tf.layers.batch_normalization(conv_out, training=True)
                conv_out = activation_h(conv_out)

    output = tf.reshape(conv_out, [batch_size, K, K, D * CNN_dim[-1]])

    return output


def homo_gnn_layer_CNN(h_bs, h_ue, H, stride, filters, CNN_hidden, FNN_hidden, process_cnn_ac=tf.nn.sigmoid,
                       combine_fnn_ac=tf.nn.softmax, pooling_function=tf.reduce_sum, name='0',
                       is_transfer=True, is_BN=True):
    K = np.shape(h_bs).as_list()[1]
    h_bs_high = tf.tile(tf.expand_dims(h_bs, axis=2), [1, 1, K, 1])
    h_bs_High_H = tf.concat([h_bs_high, H], axis=-1)

    h_ue_high = tf.tile(tf.expand_dims(h_ue, axis=2), [1, 1, K, 1])
    h_ue_High_H = tf.concat([h_ue_high, tf.transpose(H, [0, 2, 1, 3])], axis=-1)

    if is_transfer is False:
        process_cnn_ac = None
        combine_fnn_ac = None

    pro_h_bs = CNN(h_bs_High_H, stride, filters, CNN_hidden, bn=True, scope='CNN' + name, activation_h=process_cnn_ac)
    agg_bs = pooling_function(pro_h_bs, axis=2)  # [B, K, d]
    h_bs_new = FNN(tf.concat([h_bs, agg_bs], axis=-1), FNN_hidden, bn=True, scope='FNN' + name,
                   activation_h=combine_fnn_ac)

    pro_h_ue = CNN(h_ue_High_H, stride, filters, CNN_hidden, bn=True, scope='CNN' + name, activation_h=process_cnn_ac)
    agg_ue = pooling_function(pro_h_ue, axis=2)  # [B, K, d]
    h_ue_new = FNN(tf.concat([h_ue, agg_ue], axis=-1), FNN_hidden, bn=True, scope='FNN' + name,
                   activation_h=combine_fnn_ac)

    if is_BN is True:
        h_bs_new = tf.layers.batch_normalization(h_bs_new, training=True)
        h_ue_new = tf.layers.batch_normalization(h_ue_new, training=True)

    return h_bs_new, h_ue_new, pro_h_bs, pro_h_ue


def homo_gnn_layer(h_bs, h_ue, H, FNN1_hidden, FNN2_hidden, process_fnn_ac=tf.nn.sigmoid,
                   combine_fnn_ac=tf.nn.sigmoid, pooling_function=tf.reduce_sum, name='0', is_transfer=True, is_BN=True):
    K = np.shape(h_bs).as_list()[1]
    h_bs_high = tf.tile(tf.expand_dims(h_bs, axis=2), [1, 1, K, 1])
    h_bs_High_H = tf.concat([h_bs_high, H], axis=-1)

    h_ue_high = tf.tile(tf.expand_dims(h_ue, axis=2), [1, 1, K, 1])
    h_ue_High_H = tf.concat([h_ue_high, tf.transpose(H, [0, 2, 1, 3])], axis=-1)

    if is_transfer is False:
        process_fnn_ac = None
        combine_fnn_ac = None

    pro_h_bs = FNN(h_bs_High_H, FNN1_hidden, bn=True, scope='FNN1' + name, activation_h=process_fnn_ac)
    agg_bs = pooling_function(pro_h_bs, axis=2)  # [B, K, d]
    h_bs_new = FNN(tf.concat([h_bs, agg_bs], axis=-1), FNN2_hidden, bn=True, scope='FNN2' + name,
                   activation_h=combine_fnn_ac)

    pro_h_ue = FNN(h_ue_High_H, FNN1_hidden, bn=True, scope='FNN1' + name, activation_h=process_fnn_ac)
    agg_ue = pooling_function(pro_h_ue, axis=2)  # [B, K, d]
    h_ue_new = FNN(tf.concat([h_ue, agg_ue], axis=-1), FNN2_hidden, bn=True, scope='FNN2' + name,
                   activation_h=combine_fnn_ac)

    if is_BN is True:
        h_bs_new = tf.layers.batch_normalization(h_bs_new, training=True)
        h_ue_new = tf.layers.batch_normalization(h_ue_new, training=True)

    return h_bs_new, h_ue_new


