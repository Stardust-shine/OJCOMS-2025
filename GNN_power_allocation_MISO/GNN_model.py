import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


def gnn_layer(x, outshape, *adj_mat, transfer_function=tf.nn.sigmoid, pooling_function=tf.reduce_sum, name='0',
              is_transfer=True, is_BN=True):

    x_shape = np.shape(x).as_list()
    K = len(adj_mat)
    if K % 2 != 0:
        print('Error: Wrong number of arguments')
        raise NameError
    k = 0

    W_x0 = ini_weights(x_shape[2], outshape, 'W_x0'+name)
    H_x0 = tf.matmul(x, W_x0)

    while k < K:
        ak_mat = adj_mat[k]
        xk = adj_mat[k + 1]

        data_format_check(ak_mat, xk, x_shape, k)

        H_x_combine = tf.zeros([1, 1, outshape])
        H_a_combine = tf.zeros([1, x_shape[1], outshape])

        if ak_mat != '':
            ak_shape = np.shape(ak_mat).as_list()
            W_ak = ini_weights(ak_shape[3], outshape, 'W_a' + str(int(k / 2) + 1)+name)  # Cak = ak_shape[3]
            H_ak = tf.matmul(ak_mat, W_ak)
            H_a_combine = H_a_combine + pooling_function(H_ak, reduction_indices=[2])

        if xk != '':
            xk_shape = np.shape(xk).as_list()
            W_xk = ini_weights(xk_shape[2], outshape, 'W_x' + str(int(k / 2) + 1)+name)  # Cxk = xk_shape[2]
            H_xk = tf.matmul(xk, W_xk)
            H_x_combine = H_x_combine + pooling_function(H_xk, reduction_indices=[1], keepdims=True)

        H_x0 = H_x0 + H_a_combine + H_x_combine

        k = k + 2

    if is_transfer is True:
        H_x0 = transfer_function(H_x0)

    if is_BN is True:
        H_x0 = tf.layers.batch_normalization(H_x0, training=True)

    return H_x0


def gnn_pe_layer(x, outshape, *adj_mat, transfer_function=tf.nn.sigmoid, pooling_function=tf.reduce_sum,
                 name='0', is_transfer=True, is_BN=True):
    x_shape = np.shape(x).as_list()
    K = len(adj_mat)
    if K % 3 != 0:
        print('Error: Wrong number of arguments')
        raise NameError
    k = 0

    W_x0 = ini_weights(x_shape[2], outshape, 'W_x0'+name)
    H_x0 = tf.matmul(x, W_x0)

    while k < K:
        ak_mat = adj_mat[k]
        xk = adj_mat[k + 1]
        pk = adj_mat[k + 2]

        data_format_check(ak_mat, xk, x_shape, k)

        H_x_combine = tf.zeros([1, x_shape[1], outshape])  # N_s X N X outshape
        H_a_combine = tf.zeros([1, x_shape[1], outshape])  # N_s X N X outshape

        if ak_mat != '':
            H_ak = matmul_a_pe(ak_mat, pk, int(k / 3) + 1, outshape, name)
            H_a_combine = H_a_combine + pooling_function(H_ak, reduction_indices=[2])

        if xk != '':
            H_xk = matmul_x_pe(xk, pk, int(k / 3) + 1, outshape, name)
            H_x_combine = H_x_combine + pooling_function(H_xk, reduction_indices=[2])

        H_x0 = H_x0 + H_a_combine + H_x_combine

        k = k + 3

    if is_transfer is True:
        H_x0 = transfer_function(H_x0)

    if is_BN is True:
        H_x0 = tf.layers.batch_normalization(H_x0, training=True)

    return H_x0


def matmul_a_pe(ak_mat, pk, k, outshape, name):
    ak_shape = np.shape(ak_mat).as_list()
    pk_shape = np.shape(pk).as_list()
    N_s = ak_shape[0]; N = ak_shape[1]; Nk = ak_shape[2]; Cak = ak_shape[3]

    # format check
    if N != pk_shape[1] or Nk != pk_shape[2]:
        print('Error: uncapatible shape with ak_mat and pk')
        raise NameError

    W_d = ini_weights(Cak, outshape, 'W_a_d' + str(k) + name)      # 对角线上权重矩阵，Cak = ak_shape[3]
    W_nd = ini_weights(Cak, outshape, 'W_a_nd' + str(k) + name)    # 非对角线上权重矩阵，Cak = ak_shape[3]
    W_nnd = ini_weights(Cak, outshape, 'W_a_nnd' + str(k) + name)  # 非对角线上权重矩阵，Cak = ak_shape[3]

    H_ak_d = tf.matmul(ak_mat, W_d)
    H_ak_nd = tf.matmul(ak_mat, W_nd)
    H_ak_nnd = tf.matmul(ak_mat, W_nnd)

    pk = tf.cast(tf.reshape(pk, [-1, N, Nk, 1]), tf.float32)
    H_ak = H_ak_d * tf.to_float(tf.equal(pk, 2)) + H_ak_nd * tf.to_float(tf.equal(pk, 1)) + \
                                                   H_ak_nnd * tf.to_float(tf.equal(pk, 0))

    return H_ak


def matmul_x_pe(xk, pk, k, outshape, name):
    xk_shape = np.shape(xk).as_list()
    pk_shape = np.shape(pk).as_list()
    N_s = xk_shape[0]; Nk = xk_shape[1]; Cxk = xk_shape[2]; N = pk_shape[1]

    if Nk != pk_shape[2]:
        print('Error: uncapatible shape with xk and pk')
        raise NameError

    W_d = ini_weights(Cxk, outshape, 'W_x_d' + str(k) + name)        # 对角线上权重矩阵
    W_nd = ini_weights(Cxk, outshape, 'W_x_nd' + str(k) + name)      # 非对角线上权重矩阵
    W_nnd = ini_weights(Cxk, outshape, 'W_x_nnd' + str(k) + name)  # 非对角线上权重矩阵

    xk = tf.reshape(xk, [-1, 1, Nk, Cxk])
    H_xk_d = tf.matmul(xk, W_d)
    H_xk_nd = tf.matmul(xk, W_nd)
    H_xk_nnd = tf.matmul(xk, W_nnd)

    pk = tf.cast(tf.reshape(pk, [-1, N, Nk, 1]), tf.float32)
    H_xk = H_xk_d * tf.to_float(tf.equal(pk, 2)) + H_xk_nd * tf.to_float(tf.equal(pk, 1)) + \
                                                   H_xk_nnd * tf.to_float(tf.equal(pk, 0))

    return H_xk


def gnn_pe_layer_new(x, outshape, *adj_mat, transfer_function=tf.nn.sigmoid, transfer_function_fnn=tf.nn.sigmoid,
                     pooling_function=tf.reduce_mean, name='0', is_transfer=True, is_BN=True):
    x_shape = np.shape(x).as_list()
    K = len(adj_mat)
    if K % 3 != 0:
        print('Error: Wrong number of arguments')
        raise NameError
    k = 0

    W_x0 = ini_weights(x_shape[2], outshape, 'W_x0'+name)
    H_x0 = tf.matmul(x, W_x0)

    while k < K:
        ak_mat = adj_mat[k]
        xk = adj_mat[k + 1]
        pk = adj_mat[k + 2]

        data_format_check(ak_mat, xk, x_shape, k)

        H_x_combine = tf.zeros([1, x_shape[1], outshape])  # N_s X N X outshape
        H_a_combine = tf.zeros([1, x_shape[1], outshape])  # N_s X N X outshape

        if ak_mat != '':
            H_ak = matmul_a_pe_new(ak_mat, pk, int(k / 3) + 1, outshape, name,
                                   is_transfer, transfer_function=transfer_function_fnn)
            H_a_combine = H_a_combine + pooling_function(H_ak, reduction_indices=[2])

        if xk != '':
            H_xk = matmul_x_pe_new(xk, pk, int(k / 3) + 1, outshape, name, is_transfer,
                                   transfer_function=transfer_function_fnn)
            H_x_combine = H_x_combine + pooling_function(H_xk, reduction_indices=[2])

        H_x0 = H_x0 + H_a_combine + H_x_combine

        k = k + 3

    if is_transfer is True:
        H_x0 = transfer_function(H_x0)

    if is_BN is True:
        H_x0 = tf.layers.batch_normalization(H_x0, training=True)

    return H_x0


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


def FNN(x, hidden_size, bn=False, scope='FNN', training=True, activation_h=tf.nn.softmax,
        initializer_w=tf.keras.initializers.VarianceScaling(scale=1 / (3 * 1), distribution='uniform'),
        initializer_b=tf.zeros_initializer()):
    for i, h in enumerate(hidden_size[:]):
        x = fc(x, h, scope=scope + str(i), initializer_w=initializer_w, initializer_b=initializer_b)
        if activation_h is not None:
            if bn:
                x = tf.layers.batch_normalization(x, training=training)
            x = activation_h(x)
    return x


def matmul_a_pe_new(ak_mat, pk, k, outshape, name, is_transfer, transfer_function=tf.nn.leaky_relu):
    ak_shape = np.shape(ak_mat).as_list()
    pk_shape = np.shape(pk).as_list()
    N_s = ak_shape[0]; N = ak_shape[1]; Nk = ak_shape[2]; Cak = ak_shape[3]

    # format check
    if N != pk_shape[1] or Nk != pk_shape[2]:
        print('Error: uncapatible shape with ak_mat and pk')
        raise NameError

    if is_transfer is False:
        transfer_function = None

    H_ak_d = FNN(ak_mat, [2*outshape, outshape], bn=True, scope='W_a_d' + str(k) + name, activation_h=transfer_function)
    H_ak_nd = FNN(ak_mat, [2*outshape, outshape], bn=True, scope='W_a_nd' + str(k) + name, activation_h=transfer_function)
    H_ak_nnd = FNN(ak_mat, [2*outshape, outshape], bn=True, scope='W_a_nnd' + str(k) + name, activation_h=transfer_function)

    pk = tf.cast(tf.reshape(pk, [-1, N, Nk, 1]), tf.float32)
    H_ak = H_ak_d * tf.to_float(tf.equal(pk, 2)) + H_ak_nd * tf.to_float(tf.equal(pk, 1)) + \
                                                   H_ak_nnd * tf.to_float(tf.equal(pk, 0))

    return H_ak


def matmul_x_pe_new(xk, pk, k, outshape, name, is_transfer, transfer_function=tf.nn.leaky_relu):
    xk_shape = np.shape(xk).as_list()
    pk_shape = np.shape(pk).as_list()
    N_s = xk_shape[0]; Nk = xk_shape[1]; Cxk = xk_shape[2]; N = pk_shape[1]

    if Nk != pk_shape[2]:
        print('Error: uncapatible shape with xk and pk')
        raise NameError

    if is_transfer is False:
        transfer_function = None

    xk = tf.reshape(xk, [-1, 1, Nk, Cxk])
    H_xk_d = FNN(xk, [2*outshape, outshape], bn=True, scope='W_x_d' + str(k) + name, activation_h=transfer_function)
    H_xk_nd = FNN(xk, [2*outshape, outshape], bn=True, scope='W_x_nd' + str(k) + name, activation_h=transfer_function)
    H_xk_nnd = FNN(xk, [2*outshape, outshape], bn=True, scope='W_x_nnd' + str(k) + name, activation_h=transfer_function)

    pk = tf.cast(tf.reshape(pk, [-1, N, Nk, 1]), tf.float32)
    H_xk = H_xk_d * tf.to_float(tf.equal(pk, 2)) + H_xk_nd * tf.to_float(tf.equal(pk, 1)) + \
                                                   H_xk_nnd * tf.to_float(tf.equal(pk, 0))

    return H_xk


def ini_weights(in_shape, out_shape, name, stddev=0.1):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights', reuse=tf.AUTO_REUSE):
            weight_mat = tf.get_variable(name='weight', shape=[in_shape, out_shape],
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=stddev))
    return weight_mat


def ini_bias(out_shape, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('biases', reuse=tf.AUTO_REUSE):
            bias_vec = tf.get_variable(name='bias', shape=[out_shape],
                                         initializer=tf.zeros_initializer())
    return bias_vec


def data_format_check(ak_mat, xk, x_shape, k):

    if ak_mat != '':
        ak_shape = np.shape(ak_mat).as_list()
        if len(ak_shape) != 4:
            print('Wrong with the dimensions in the ' + str(int(k / 2) + 1) + ' th adjedent matrix')
            raise NameError
        if ak_shape[0] != x_shape[0]:
            print('Wrong with the number of samples in the ' + str(int(k / 2) + 1) + ' th adjedent matrix')
            raise NameError
        if ak_shape[1] != x_shape[1]:
            print('Error: wrong with the shape of the ' + str(int(k / 2) + 1) + ' th adjedency matrix ')
            raise NameError

    if xk != '':
        xk_shape = np.shape(xk).as_list()
        if len(xk_shape) != 3:
            print('Wrong with the dimensions in the ' + str(int(k / 2) + 1) + ' th feature vector')
            raise NameError
        if xk_shape[0] != x_shape[0]:
            print('Wrong with the number of samples in the ' + str(int(k / 2) + 1) + ' th feature vector')
            raise NameError

    if ak_mat != '' and xk != '':
        ak_shape = np.shape(ak_mat).as_list()
        xk_shape = np.shape(xk).as_list()
        if ak_shape[2] != xk_shape[1]:
            print('Error: wrong with the shape of the ' + str(int(k / 2) + 1) + ' th feature vector ')
            raise NameError


def cal_unspv_cost(p, H, var_noise, access_UE_BS):
    # nsample = int(p.shape[0])
    nBS = int(p.shape[1])
    access_UE_BS = tf.to_float(tf.reshape(access_UE_BS, [1, int(access_UE_BS.shape[0]), int(access_UE_BS.shape[1]), 1]))
    HH = tf.square(H) * tf.reshape(p, [-1, nBS, 1, 1])
    SINR = tf.reduce_sum(HH * access_UE_BS, reduction_indices=1) / \
           (tf.reduce_sum(HH * (1-access_UE_BS), reduction_indices=1) + var_noise)
    cost = -1.0*tf.reduce_sum(tf.log(1.0 + SINR) / tf.log(2.0))

def standard_scale(X_train, X_test):
    mean_val = np.mean(X_train, axis=0, keepdims=True)
    std_val = np.std(X_train, axis=0, keepdims=True)
    X_train = (X_train - mean_val) / std_val
    X_test = (X_test - mean_val) / std_val
    return X_train, X_test


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


def f_gnn_layer(x, outshape, *adj_mat, transfer_function=tf.nn.sigmoid, pooling_function=tf.reduce_mean,
                name='0', is_transfer=True, is_BN=True):
    x_shape = np.shape(x).as_list()
    K = len(adj_mat)
    if K % 3 != 0:
        print('Error: Wrong number of arguments')
        raise NameError
    k = 0

    W_x0 = ini_weights(x_shape[2], outshape, 'W_x0'+name)
    H_x0 = tf.matmul(x, W_x0)

    while k < K:
        ak_mat = adj_mat[k]
        xk = adj_mat[k + 1]
        pk = adj_mat[k + 2]

        data_format_check(ak_mat, xk, x_shape, k)

        H_x_combine = tf.zeros([1, x_shape[1], outshape])  # N_s X N X outshape
        H_a_combine = tf.zeros([1, x_shape[1], outshape])  # N_s X N X outshape

        if ak_mat != '':
            H_ak = f_matmul_a_pe(ak_mat, pk, int(k / 3) + 1, outshape, name)
            H_a_combine = H_a_combine + pooling_function(H_ak, reduction_indices=[2])

        if xk != '':
            H_xk = f_matmul_x_pe(xk, pk, int(k / 3) + 1, outshape, name)
            H_x_combine = H_x_combine + pooling_function(H_xk, reduction_indices=[2])

        H_x0 = H_x0 + H_a_combine + H_x_combine

        k = k + 3

    if is_transfer is True:
        H_x0 = transfer_function(H_x0)

    if is_BN is True:
        H_x0 = tf.layers.batch_normalization(H_x0, training=True)

    return H_x0


def f_matmul_a_pe(ak_mat, pk, k, outshape, name):
    ak_shape = np.shape(ak_mat).as_list()
    pk_shape = np.shape(pk).as_list()
    N_s = tf.shape(ak_mat)[0]; N = ak_shape[1]; Nk = ak_shape[2]; Cak = ak_shape[3]

    # format check
    if N != pk_shape[1] or Nk != pk_shape[2]:
        print('Error: uncapatible shape with ak_mat and pk')
        raise NameError

    W_d = tf.tile(f_ini_weights([1, N, Nk, Cak, outshape], 'W_a_d' + str(k) + name),
                  [N_s, 1, 1, 1, 1])  # 对角线上权重矩阵，Cak = ak_shape[3]
    W_nd = tf.tile(f_ini_weights([1, N, Nk, Cak, outshape], 'W_a_nd' + str(k) + name),
                   [N_s, 1, 1, 1, 1])  # 非对角线上权重矩阵，Cak = ak_shape[3]
    W_nnd = tf.tile(f_ini_weights([1, N, Nk, Cak, outshape], 'W_a_nnd' + str(k) + name),
                    [N_s, 1, 1, 1, 1])  # 非对角线上权重矩阵，Cak = ak_shape[3]

    ak_mat = tf.expand_dims(ak_mat, axis=-2)
    H_ak_d = tf.matmul(ak_mat, W_d)
    H_ak_nd = tf.matmul(ak_mat, W_nd)
    H_ak_nnd = tf.matmul(ak_mat, W_nnd)

    H_ak_d = tf.reshape(H_ak_d, [N_s, N, Nk, outshape])
    H_ak_nd = tf.reshape(H_ak_nd, [N_s, N, Nk, outshape])
    H_ak_nnd = tf.reshape(H_ak_nnd, [N_s, N, Nk, outshape])

    pk = tf.cast(tf.reshape(pk, [-1, N, Nk, 1]), tf.float32)
    H_ak = H_ak_d * tf.to_float(tf.equal(pk, 2)) + H_ak_nd * tf.to_float(tf.equal(pk, 1)) + \
                                                   H_ak_nnd * tf.to_float(tf.equal(pk, 0))

    return H_ak


def f_matmul_x_pe(xk, pk, k, outshape, name):
    xk_shape = np.shape(xk).as_list()
    pk_shape = np.shape(pk).as_list()
    N_s = tf.shape(xk)[0]; Nk = xk_shape[1]; Cxk = xk_shape[2]; N = pk_shape[1]

    if Nk != pk_shape[2]:
        print('Error: uncapatible shape with xk and pk')
        raise NameError

    W_d = tf.tile(f_ini_weights([1, 1, Nk, Cxk, outshape], 'W_x_d' + str(k) + name), [N_s, 1, 1, 1, 1])  # 对角线上权重矩阵
    W_nd = tf.tile(f_ini_weights([1, 1, Nk, Cxk, outshape], 'W_x_nd' + str(k) + name), [N_s, 1, 1, 1, 1])  # 非对角线上权重矩阵
    W_nnd = tf.tile(f_ini_weights([1, 1, Nk, Cxk, outshape], 'W_x_nnd' + str(k) + name), [N_s, 1, 1, 1, 1])  # 非对角线上权重矩阵

    xk = tf.reshape(xk, [-1, 1, Nk, Cxk])
    xk = tf.expand_dims(xk, axis=-2)
    H_xk_d = tf.matmul(xk, W_d)
    H_xk_nd = tf.matmul(xk, W_nd)
    H_xk_nnd = tf.matmul(xk, W_nnd)

    H_xk_d = tf.reshape(H_xk_d, [N_s, 1, Nk, outshape])
    H_xk_nd = tf.reshape(H_xk_nd, [N_s, 1, Nk, outshape])
    H_xk_nnd = tf.reshape(H_xk_nnd, [N_s, 1, Nk, outshape])

    pk = tf.cast(tf.reshape(pk, [-1, N, Nk, 1]), tf.float32)
    H_xk = H_xk_d * tf.to_float(tf.equal(pk, 2)) + H_xk_nd * tf.to_float(tf.equal(pk, 1)) + \
                                                   H_xk_nnd * tf.to_float(tf.equal(pk, 0))

    return H_xk


def f_ini_weights(shape, name, stddev=0.1):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights', reuse=tf.AUTO_REUSE):
            weight_mat = tf.get_variable(name='weight', shape=shape,
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=stddev))
    return weight_mat


def f_ini_bias(shape, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('biases', reuse=tf.AUTO_REUSE):
            bias_vec = tf.get_variable(name='bias', shape=shape,
                                         initializer=tf.zeros_initializer())
    return bias_vec


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


def gnn_pe_layer_two_edge_type(x, outshape, *adj_mat, transfer_function=tf.nn.sigmoid,
                               transfer_function_fnn=tf.nn.sigmoid, pooling_function=tf.reduce_mean, name='0',
                               is_transfer=True, is_BN=True):
    x_shape = np.shape(x).as_list()
    K = len(adj_mat)
    if K % 3 != 0:
        print('Error: Wrong number of arguments')
        raise NameError
    k = 0

    W_x0 = ini_weights(x_shape[2], outshape, 'W_x0'+name)
    H_x0 = tf.matmul(x, W_x0)

    while k < K:
        ak_mat = adj_mat[k]
        xk = adj_mat[k + 1]
        pk = adj_mat[k + 2]

        data_format_check(ak_mat, xk, x_shape, k)

        H_x_combine = tf.zeros([1, x_shape[1], outshape])  # N_s X N X outshape
        H_a_combine = tf.zeros([1, x_shape[1], outshape])  # N_s X N X outshape

        if ak_mat != '':
            H_ak = matmul_a_pe_two_edge_type(ak_mat, pk, int(k / 3) + 1, outshape, name,
                                             is_transfer, transfer_function=transfer_function_fnn)
            H_a_combine = H_a_combine + pooling_function(H_ak, reduction_indices=[2])

        if xk != '':
            H_xk = matmul_x_pe_two_edge_type(xk, pk, int(k / 3) + 1, outshape, name, is_transfer,
                                             transfer_function=transfer_function_fnn)
            H_x_combine = H_x_combine + pooling_function(H_xk, reduction_indices=[2])

        H_x0 = H_x0 + H_a_combine + H_x_combine

        k = k + 3

    if is_transfer is True:
        H_x0 = transfer_function(H_x0)

    if is_BN is True:
        H_x0 = tf.layers.batch_normalization(H_x0, training=True)

    return H_x0


def matmul_a_pe_two_edge_type(ak_mat, pk, k, outshape, name, is_transfer, transfer_function=tf.nn.leaky_relu):
    ak_shape = np.shape(ak_mat).as_list()
    pk_shape = np.shape(pk).as_list()
    N_s = ak_shape[0]; N = ak_shape[1]; Nk = ak_shape[2]; Cak = ak_shape[3]

    # format check
    if N != pk_shape[1] or Nk != pk_shape[2]:
        print('Error: uncapatible shape with ak_mat and pk')
        raise NameError

    if is_transfer is False:
        transfer_function = None

    H_ak_d = FNN(ak_mat, [2*outshape, outshape], bn=True, scope='W_a_d' + str(k) + name, activation_h=transfer_function)
    H_ak_nd = FNN(ak_mat, [2*outshape, outshape], bn=True, scope='W_a_nd' + str(k) + name, activation_h=transfer_function)
    H_ak_nnd = FNN(ak_mat, [2*outshape, outshape], bn=True, scope='W_a_nd' + str(k) + name, activation_h=transfer_function)

    pk = tf.cast(tf.reshape(pk, [-1, N, Nk, 1]), tf.float32)
    H_ak = H_ak_d * tf.to_float(tf.equal(pk, 2)) + H_ak_nd * tf.to_float(tf.equal(pk, 1)) + \
                                                   H_ak_nnd * tf.to_float(tf.equal(pk, 0))

    return H_ak


def matmul_x_pe_two_edge_type(xk, pk, k, outshape, name, is_transfer, transfer_function=tf.nn.leaky_relu):
    xk_shape = np.shape(xk).as_list()
    pk_shape = np.shape(pk).as_list()
    N_s = xk_shape[0]; Nk = xk_shape[1]; Cxk = xk_shape[2]; N = pk_shape[1]

    if Nk != pk_shape[2]:
        print('Error: uncapatible shape with xk and pk')
        raise NameError

    if is_transfer is False:
        transfer_function = None

    xk = tf.reshape(xk, [-1, 1, Nk, Cxk])
    H_xk_d = FNN(xk, [2*outshape, outshape], bn=True, scope='W_x_d' + str(k) + name, activation_h=transfer_function)
    H_xk_nd = FNN(xk, [2*outshape, outshape], bn=True, scope='W_x_nd' + str(k) + name, activation_h=transfer_function)
    H_xk_nnd = FNN(xk, [2*outshape, outshape], bn=True, scope='W_x_nd' + str(k) + name, activation_h=transfer_function)

    pk = tf.cast(tf.reshape(pk, [-1, N, Nk, 1]), tf.float32)
    H_xk = H_xk_d * tf.to_float(tf.equal(pk, 2)) + H_xk_nd * tf.to_float(tf.equal(pk, 1)) + \
                                                   H_xk_nnd * tf.to_float(tf.equal(pk, 0))

    return H_xk