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


def add_1d_pe_layer(input, inshape, outshape, name, stddev=0.1, transfer_function=tf.nn.relu):
    U = ini_weights(inshape, outshape, name, stddev)
    V = ini_weights(inshape, outshape, name, stddev)
    b = ini_bias(outshape, name)
    input_sum = tf.reduce_sum(input, axis=1)
    output = transfer_function(tf.matmul(input, U - V) + tf.expand_dims(tf.matmul(input_sum, V), axis=1) + b)
    return output


def penn_1d(input, n_obj, layernum, output_activation=tf.nn.sigmoid):
    hidden = dict()
    input_exdim = tf.reshape(input, [-1, n_obj, int(input.shape[1] // n_obj)])
    for i in range(len(layernum)):
        if i == 0:
            hidden[str(i)] = add_1d_pe_layer(input_exdim, int(input_exdim.shape[2]), layernum[i], 'layer' + str(i + 1))
        elif i != len(layernum) - 1:
            hidden[str(i)] = add_1d_pe_layer(hidden[str(i - 1)], layernum[i - 1], layernum[i], 'layer' + str(i + 1))
        else:
            output_exdim = add_1d_pe_layer(hidden[str(i - 1)], layernum[i - 1], layernum[i], 'layer' + str(i + 1),
                                           transfer_function=output_activation)
    output = tf.reshape(output_exdim, [-1, layernum[-1] * n_obj])
    return output


def add_2d_pe_layer(input, inshape, outshape, name, stddev=0.1, transfer_function=tf.nn.relu):
    U = ini_weights(inshape, outshape, name, stddev)
    V = ini_weights(inshape, outshape, name, stddev)
    P = ini_weights(inshape, outshape, name, stddev)
    Q = ini_weights(inshape, outshape, name, stddev)
    b = ini_bias(outshape, name)
    input_sum_dim1 = tf.reduce_sum(input, axis=1)
    input_sum_dim2 = tf.reduce_sum(input, axis=2)
    input_sum_dim12 = tf.reduce_sum(input_sum_dim1, axis=1)
    output = transfer_function(tf.matmul(input, U - V - P - Q) +
                               tf.expand_dims(tf.matmul(input_sum_dim1, V - Q), axis=1) +
                               tf.expand_dims(tf.matmul(input_sum_dim2, P - Q), axis=2) +
                               tf.expand_dims(tf.expand_dims(tf.matmul(input_sum_dim12, Q), axis=1), axis=2) + b)
    return output


def penn_2d(input, n_obj1, n_obj2, layernum, hidden_activation=tf.nn.sigmoid, output_activation=tf.nn.sigmoid):
    hidden = dict()
    input_exdim = tf.reshape(input, [-1, n_obj1, n_obj2, int(input.shape[1] // n_obj1 // n_obj2)])
    for i in range(len(layernum)):
        if i == 0:
            hidden[str(i)] = add_2d_pe_layer(input_exdim, int(input_exdim.shape[3]), layernum[i], 'layer' + str(i + 1),
                                             transfer_function=hidden_activation)
        elif i != len(layernum) - 1:
            hidden[str(i)] = add_2d_pe_layer(hidden[str(i - 1)], layernum[i - 1], layernum[i], 'layer' + str(i + 1),
                                             transfer_function=hidden_activation)
        else:
            output_exdim = add_2d_pe_layer(hidden[str(i - 1)], layernum[i - 1], layernum[i], 'layer' + str(i + 1),
                                           transfer_function=output_activation)
    output = tf.reshape(output_exdim, [-1, layernum[-1] * n_obj1 * n_obj2])
    return output


# def penn_joint(input, n_obj, layernum, output_activation=tf.nn.sigmoid):
#     hidden = dict()
#     input_exdim = tf.reshape(input, [-1, n_obj, n_obj, int(input.shape[1]//n_obj//n_obj)])
#     for i in range(len(layernum)):
#         if i == 0:
#             hidden[str(i)] = add_2d_pe_layer(input_exdim, int(input_exdim.shape[3]), layernum[i], 'layer' + str(i + 1))
#         elif i != len(layernum)-1:
#             hidden[str(i)] = add_2d_pe_layer(hidden[str(i - 1)], layernum[i-1], layernum[i], 'layer' + str(i + 1))
#         else:
#             output_exdim = add_2d_pe_layer(hidden[str(i - 1)], layernum[i-1], layernum[i], 'layer' + str(i + 1),
#                                            transfer_function=output_activation)
#     output = tf.reshape(tf.reduce_sum(tf.expand_dims(tf.expand_dims(tf.eye(n_obj), axis=0), axis=3) * output_exdim,
#                         axis=(1, 2)), [-1, n_obj*layernum[-1]])
#     return output

def gen_index(num_obj):
    INDEX_W = []
    INDEX_B = []
    for i in range(num_obj):
        for j in range(num_obj):
            index = []
            if i == j:
                INDEX_B.append(0)
                for ix in range(num_obj):
                    for jx in range(num_obj):
                        if ix == i:
                            if jx == i:
                                index.append(0.)
                            else:
                                index.append(1.)
                        else:
                            if jx == i:
                                index.append(2.)
                            elif jx == ix:
                                index.append(3.)
                            else:
                                index.append(4.)
            else:
                INDEX_B.append(1)
                for ix in range(num_obj):
                    for jx in range(num_obj):
                        if ix == j:
                            if jx == j:
                                index.append(5.)
                            elif jx == i:
                                index.append(6.)
                            else:
                                index.append(7.)
                        elif ix == i:
                            if jx == j:
                                index.append(8.)
                            elif jx == i:
                                index.append(9.)
                            else:
                                index.append(10.)
                        else:
                            if jx == ix:
                                index.append(11.)
                            elif jx == i:
                                index.append(12.)
                            else:
                                index.append(13.)
            INDEX_W.append(index)
    index_w = np.array(INDEX_W).astype(np.float32)
    index_b = np.array(INDEX_B).reshape(num_obj * num_obj, 1)
    index_b = index_b.astype(np.float32)
    return index_w, index_b


def gen_weight(num_pair, inshape, outshape, index_w, coefficient, name):
    w1 = ini_weights(inshape, outshape, name + 'w1', stddev=0.1)
    w2 = ini_weights(inshape, outshape, name + 'w2', stddev=0.1) * coefficient
    w5 = ini_weights(inshape, outshape, name + 'w5', stddev=0.1) * coefficient
    w6 = ini_weights(inshape, outshape, name + 'w6', stddev=0.1) * coefficient
    w7 = ini_weights(inshape, outshape, name + 'w7', stddev=0.1) * coefficient

    y1 = ini_weights(inshape, outshape, name + 'y1', stddev=0.1) * coefficient
    y2 = ini_weights(inshape, outshape, name + 'y2', stddev=0.1) * coefficient
    y3 = ini_weights(inshape, outshape, name + 'y3', stddev=0.1) * coefficient
    y5 = ini_weights(inshape, outshape, name + 'y5', stddev=0.1)
    y6 = ini_weights(inshape, outshape, name + 'y6', stddev=0.1) * coefficient
    y7 = ini_weights(inshape, outshape, name + 'y7', stddev=0.1) * coefficient
    y9 = ini_weights(inshape, outshape, name + 'y9', stddev=0.1) * coefficient
    y10 = ini_weights(inshape, outshape, name + 'y10', stddev=0.1) * coefficient
    y11 = ini_weights(inshape, outshape, name + 'y11', stddev=0.1) * coefficient

    W = tf.expand_dims(tf.expand_dims(index_w, axis=-1), axis=-1)
    W = tf.tile(W, [1, 1, inshape, outshape])
    W = tf.multiply(tf.to_float(tf.equal(W, 0. * tf.ones(tf.shape(W)))), w1) + \
        tf.multiply(tf.to_float(tf.equal(W, 1. * tf.ones(tf.shape(W)))), w2) + \
        tf.multiply(tf.to_float(tf.equal(W, 2. * tf.ones(tf.shape(W)))), w5) + \
        tf.multiply(tf.to_float(tf.equal(W, 3. * tf.ones(tf.shape(W)))), w6) + \
        tf.multiply(tf.to_float(tf.equal(W, 4. * tf.ones(tf.shape(W)))), w7) + \
        tf.multiply(tf.to_float(tf.equal(W, 5. * tf.ones(tf.shape(W)))), y1) + \
        tf.multiply(tf.to_float(tf.equal(W, 6. * tf.ones(tf.shape(W)))), y2) + \
        tf.multiply(tf.to_float(tf.equal(W, 7. * tf.ones(tf.shape(W)))), y3) + \
        tf.multiply(tf.to_float(tf.equal(W, 8. * tf.ones(tf.shape(W)))), y5) + \
        tf.multiply(tf.to_float(tf.equal(W, 9. * tf.ones(tf.shape(W)))), y6) + \
        tf.multiply(tf.to_float(tf.equal(W, 10. * tf.ones(tf.shape(W)))), y7) + \
        tf.multiply(tf.to_float(tf.equal(W, 11. * tf.ones(tf.shape(W)))), y9) + \
        tf.multiply(tf.to_float(tf.equal(W, 12. * tf.ones(tf.shape(W)))), y10) + \
        tf.multiply(tf.to_float(tf.equal(W, 13. * tf.ones(tf.shape(W)))), y11)
    W = tf.transpose(W, [0, 2, 1, 3])
    W = tf.reshape(W, shape=(num_pair * num_pair * inshape, num_pair * num_pair * outshape))

    return W


def gen_bias(num_pair, outshape, index_b, coefficient, name):
    b1 = tf.reshape(ini_bias(outshape, name + 'b1'), [1, outshape]) * coefficient
    b2 = tf.reshape(ini_bias(outshape, name + 'b2'), [1, outshape]) * coefficient

    B = tf.multiply(tf.to_float(tf.equal(index_b, 0. * tf.ones(tf.shape(index_b)))), b1) + \
        tf.multiply(tf.to_float(tf.equal(index_b, 1. * tf.ones(tf.shape(index_b)))), b2)
    B = tf.reshape(B, [1, num_pair * num_pair * outshape])
    # B = tf.squeeze(B)

    return B


def add_joint_pe_layer(num_obj_1, num_obj_2, input, inshape, outshape, index_w_1, index_b_1, index_w_2, index_b_2,
                       coefficient, name, keep_prob, transfer_function=tf.nn.sigmoid,
                       is_BN=True, is_transfer=True, is_test=True):
    W1 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=coefficient, name=name + 'W1')
    W2 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'W2') * coefficient
    W5 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'W5') * coefficient
    W6 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'W6') * coefficient
    W7 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'W7') * coefficient

    Y1 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'Y1') * coefficient
    Y2 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'Y2') * coefficient
    Y3 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'Y3') * coefficient
    Y5 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=coefficient, name=name + 'Y5')
    Y6 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'Y6') * coefficient
    Y7 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'Y7') * coefficient
    Y9 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'Y9') * coefficient
    Y10 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'Y10') * coefficient
    Y11 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'Y11') * coefficient

    B1 = gen_bias(num_obj_2, outshape, index_b_2, coefficient=coefficient, name=name + 'B1')
    B2 = gen_bias(num_obj_2, outshape, index_b_2, coefficient=coefficient, name=name + 'B2')

    W = tf.expand_dims(tf.expand_dims(index_w_1, axis=-1), axis=-1)
    W = tf.tile(W, [1, 1, num_obj_2 * num_obj_2 * inshape, num_obj_2 * num_obj_2 * outshape])
    W = tf.multiply(tf.to_float(tf.equal(W, 0. * tf.ones(tf.shape(W)))), W1) + \
        tf.multiply(tf.to_float(tf.equal(W, 1. * tf.ones(tf.shape(W)))), W2) + \
        tf.multiply(tf.to_float(tf.equal(W, 2. * tf.ones(tf.shape(W)))), W5) + \
        tf.multiply(tf.to_float(tf.equal(W, 3. * tf.ones(tf.shape(W)))), W6) + \
        tf.multiply(tf.to_float(tf.equal(W, 4. * tf.ones(tf.shape(W)))), W7) + \
        tf.multiply(tf.to_float(tf.equal(W, 5. * tf.ones(tf.shape(W)))), Y1) + \
        tf.multiply(tf.to_float(tf.equal(W, 6. * tf.ones(tf.shape(W)))), Y2) + \
        tf.multiply(tf.to_float(tf.equal(W, 7. * tf.ones(tf.shape(W)))), Y3) + \
        tf.multiply(tf.to_float(tf.equal(W, 8. * tf.ones(tf.shape(W)))), Y5) + \
        tf.multiply(tf.to_float(tf.equal(W, 9. * tf.ones(tf.shape(W)))), Y6) + \
        tf.multiply(tf.to_float(tf.equal(W, 10. * tf.ones(tf.shape(W)))), Y7) + \
        tf.multiply(tf.to_float(tf.equal(W, 11. * tf.ones(tf.shape(W)))), Y9) + \
        tf.multiply(tf.to_float(tf.equal(W, 12. * tf.ones(tf.shape(W)))), Y10) + \
        tf.multiply(tf.to_float(tf.equal(W, 13. * tf.ones(tf.shape(W)))), Y11)
    W = tf.transpose(W, [0, 2, 1, 3])
    W = tf.reshape(W, shape=(num_obj_1 * num_obj_1 * num_obj_2 * num_obj_2 * inshape,
                             num_obj_1 * num_obj_1 * num_obj_2 * num_obj_2 * outshape))

    B = tf.multiply(tf.to_float(tf.equal(index_b_1, 0. * tf.ones(tf.shape(index_b_1)))), B1) + \
        tf.multiply(tf.to_float(tf.equal(index_b_1, 1. * tf.ones(tf.shape(index_b_1)))), B2)
    B = tf.reshape(B, [num_obj_1 * num_obj_1 * num_obj_2 * num_obj_2 * outshape, 1])
    B = tf.squeeze(B)

    C = tf.matmul(input, W)
    output = C
    batch = tf.shape(output)[0]
    output = tf.reshape(output, [batch, num_obj_1 * num_obj_1, num_obj_2 * num_obj_2 * outshape])
    output = tf.reshape(output, [batch, num_obj_1 * num_obj_1, num_obj_2 * num_obj_2, outshape])

    if is_transfer is True:
        output = transfer_function(output)
    output = tf.nn.dropout(output, keep_prob=keep_prob)
    if is_BN is True:
        if is_test is False:
            output = tf.layers.batch_normalization(output, training=True)
        else:
            output = tf.layers.batch_normalization(output, training=False)
    output = tf.reshape(output, [batch, num_obj_1 * num_obj_1 * num_obj_2 * num_obj_2 * outshape])

    return output


def add_joint_pe_layer_new(num_obj_1, num_obj_2, input, inshape, outshape, index_w_1, index_b_1, index_w_2, index_b_2,
                           coefficient, name, keep_prob, transfer_function=tf.nn.sigmoid,
                           is_BN=True, is_transfer=True, is_test=True, agg='sum'):
    input = tf.tile(input, [1, 1, num_obj_1 * num_obj_1 * num_obj_2 * num_obj_2, 1, 1])

    W1 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=coefficient, name=name + 'W1')
    W2 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'W2') * coefficient
    W5 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'W5') * coefficient
    W6 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'W6') * coefficient
    W7 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'W7') * coefficient

    Y1 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'Y1') * coefficient
    Y2 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'Y2') * coefficient
    Y3 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'Y3') * coefficient
    Y5 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=coefficient, name=name + 'Y5')
    Y6 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'Y6') * coefficient
    Y7 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'Y7') * coefficient
    Y9 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'Y9') * coefficient
    Y10 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'Y10') * coefficient
    Y11 = gen_weight(num_obj_2, inshape, outshape, index_w_2, coefficient=1.0, name=name + 'Y11') * coefficient

    B1 = gen_bias(num_obj_2, outshape, index_b_2, coefficient=coefficient, name=name + 'B1')
    B2 = gen_bias(num_obj_2, outshape, index_b_2, coefficient=coefficient, name=name + 'B2')

    W = tf.expand_dims(tf.expand_dims(index_w_1, axis=-1), axis=-1)
    W = tf.tile(W, [1, 1, num_obj_2 * num_obj_2 * inshape, num_obj_2 * num_obj_2 * outshape])
    W = tf.multiply(tf.to_float(tf.equal(W, 0. * tf.ones(tf.shape(W)))), W1) + \
        tf.multiply(tf.to_float(tf.equal(W, 1. * tf.ones(tf.shape(W)))), W2) + \
        tf.multiply(tf.to_float(tf.equal(W, 2. * tf.ones(tf.shape(W)))), W5) + \
        tf.multiply(tf.to_float(tf.equal(W, 3. * tf.ones(tf.shape(W)))), W6) + \
        tf.multiply(tf.to_float(tf.equal(W, 4. * tf.ones(tf.shape(W)))), W7) + \
        tf.multiply(tf.to_float(tf.equal(W, 5. * tf.ones(tf.shape(W)))), Y1) + \
        tf.multiply(tf.to_float(tf.equal(W, 6. * tf.ones(tf.shape(W)))), Y2) + \
        tf.multiply(tf.to_float(tf.equal(W, 7. * tf.ones(tf.shape(W)))), Y3) + \
        tf.multiply(tf.to_float(tf.equal(W, 8. * tf.ones(tf.shape(W)))), Y5) + \
        tf.multiply(tf.to_float(tf.equal(W, 9. * tf.ones(tf.shape(W)))), Y6) + \
        tf.multiply(tf.to_float(tf.equal(W, 10. * tf.ones(tf.shape(W)))), Y7) + \
        tf.multiply(tf.to_float(tf.equal(W, 11. * tf.ones(tf.shape(W)))), Y9) + \
        tf.multiply(tf.to_float(tf.equal(W, 12. * tf.ones(tf.shape(W)))), Y10) + \
        tf.multiply(tf.to_float(tf.equal(W, 13. * tf.ones(tf.shape(W)))), Y11)
    W = tf.reshape(W, shape=(num_obj_1 * num_obj_1 * num_obj_2 * num_obj_2, inshape,
                             num_obj_1 * num_obj_1 * num_obj_2 * num_obj_2, outshape))
    W = tf.transpose(W, [0, 2, 1, 3])

    C = tf.linalg.matmul(input, W)
    output = C

    if agg == 'sum':
        output = tf.reduce_sum(output, axis=2, keepdims=True)
    elif agg == 'max':
        output = tf.reduce_max(output, axis=2, keepdims=True)

    if is_transfer is True:
        output = transfer_function(output)
    output = tf.nn.dropout(output, keep_prob=keep_prob)
    if is_BN is True:
        if is_test is False:
            output = tf.layers.batch_normalization(output, training=True)
        else:
            output = tf.layers.batch_normalization(output, training=False)

    return output


def penn(num_obj_1, num_obj_2, input, layernum, index_w_1, index_b_1, index_w_2, index_b_2, coefficient,
         hidden_transfer=tf.nn.sigmoid, output_transfer=tf.nn.sigmoid, keep_prob=1.0, name='0', is_transfer=True,
         is_BN=True, is_test=False):
    input_shape = np.shape(input).as_list()
    hidden = dict()
    num_object = num_obj_1 * num_obj_2
    if len(layernum) == 1:
        output = add_joint_pe_layer(num_obj_1, num_obj_2, input,
                                    int(input_shape[len(input_shape) - 1] / num_object / num_object),
                                    layernum[0], index_w_1, index_b_1, index_w_2, index_b_2, coefficient,
                                    name + 'layer' + str(0 + 1), keep_prob, is_BN=is_BN,
                                    is_transfer=is_transfer, transfer_function=hidden_transfer, is_test=True)
    else:
        for i in range(len(layernum)):  # neural network layer
            if i == 0:
                hidden[str(i)] = add_joint_pe_layer(num_obj_1, num_obj_2, input,
                                                    int(input_shape[len(input_shape) - 1] / num_object / num_object),
                                                    layernum[i], index_w_1, index_b_1, index_w_2, index_b_2,
                                                    coefficient,
                                                    name + 'layer' + str(i + 1), keep_prob, is_BN=is_BN,
                                                    is_transfer=is_transfer, transfer_function=hidden_transfer,
                                                    is_test=True)
            elif i != len(layernum) - 1:
                hidden[str(i)] = add_joint_pe_layer(num_obj_1, num_obj_2, hidden[str(i - 1)], layernum[i - 1],
                                                    layernum[i], index_w_1, index_b_1, index_w_2, index_b_2,
                                                    coefficient,
                                                    name + 'layer' + str(i + 1), keep_prob, is_BN=is_BN,
                                                    is_transfer=is_transfer, transfer_function=hidden_transfer,
                                                    is_test=True)
            else:
                output = add_joint_pe_layer(num_obj_1, num_obj_2, hidden[str(i - 1)], layernum[i - 1],
                                            layernum[i], index_w_1, index_b_1, index_w_2, index_b_2, coefficient,
                                            name + 'layer' + str(i + 1), keep_prob, is_BN=False,
                                            is_transfer=False, transfer_function=None, is_test=True)
    return output


def penn_new(num_obj_1, num_obj_2, input, layernum, index_w_1, index_b_1, index_w_2, index_b_2, coefficient,
             hidden_transfer=tf.nn.sigmoid, output_transfer=tf.nn.sigmoid, keep_prob=1.0, name='0', is_transfer=True,
             is_BN=True, is_test=False, agg='sum'):
    input_shape = np.shape(input).as_list()
    hidden = dict()
    num_object = num_obj_1 * num_obj_2
    if len(layernum) == 1:
        output = add_joint_pe_layer_new(num_obj_1, num_obj_2, input,
                                        int(input_shape[len(input_shape) - 1]),
                                        layernum[0], index_w_1, index_b_1, index_w_2, index_b_2, coefficient,
                                        name + 'layer' + str(0 + 1), keep_prob, is_BN=is_BN,
                                        is_transfer=is_transfer, transfer_function=hidden_transfer, is_test=True,
                                        agg=agg)
    else:
        for i in range(len(layernum)):
            if i == 0:
                hidden[str(i)] = add_joint_pe_layer_new(num_obj_1, num_obj_2, input,
                                                        int(input_shape[len(input_shape) - 1]),
                                                        layernum[i], index_w_1, index_b_1, index_w_2, index_b_2,
                                                        coefficient,
                                                        name + 'layer' + str(i + 1), keep_prob, is_BN=is_BN,
                                                        is_transfer=is_transfer, transfer_function=hidden_transfer,
                                                        is_test=True, agg=agg)
            elif i != len(layernum) - 1:
                hidden[str(i)] = add_joint_pe_layer_new(num_obj_1, num_obj_2, hidden[str(i - 1)], layernum[i - 1],
                                                        layernum[i], index_w_1, index_b_1, index_w_2, index_b_2,
                                                        coefficient,
                                                        name + 'layer' + str(i + 1), keep_prob, is_BN=is_BN,
                                                        is_transfer=is_transfer, transfer_function=hidden_transfer,
                                                        is_test=True, agg=agg)
            else:
                output = add_joint_pe_layer_new(num_obj_1, num_obj_2, hidden[str(i - 1)], layernum[i - 1],
                                                layernum[i], index_w_1, index_b_1, index_w_2, index_b_2, coefficient,
                                                name + 'layer' + str(i + 1), keep_prob, is_BN=False,
                                                is_transfer=False, transfer_function=None, is_test=True, agg=agg)
    output = tf.reduce_sum(output, axis=[2, 3, 4])

    return output


def get_random_block_from_data(A, B, batch_size):
    start_index = np.random.randint(0, len(A) - batch_size)
    return A[start_index: start_index + batch_size], \
           B[start_index: start_index + batch_size]
