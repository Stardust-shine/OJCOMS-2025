import numpy as np


def norm_data(a_train, a_test):
    """
    在 a_train.shape==(B, K, Nt, 2)

    返回:
        out: shape == (B, K, Nt, 2)
    """
    a_train_norm = np.zeros(np.shape(a_train))
    a_test_norm = np.zeros(np.shape(a_test))
    a_shape_0 = np.shape(a_train)[0]
    a_shape_1 = np.shape(a_train)[1]
    a_shape_2 = np.shape(a_train)[2]
    mean_a_1 = np.sum(a_train[:, :, :, 0]) / a_shape_0 / a_shape_1 / a_shape_2
    var_a_1 = np.sqrt(np.sum(np.square(a_train[:, :, :, 0] - mean_a_1)) / a_shape_0 / a_shape_1 / a_shape_2)
    a_train_norm[:, :, :, 0] = (a_train[:, :, :, 0] - mean_a_1) / var_a_1

    mean_a_2 = np.sum(a_train[:, :, :, 1]) / a_shape_0 / a_shape_1 / a_shape_2
    var_a_2 = np.sqrt(np.sum(np.square(a_train[:, :, :, 1] - mean_a_2)) / a_shape_0 / a_shape_1 / a_shape_2)
    a_train_norm[:, :, :, 1] = (a_train[:, :, :, 1] - mean_a_2) / var_a_2

    a_test_norm[:, :, :, 0] = (a_test[:, :, :, 0] - mean_a_1) / var_a_1
    a_test_norm[:, :, :, 1] = (a_test[:, :, :, 1] - mean_a_2) / var_a_2

    return a_train_norm, a_test_norm
