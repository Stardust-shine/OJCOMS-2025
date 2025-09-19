import numpy as np


def top_n_rows_per_block(A, N, M):
    B, CK, CM, U = A.shape
    assert U == 2
    C = CM // M
    A_blocked = (
        A.reshape(B, CK, C, M, U).transpose(0, 2, 4, 1, 3)
    )
    B2, C2, U2, CK2, M2 = A_blocked.shape
    S = B2 * C2 * U2
    slices = A_blocked.reshape(S, CK2, M2)

    row_scores = slices[:, :, 0]
    idx = np.argpartition(-row_scores, N - 1, axis=1)[:, :N]
    selected = np.take_along_axis(
        slices,
        idx[:, :, None],  # -> (S, N, 1)
        axis=1
    )
    selected = selected.reshape(B2, C2, U2, N, M2)
    out = (
        selected.transpose(0, 3, 1, 4, 2).reshape(B2, N, C2 * M2, U2)
    )
    return out


def norm_data(a_train, a_test, b_train, b_test):
    a_train_norm = np.zeros(np.shape(a_train))
    a_test_norm = np.zeros(np.shape(a_test))
    a_shape_0 = np.shape(a_train)[0]
    a_shape_1 = np.shape(a_train)[1]
    a_shape_2 = np.shape(a_train)[2]
    a_shape_3 = np.shape(a_train)[3]
    mean_a_1 = np.sum(a_train[:, :, :, :, 0]) / a_shape_0 / a_shape_1 / a_shape_2 / a_shape_3
    var_a_1 = np.sqrt(np.sum(np.square(a_train[:, :, :, :, 0] - mean_a_1)) / a_shape_0 / a_shape_1 / a_shape_2 / a_shape_3)
    a_train_norm[:, :, :, :, 0] = (a_train[:, :, :, :, 0] - mean_a_1) / var_a_1

    mean_a_2 = np.sum(a_train[:, :, :, :, 1]) / a_shape_0 / a_shape_1 / a_shape_2 / a_shape_3
    var_a_2 = np.sqrt(np.sum(np.square(a_train[:, :, :, :, 1] - mean_a_2)) / a_shape_0 / a_shape_1 / a_shape_2 / a_shape_3)
    a_train_norm[:, :, :, :, 1] = (a_train[:, :, :, :, 1] - mean_a_2) / var_a_2

    a_test_norm[:, :, :, :, 0] = (a_test[:, :, :, :, 0] - mean_a_1) / var_a_1
    a_test_norm[:, :, :, :, 1] = (a_test[:, :, :, :, 1] - mean_a_2) / var_a_2

    b_train_norm = np.zeros(np.shape(b_train))
    b_test_norm = np.zeros(np.shape(b_test))
    b_shape_0 = np.shape(b_train)[0]
    b_shape_1 = np.shape(b_train)[1]
    b_shape_2 = np.shape(b_train)[2]
    b_shape_3 = np.shape(b_train)[3]
    mean_b_1 = np.sum(b_train[:, :, :, :, 0]) / b_shape_0 / b_shape_1 / b_shape_2 / b_shape_3
    var_b_1 = np.sqrt(np.sum(np.square(b_train[:, :, :, :, 0] - mean_b_1)) / b_shape_0 / b_shape_1 / b_shape_2 / b_shape_3)
    b_train_norm[:, :, :, :, 0] = (b_train[:, :, :, :, 0] - mean_b_1) / var_b_1

    mean_b_2 = np.sum(b_train[:, :, :, :, 1]) / b_shape_0 / b_shape_1 / b_shape_2 / b_shape_3
    var_b_2 = np.sqrt(np.sum(np.square(b_train[:, :, :, :, 1] - mean_b_2)) / b_shape_0 / b_shape_1 / b_shape_2 / b_shape_3)
    b_train_norm[:, :, :, :, 1] = (b_train[:, :, :, :, 1] - mean_b_2) / var_b_2

    b_test_norm[:, :, :, :, 0] = (b_test[:, :, :, :, 0] - mean_b_1) / var_b_1
    b_test_norm[:, :, :, :, 1] = (b_test[:, :, :, :, 1] - mean_b_2) / var_b_2

    return a_train_norm, a_test_norm, b_train_norm, b_test_norm
