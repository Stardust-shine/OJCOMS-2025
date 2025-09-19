import numpy as np
import math
import scipy.io as sio
import time
from powercontrol_wmmse import ZF_precoding, gen_H_ZF


def WMMSE_algo(H_abs_ZF, nUE_BS, Pmax_BS, var_noise, constraint='perBS'):
    H = H_abs_ZF

    # weight = np.divide(1, np.log2(1 + np.diag(H) * np.repeat(Pmax_BS, nUE_BS) / var_noise))  # [总用户数]
    # weight = weight / np.sum(weight)

    if len(nUE_BS) != len(Pmax_BS):
        print('dimension error: len(nUE_BS)!=len(Pmax_BS)')
        raise NameError

    nBS = len(nUE_BS)
    nUE = sum(nUE_BS)
    v = np.zeros(nUE)
    v_max = np.zeros(nUE)
    rnew = 0

    iUE = 0

    for iBS in range(nBS):
        v[iUE: iUE+nUE_BS[iBS]] = np.sqrt(Pmax_BS[iBS]/nUE_BS[iBS])
        v_max[iUE: iUE+nUE_BS[iBS]] = np.sqrt(Pmax_BS[iBS])
        iUE = iUE + nUE_BS[iBS]

    u = np.zeros(nUE)
    w = np.zeros(nUE)

    for iter in range(100):
        rold = rnew
        for iUE in range(nUE):
            u[iUE] = H[iUE, iUE] * v[iUE] / (np.square(H[iUE, :]) @ np.square(v) + var_noise)
        for iUE in range(nUE):
            w[iUE] = 1 / (1 - u[iUE] * H[iUE, iUE] * v[iUE])
        v_nmt = np.zeros(nUE)
        v_dmt = np.zeros(nUE)
        if constraint == 'perBS':
            for iUE in range(nUE):
                v_nmt[iUE] = H[iUE, iUE] * u[iUE] * w[iUE]
                v_dmt[iUE] = np.sum(np.square(H[:, iUE]) * w * np.square(u))
            iUE = 0
            LAMDA = np.zeros(nUE)
            for iBS in range(nBS):
                lamda_l = 0.0; lamda_h = 1000.0
                # previous: while lamda_h - lamda_l > 1e-8 or sum_p > Pmax_BS[iBS]:
                while lamda_h - lamda_l > 1e-2 or sum_p > Pmax_BS[iBS]:
                    lamda = (lamda_l + lamda_h)/2
                    vtmp = v_nmt/(v_dmt + lamda)
                    sum_p = np.sum(np.square(vtmp[iUE: iUE+nUE_BS[iBS]]))
                    if sum_p > Pmax_BS[iBS]:
                        lamda_l = lamda
                    else:
                        lamda_h = lamda
                LAMDA[iUE: iUE+nUE_BS[iBS]] = lamda
                iUE = iUE + nUE_BS[iBS]
            v = v_nmt / (v_dmt + LAMDA)
        else:
            vtmp = np.zeros(nUE)
            for iUE in range(nUE):
                v_nmt[iUE] = H[iUE, iUE] * u[iUE] * w[iUE]
                v_dmt[iUE] = np.sum(np.square(H[:, iUE]) * w * np.square(u))
                vtmp[iUE] = v_nmt[iUE]/v_dmt[iUE]
            v = np.minimum(vtmp, v_max) + np.maximum(vtmp, 0) - vtmp

        for iUE in range(nUE):
            u[iUE] = H[iUE, iUE] * v[iUE] / (np.square(H[iUE, :]) @ np.square(v) + var_noise)
            w[iUE] = 1 / (1 - u[iUE] * H[iUE, iUE] * v[iUE])
            rnew = rnew + math.log2(w[iUE])

        if abs(rnew - rold) <= 1e-3:
            break

    Popt_UE = np.square(v)
    return Popt_UE


def WMMSE_algo_weight(H_abs_ZF, nUE_BS, Pmax_BS, var_noise, constraint='perBS'):
    H = H_abs_ZF

    # weight = np.divide(1, np.log2(1 + np.diag(np.square(H)) * np.repeat(Pmax_BS, nUE_BS) / var_noise))  # [总用户数]
    # weight = weight / np.sum(weight)

    weight = np.divide(1, np.abs(np.log2(np.diag(np.square(H)) * np.repeat(Pmax_BS, nUE_BS) / var_noise)))  # [总用户数]
    # weight = weight / np.sum(weight)

    if len(nUE_BS) != len(Pmax_BS):
        print('dimension error: len(nUE_BS)!=len(Pmax_BS)')
        raise NameError

    nBS = len(nUE_BS)
    nUE = sum(nUE_BS)
    v = np.zeros(nUE)
    v_max = np.zeros(nUE)
    rnew = 0

    iUE = 0

    for iBS in range(nBS):
        v[iUE: iUE+nUE_BS[iBS]] = np.sqrt(Pmax_BS[iBS]/nUE_BS[iBS])
        v_max[iUE: iUE+nUE_BS[iBS]] = np.sqrt(Pmax_BS[iBS])
        iUE = iUE + nUE_BS[iBS]

    u = np.zeros(nUE)
    w = np.zeros(nUE)

    for iter in range(100):
        rold = rnew
        for iUE in range(nUE):
            u[iUE] = H[iUE, iUE] * v[iUE] / (np.square(H[iUE, :]) @ np.square(v) + var_noise)
        for iUE in range(nUE):
            w[iUE] = 1 / (1 - u[iUE] * H[iUE, iUE] * v[iUE])
        v_nmt = np.zeros(nUE)
        v_dmt = np.zeros(nUE)
        if constraint == 'perBS':
            for iUE in range(nUE):
                v_nmt[iUE] = weight[iUE] * H[iUE, iUE] * u[iUE] * w[iUE]
                v_dmt[iUE] = np.sum(weight * np.square(H[:, iUE]) * w * np.square(u))
            iUE = 0
            LAMDA = np.zeros(nUE)
            for iBS in range(nBS):
                lamda_l = 0.0; lamda_h = 1000.0
                # previous: while lamda_h - lamda_l > 1e-8 or sum_p > Pmax_BS[iBS]:
                while lamda_h - lamda_l > 1e-2 or sum_p > Pmax_BS[iBS]:
                    lamda = (lamda_l + lamda_h)/2
                    vtmp = v_nmt/(v_dmt + lamda)
                    sum_p = np.sum(np.square(vtmp[iUE: iUE+nUE_BS[iBS]]))
                    if sum_p > Pmax_BS[iBS]:
                        lamda_l = lamda
                    else:
                        lamda_h = lamda
                LAMDA[iUE: iUE+nUE_BS[iBS]] = lamda
                iUE = iUE + nUE_BS[iBS]
            v = v_nmt / (v_dmt + LAMDA)
        else:
            vtmp = np.zeros(nUE)
            for iUE in range(nUE):
                v_nmt[iUE] = H[iUE, iUE] * u[iUE] * w[iUE]
                v_dmt[iUE] = np.sum(np.square(H[:, iUE]) * w * np.square(u))
                vtmp[iUE] = v_nmt[iUE]/v_dmt[iUE]
            v = np.minimum(vtmp, v_max) + np.maximum(vtmp, 0) - vtmp

        for iUE in range(nUE):
            u[iUE] = H[iUE, iUE] * v[iUE] / (np.square(H[iUE, :]) @ np.square(v) + var_noise)
            w[iUE] = 1 / (1 - u[iUE] * H[iUE, iUE] * v[iUE])
            rnew = rnew + math.log2(w[iUE])

        if abs(rnew - rold) <= 1e-3:
            break

    Popt_UE = np.square(v)
    return Popt_UE


def cal_sum_rate(H_abs_ZF, Popt_UE, var_noise, nUE_BS):
    H = H_abs_ZF

    nBS = len(nUE_BS)
    nUE = sum(nUE_BS)

    p = Popt_UE

    rate = 0.0
    rate_ue = np.zeros(nUE)
    for iUE in range(nUE):
        s = var_noise
        for jUE in range(nUE):
            if jUE != iUE:
                s = s + H[iUE, jUE]**2 * p[jUE]
        r = math.log2(1 + H[iUE, iUE]**2 * p[iUE] / s)
        rate_ue[iUE] = r
        rate = rate + r

    return rate, rate_ue


def cal_sum_rate1(H_SET, P_SET, var_noise, nUE_BS, nTX_BS, access_UE_BS):
    N_s = np.shape(H_SET)[0]

    if nUE_BS.ndim == 1:
        nUE_BS = np.int64(np.ones([N_s, 1]) * np.reshape(nUE_BS, [1, -1]))
    H_SET = np.transpose(H_SET[:, :, :, 0], axes=[0, 2, 1])
    HH = H_SET * H_SET
    eye = np.expand_dims(np.eye(sum(nUE_BS[0])), axis=0)
    P_SET = np.expand_dims(P_SET, axis=1)
    rate_user = np.log2(1 + np.sum(eye * HH * P_SET, axis=2) / (np.sum((1 - eye) * HH * P_SET, axis=2) + var_noise))
    SR = np.sum(rate_user, axis=1)

    return sum(SR), rate_user


def gen_channel(nUE_BS, nTX_BS, Pmax_BS, var_noise, config, chl_model='Ray', KdB=0):
    if len(nUE_BS) != len(nTX_BS):
        print('dimension error: len(nUE_BS)!=len(nTX_BS)')
        raise NameError

    nUE = sum(nUE_BS)
    nTX = sum(nTX_BS)
    nBS = len(nUE_BS)

    Alp_mat, dist = gen_pathloss(config['n_marco'], config['n_pico'], config['nUE_marco'], config['nUE_pico'],
                                 config['nTX_marco'], config['nTX_pico'])
    Hmat = 1 / np.sqrt(2) * (np.random.randn(nUE, nTX) + 1j * np.random.randn(nUE, nTX))
    if chl_model == 'Ric':
        K = 10.0**(KdB / 10.0)
        Hmat = np.sqrt(K / (K + 1)) + np.sqrt(1 / (K + 1)) * Hmat
    Hmat = Hmat * np.sqrt(Alp_mat)
    # start = time.clock()
    H_abs_ZF = gen_H_ZF(Hmat, nUE_BS, nTX_BS)
    Popt_UE = WMMSE_algo(H_abs_ZF, nUE_BS, Pmax_BS, var_noise)
    # test_time = time.clock() - start
    # print('%.5f' % test_time)
    sum_rate, rate_user = cal_sum_rate(H_abs_ZF, Popt_UE, var_noise, nUE_BS)

    return H_abs_ZF, Popt_UE, sum_rate, rate_user, dist


def gen_sample(N_s, nUE_BS, nTX_BS, Pmax_BS, var_noise, config=None, chl_model='Ray', KdB=0, disp_space=100):
    nUE = sum(nUE_BS)
    nBS = len(nUE_BS)

    X_H = np.zeros([N_s, nUE, nUE])
    Y_P = np.zeros([N_s, nUE])
    X_D = np.zeros([N_s, nBS, nUE])
    SR = np.zeros(N_s)
    SR_UE = np.zeros([N_s, nUE])

    for i in range(N_s):
        H, P, R, R_U, D = gen_channel(nUE_BS, nTX_BS, Pmax_BS, var_noise, config, chl_model, KdB)
        X_H[i] = np.transpose(H)
        Y_P[i] = P
        X_D[i] = D
        SR[i] = R
        SR_UE[i] = R_U
        if (i+1) % disp_space == 0:
            print('The ' + str(i+1) + ' th sample has been generated...')

    X_H = np.reshape(X_H, [N_s, nUE, nUE, 1])
    X_D = np.reshape(X_D, [N_s, nBS, nUE, 1])
    X_P = np.ones([N_s, nBS, 1]) * np.reshape(Pmax_BS, [1, nBS, 1])
    X_UE = np.zeros([N_s, nUE, 1])
    Y_P = np.reshape(Y_P, [N_s, nUE, 1])

    return X_H, X_P, X_UE, Y_P, SR, SR_UE, X_D


def gen_channel_weight(nUE_BS, nTX_BS, Pmax_BS, var_noise, config, chl_model='Ray', KdB=0):

    if len(nUE_BS) != len(nTX_BS):
        print('dimension error: len(nUE_BS)!=len(nTX_BS)')
        raise NameError

    nUE = sum(nUE_BS)
    nTX = sum(nTX_BS)
    nBS = len(nUE_BS)

    Alp_mat, dist = gen_pathloss(config['n_marco'], config['n_pico'], config['nUE_marco'], config['nUE_pico'],
                                 config['nTX_marco'], config['nTX_pico'])
    Hmat = 1 / np.sqrt(2) * (np.random.randn(nUE, nTX) + 1j * np.random.randn(nUE, nTX))
    if chl_model == 'Ric':
        K = 10.0**(KdB / 10.0)
        Hmat = np.sqrt(K / (K + 1)) + np.sqrt(1 / (K + 1)) * Hmat
    Hmat = Hmat * np.sqrt(Alp_mat)
    # start = time.clock()
    H_abs_ZF = gen_H_ZF(Hmat, nUE_BS, nTX_BS)
    Popt_UE = WMMSE_algo_weight(H_abs_ZF, nUE_BS, Pmax_BS, var_noise)
    # test_time = time.clock() - start
    # print('%.5f' % test_time)
    sum_rate, rate_user = cal_sum_rate(H_abs_ZF, Popt_UE, var_noise, nUE_BS)

    return H_abs_ZF, Popt_UE, sum_rate, rate_user, dist


def gen_sample_weight(N_s, nUE_BS, nTX_BS, Pmax_BS, var_noise, config=None, chl_model='Ray', KdB=0, disp_space=100):
    nUE = sum(nUE_BS)
    nBS = len(nUE_BS)

    X_H = np.zeros([N_s, nUE, nUE])
    Y_P = np.zeros([N_s, nUE])
    X_D = np.zeros([N_s, nBS, nUE])
    SR = np.zeros(N_s)
    SR_UE = np.zeros([N_s, nUE])

    for i in range(N_s):
        H, P, R, R_U, D = gen_channel_weight(nUE_BS, nTX_BS, Pmax_BS, var_noise, config, chl_model, KdB)
        X_H[i] = np.transpose(H)
        Y_P[i] = P
        X_D[i] = D
        SR[i] = R
        SR_UE[i] = R_U
        if (i+1) % disp_space == 0:
            print('The ' + str(i+1) + ' th sample has been generated...')

    X_H = np.reshape(X_H, [N_s, nUE, nUE, 1])
    X_D = np.reshape(X_D, [N_s, nBS, nUE, 1])
    X_P = np.ones([N_s, nBS, 1]) * np.reshape(Pmax_BS, [1, nBS, 1])
    X_UE = np.zeros([N_s, nUE, 1])
    Y_P = np.reshape(Y_P, [N_s, nUE, 1])

    return X_H, X_P, X_UE, Y_P, SR, SR_UE, X_D


def gen_channel_real(H, nUE_BS, nTX_BS, Pmax_BS, var_noise, config, chl_model='Ray', KdB=0):

    if len(nUE_BS) != len(nTX_BS):
        print('dimension error: len(nUE_BS)!=len(nTX_BS)')
        raise NameError

    nUE = sum(nUE_BS)
    nTX = sum(nTX_BS)
    nBS = len(nUE_BS)

    Alp_mat, dist = gen_pathloss(config['n_marco'], config['n_pico'], config['nUE_marco'], config['nUE_pico'],
                                 config['nTX_marco'], config['nTX_pico'])
    Hmat = H
    H_abs_ZF = gen_H_ZF(Hmat, nUE_BS, nTX_BS)
    Popt_UE = WMMSE_algo(H_abs_ZF, nUE_BS, Pmax_BS, var_noise)
    sum_rate, rate_user = cal_sum_rate(H_abs_ZF, Popt_UE, var_noise, nUE_BS)

    return H_abs_ZF, Popt_UE, sum_rate, rate_user, dist


def gen_sample_real(Hmat, N_s, nUE_BS, nTX_BS, Pmax_BS, var_noise, config=None, chl_model='Ray',
                    KdB=0, disp_space=100):
    nUE = sum(nUE_BS)
    nBS = len(nUE_BS)

    X_H = np.zeros([N_s, nUE, nUE])
    Y_P = np.zeros([N_s, nUE])
    X_D = np.zeros([N_s, nBS, nUE])
    SR = np.zeros(N_s)
    SR_UE = np.zeros([N_s, nUE])

    for i in range(N_s):
        H, P, R, R_U, D = gen_channel_real(Hmat[i, :, :], nUE_BS, nTX_BS, Pmax_BS, var_noise, config, chl_model, KdB)
        X_H[i] = np.transpose(H)
        Y_P[i] = P
        X_D[i] = D
        SR[i] = R
        SR_UE[i] = R_U
        if (i+1) % disp_space == 0:
            print('The ' + str(i+1) + ' th sample has been generated...')

    X_H = np.reshape(X_H, [N_s, nUE, nUE, 1])
    X_D = np.reshape(X_D, [N_s, nBS, nUE, 1])
    X_P = np.ones([N_s, nBS, 1]) * np.reshape(Pmax_BS, [1, nBS, 1])
    X_UE = np.zeros([N_s, nUE, 1])
    Y_P = np.reshape(Y_P, [N_s, nUE, 1])

    return X_H, X_P, X_UE, Y_P, SR, SR_UE, X_D


def gen_sample_scalability(N_s, n_marco, n_pico, nUE_marco, nUE_pico, nTX_marco, nTX_pico, nUE_max, nBS_max,
                           chl_model='Ray', KdB=0):

    X_H = np.zeros([N_s, nUE_max, nUE_max])
    ACS = np.zeros([N_s, nUE_max, nUE_max])
    Y_P = np.zeros([N_s, nUE_max])
    SR = np.zeros(N_s)

    for i in range(N_s):
        nBS = sum(nUE_BS[i, :]!=0)
        nUE = sum(nUE_BS[i, :])
        for iBS in range(nBS):
            ACS[i, sum(nUE_BS[0:iBS]): sum(nUE_BS[0:iBS+1]), sum(nUE_BS[i, 0:iBS]):sum(nUE_BS[i, 0:iBS+1])] = 1
        H, P, R, _ = gen_channel(nUE_BS[i, 0:nBS], nTX_BS[i, 0:nBS], Pmax_BS[i, 0:nBS], var_noise, chl_model, KdB)
        X_H[i, 0:nUE, 0:nUE] = np.transpose(H)
        Y_P[i, 0:nUE] = P
        SR[i] = R

    X_H = np.reshape(X_H, [N_s, nUE_max, nUE_max, 1])
    X_P = np.reshape(Pmax_BS, [N_s, nBS_max, 1])
    X_UE = np.zeros([N_s, nUE_max, 1])
    Y_P = np.reshape(Y_P, [N_s, nUE_max, 1])

    return X_H, X_P, X_UE, Y_P, ACS, SR


def preprocess_GeoYeLi(n_marco, n_pico, nUE_marco, nUE_pico, nUE_BS, N_s, data='Train'):
    filename = 'Dataset/'+ data + '_' + str(n_marco) + 'MBS_' + str(nUE_marco) + 'UE_' + \
                                   str(n_pico) + 'PBS_' + str(nUE_pico) + 'UEA1.mat'
    file = sio.loadmat(filename)
    A_BS_UE = file['A_BS_UE']; X_BS = file['X_BS']; X_UE = file['X_UE'];
    Y_BS = file['Y_BS']; SR = file['SR']
    A_BS_UE = A_BS_UE[0: N_s]; X_BS = X_BS[0: N_s]; X_UE = X_UE[0: N_s];
    Y_BS = Y_BS[0: N_s]; SR = SR[0: N_s]
    Cak = int(A_BS_UE.shape[3])
    nUE_max = max(nUE_marco, nUE_pico)
    nUE_max = 10
    nBS = n_marco + n_pico
    E = np.zeros([N_s, nBS*nUE_max, nBS*nUE_max, 1])
    H_BS = np.zeros([N_s, nBS, nUE_max*nUE_max*Cak])
    Y_BS1 = np.zeros([N_s, nBS, nUE_max])
    for iBS in range(nBS):
        E[:, iBS*nUE_max:iBS*nUE_max+nUE_BS[iBS], iBS*nUE_max:iBS*nUE_max+nUE_BS[iBS], :] = \
            A_BS_UE[:, sum(nUE_BS[0:iBS]):sum(nUE_BS[0:iBS+1]), sum(nUE_BS[0:iBS]):sum(nUE_BS[0:iBS+1]), :]
        Y_BS1[:, iBS, 0:nUE_BS[iBS]] = Y_BS[:, sum(nUE_BS[0:iBS]):sum(nUE_BS[0:iBS+1]), 0]
    E0 = E
    E = np.array(np.split(np.array(np.split(np.array(np.split(E, nBS, axis=2)), nBS, axis=2)), N_s, axis=2))
    # E = np.transpose(E, [2, 0, 1, 3, 4, 5])
    E = np.reshape(E, [N_s, nBS, nBS, nUE_max * nUE_max * Cak])
    for iBS in range(nBS):
        H_BS[:, iBS, :] = E[:, iBS, iBS, :]
        # E[:, iBS, iBS, :] = 0.0
    E = np.concatenate((E, np.transpose(E, axes=[0, 2, 1, 3])), axis=3)

    return nUE_max, A_BS_UE, E, X_BS, H_BS, Y_BS1, SR, E0


def power_nomalize(p_pred, p_max, nUE_BS):
    nBS = len(nUE_BS)
    for iBS in range(nBS):
        p_pred[:, sum(nUE_BS[0:iBS]):sum(nUE_BS[0:iBS+1])] = p_pred[:, sum(nUE_BS[0:iBS]):sum(nUE_BS[0:iBS+1])] * \
                        np.minimum(p_max[:, iBS: iBS+1]/ \
                        (np.sum(p_pred[:, sum(nUE_BS[0:iBS]):sum(nUE_BS[0:iBS+1])], axis=1, keepdims=True)+1e-4), 1)
    return p_pred


def gen_pathloss_old(n_marco, n_pico, nUE_marco, nUE_pico, nTX_marco, nTX_pico, R_marco=240.0, R_pico=40.0,
                     alpha_mBS=13.54, beta_mBS=39.08, alpha_pBS=32.4, beta_pBS=31.9, h_p=1.0, h_m=30.0):
    D_marco = 2 * R_marco; D_pico = 2 * R_pico
    posi_marco = np.reshape(np.linspace(0, (n_marco - 1) * D_marco, n_marco), [-1, 1])
    # posi_pico = np.reshape(np.linspace(0, (n_pico - 1) * D_pico, n_pico) + 1j * 120.0, [-1, 1])
    theta_pBS = np.linspace(0, n_pico - 1, n_pico) / n_pico * 2 * np.pi
    posi_pico = np.reshape((np.cos(theta_pBS) + 1j * np.sin(theta_pBS)) * 2 * D_pico, [-1, 1])

    dis_pico = np.random.rand(n_pico, nUE_pico) * (R_pico-h_p) + h_p
    theta_pico = np.random.rand(n_pico, nUE_pico) * 2 * np.pi
    posi_pUE = (np.cos(theta_pico) + 1j * np.sin(theta_pico)) * dis_pico + posi_pico

    dis_marco = np.random.rand(n_marco, nUE_marco) * (R_marco-h_m) + h_m
    theta_marco = np.random.rand(n_marco, nUE_marco)*2*np.pi
    posi_mUE = (np.cos(theta_marco) + 1j * np.sin(theta_marco)) * dis_marco + posi_marco

    posi_allUE = np.concatenate((np.reshape(posi_pUE, [1, -1]), np.reshape(posi_mUE, [1, -1])), axis=1)
    posi_allBS = np.concatenate((np.reshape(posi_pico, [-1, 1]), np.reshape(posi_marco, [-1, 1])), axis=0)
    dist0 = abs(posi_allBS - posi_allUE)

    posi_allBS = np.concatenate((np.reshape(posi_pico * np.ones([1, nTX_pico]), [-1, 1]),
                                 np.reshape(posi_marco * np.ones([1, nTX_marco]), [-1, 1])), axis=0)

    dist = abs(posi_allBS - posi_allUE)

    alpha0 = np.concatenate((np.ones([n_pico*nTX_pico, 1])*alpha_pBS, np.ones([n_marco*nTX_marco, 1])*alpha_mBS),axis=0)
    beta = np.concatenate((np.ones([n_pico*nTX_pico, 1]) * beta_pBS, np.ones([n_marco*nTX_marco, 1])*beta_mBS), axis=0)
    alpha_dB = -1.0*(alpha0 + beta * np.log10(dist))
    alpha = np.power(10.0, alpha_dB/10.0)

    return np.transpose(alpha), dist0


def gen_pathloss(n_marco, n_pico, nUE_marco, nUE_pico, nTX_marco, nTX_pico, R_marco=240.0, R_pico=60,
                 alpha_mBS=13.54, beta_mBS=39.08, alpha_pBS=32.4, beta_pBS=31.9, h_p=1.0, h_m=30.0):
    # R_pico=40
    D_marco = 2 * R_marco; D_pico = 2 * R_pico
    posi_marco = np.reshape(np.linspace(0, (n_marco - 1) * D_marco, n_marco), [-1, 1])
    theta_pBS = np.linspace(0, n_pico - 1, n_pico) / n_pico * 2 * np.pi
    posi_pico = np.reshape((np.cos(theta_pBS) + 1j * np.sin(theta_pBS)) * (D_pico/np.sqrt(2)), [-1, 1])
    # posi_pico = np.reshape((np.cos(theta_pBS) + 1j * np.sin(theta_pBS)) * R_pico/np.sqrt(2), [-1, 1])
    # R = R_pico / np.sin(np.pi / n_pico)
    # posi_pico = np.reshape((np.cos(theta_pBS) + 1j * np.sin(theta_pBS)) * R, [-1, 1])

    dis_pico = np.random.rand(n_pico, nUE_pico) * (R_pico-h_p) + h_p
    theta_pico = np.random.rand(n_pico, nUE_pico) * 2 * np.pi
    posi_pUE = (np.cos(theta_pico) + 1j * np.sin(theta_pico)) * dis_pico + posi_pico

    dis_marco = np.random.rand(n_marco, nUE_marco) * (R_marco-h_m) + h_m
    theta_marco = np.random.rand(n_marco, nUE_marco)*2*np.pi
    posi_mUE = (np.cos(theta_marco) + 1j * np.sin(theta_marco)) * dis_marco + posi_marco

    posi_allUE = np.concatenate((np.reshape(posi_pUE, [1, -1]), np.reshape(posi_mUE, [1, -1])), axis=1)
    posi_allBS = np.concatenate((np.reshape(posi_pico, [-1, 1]), np.reshape(posi_marco, [-1, 1])), axis=0)
    dist0 = abs(posi_allBS - posi_allUE)

    posi_allBS = np.concatenate((np.reshape(posi_pico * np.ones([1, nTX_pico]), [-1, 1]),
                                 np.reshape(posi_marco * np.ones([1, nTX_marco]), [-1, 1])), axis=0)

    dist = abs(posi_allBS - posi_allUE)

    alpha0 = np.concatenate((np.ones([n_pico*nTX_pico, 1])*alpha_pBS, np.ones([n_marco*nTX_marco, 1])*alpha_mBS),axis=0)
    beta = np.concatenate((np.ones([n_pico*nTX_pico, 1]) * beta_pBS, np.ones([n_marco*nTX_marco, 1])*beta_mBS), axis=0)
    alpha_dB = -1.0*(alpha0 + beta * np.log10(dist) + 20 * np.log10(5))
    large_scale = np.power(10.0, alpha_dB/10.0)

    log_shadowing = np.random.normal(0, 8, size=np.shape(large_scale))
    shadowing = np.power(10.0, -log_shadowing/10.0)

    alpha = np.multiply(large_scale, shadowing)

    return np.transpose(alpha), dist0


if __name__ == '__main__':
    nUE_BS = np.array([5, 5, 5, 5, 5, 10])
    nTX_BS = np.array([8, 8, 8, 8, 8, 16])
    Pmax_BS = np.array([1, 1, 1, 1, 1, 40])
    var_noise = 3.162e-6
    config = dict()
    config['n_marco'] = 1
    config['n_pico'] = 5
    config['nUE_marco'] = 10
    config['nUE_pico'] = 5
    config['nTX_marco'] = 16
    config['nTX_pico'] = 8
    H_abs_ZF, Popt_UE, sum_rate, dist = gen_channel(nUE_BS, nTX_BS, Pmax_BS, var_noise, config)

    '''
    for k in range(1000):
        p_test = np.random.rand(30)
        p_test[0:5] = p_test[0:5] / np.sum(p_test[0:5])*Pmax_BS[0]
        p_test[5:10] = p_test[5:10] / np.sum(p_test[5:10]) * Pmax_BS[1]
        p_test[11:15] = p_test[11:15] / np.sum(p_test[11:15]) * Pmax_BS[2]
        p_test[15:20] = p_test[15:20] / np.sum(p_test[15:20]) * Pmax_BS[3]
        p_test[20:25] = p_test[20:25] / np.sum(p_test[20:25]) * Pmax_BS[4]
        p_test[25:30] = p_test[25:30] / np.sum(p_test[25:30]) * Pmax_BS[5]
        sum_rate1 = cal_sum_rate(H_abs_ZF, p_test, var_noise, nUE_BS)
        if sum_rate1 > sum_rate:
            sum_rate_temp = sum_rate1
            p_temp = p_test
            print('Error!')
    '''