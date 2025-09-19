import numpy as np
import math
import scipy.io as sio
import time


def ZF_precoding(H):
    HT = np.transpose(H)
    W = np.matmul(HT, np.linalg.inv(np.matmul(H, HT)))          # 求伪逆
    W = W / np.sqrt(np.sum(np.abs(W)**2, axis=0, keepdims=True))
    H_abs_ZF = np.diagonal(np.matmul(H, W))

    return W


def gen_H_ZF(Hmat, nUE_BS, nTX_BS):
    if len(nUE_BS) != len(nTX_BS):
        print('dimension error: len(nUE_BS)!=len(nTX_BS)')
        raise NameError

    nBS = len(nUE_BS)
    Wmat = np.zeros([Hmat.shape[1], Hmat.shape[0]]) + 1j * np.zeros([Hmat.shape[1], Hmat.shape[0]])
    iUE = 0
    iTX = 0
    for iBS in range(nBS):
        H_BS = Hmat[iUE:iUE+nUE_BS[iBS], iTX:iTX+nUE_BS[iBS]]
        W_BS = ZF_precoding(H_BS)
        Wmat[iTX:iTX+nUE_BS[iBS], iUE:iUE+nUE_BS[iBS]] = W_BS
        iUE = iUE + nUE_BS[iBS]
        iTX = iTX + nTX_BS[iBS]

    H_ZF = np.matmul(Hmat, Wmat)
    H_abs_ZF = np.abs(H_ZF)

    return H_abs_ZF


def WMMSE_algo(H_abs_ZF, nUE_BS, Pmax_BS, var_noise):

    H = H_abs_ZF

    if len(nUE_BS) != len(Pmax_BS):
        print('dimension error: len(nUE_BS)!=len(Pmax_BS)')
        raise NameError

    nBS = len(nUE_BS)
    nUE = sum(nUE_BS)
    v = np.zeros(nUE)
    rnew = 0

    iUE = 0

    for iBS in range(nBS):
        v[iUE: iUE+nUE_BS[iBS]] = np.sqrt(Pmax_BS[iBS]/nUE_BS[iBS])
        iUE = iUE + nUE_BS[iBS]

    u = np.zeros(nUE)
    w = np.zeros(nUE)

    for iUE in range(nUE):
        u[iUE] = H[iUE, iUE] * v[iUE] / (np.square(H[iUE, :]) @ np.square(v) + var_noise)
        w[iUE] = 1 / (1 - u[iUE] * H[iUE, iUE] * v[iUE])

    for iter in range(100):
        rold = rnew
        iUE = 0
        for iBS in range(nBS):
            jUE = iUE+nUE_BS[iBS]
            vtmp = np.sum(w[iUE:jUE] * u[iUE:jUE] * np.diagonal(H[iUE:jUE, iUE:jUE])) / \
                   np.sum(w * np.square(u) * np.sum(np.square(H[:, iUE:jUE]), axis=1))
            v[iUE: iUE+nUE_BS[iBS]] = min(vtmp, np.sqrt(Pmax_BS[iBS]/nUE_BS[iBS])) + \
                                      max(vtmp, 0) - vtmp
            iUE = iUE+nUE_BS[iBS]

        rnew = 0
        for iUE in range(nUE):
            u[iUE] = H[iUE, iUE] * v[iUE] / (np.square(H[iUE, :]) @ np.square(v) + var_noise)
            w[iUE] = 1 / (1 - u[iUE] * H[iUE, iUE] * v[iUE])
            rnew = rnew + math.log2(w[iUE])

        if abs(rnew - rold) <= 1e-3:
            break

    Popt_BS = np.zeros([nBS])
    iUE = 0
    for iBS in range(nBS):
        Popt_BS[iBS] = np.square(v[iUE]) * nUE_BS[iBS]
        iUE = iUE + nUE_BS[iBS]

    return Popt_BS


def cal_sum_rate(H_abs_ZF, Popt_BS, var_noise, nUE_BS):
    H = H_abs_ZF

    if len(nUE_BS) != len(Popt_BS):
        print('dimension error: len(nUE_BS)!=len(Popt_BS)')
        raise NameError

    nBS = len(nUE_BS)
    nUE = sum(nUE_BS)

    p = np.zeros([nUE])
    iUE = 0
    for iBS in range(nBS):
        p[iUE:iUE+nUE_BS[iBS]] = Popt_BS[iBS]/nUE_BS[iBS]
        iUE = iUE + nUE_BS[iBS]

    rate = 0.0
    for iUE in range(nUE):
        s = var_noise
        for jUE in range(nUE):
            if jUE != iUE:
                s = s + H[iUE, jUE]**2 * p[jUE]
        rate = rate + math.log2(1 + H[iUE, iUE]**2 * p[iUE] / s)

    return rate


def cal_sum_rate1(H_SET, P_SET, var_noise, nUE_BS, access_UE_BS):
    N_s = np.shape(H_SET)[0]
    nBS_max = int(access_UE_BS.shape[1])
    nUE_max = int(access_UE_BS.shape[2])

    if nUE_BS.ndim == 1:
        nUE_BS = np.ones([N_s, 1]) * np.reshape(nUE_BS, [1, -1])
    if int(access_UE_BS.shape[0])==1:
        access_UE_BS = np.ones([N_s, 1, 1]) * access_UE_BS

    SR = np.zeros(N_s)

    for i in range(N_s):
        P = P_SET[i]
        H = H_SET[i]

        HH = (H**2) * np.reshape(P/(nUE_BS[i]+1e-8), [-1, 1])
        SINR = np.sum(HH * access_UE_BS[i], axis=0) / (np.sum(HH * (1-access_UE_BS[i]), axis=0) + var_noise)
        SR[i] = sum(np.log2(1 + SINR))

    return sum(SR), SR


def gen_channel(nUE_BS, nTX_BS, Pmax_BS, var_noise, chl_model='Ray', KdB=0):
    if len(nUE_BS) != len(nTX_BS):
        print('dimension error: len(nUE_BS)!=len(nTX_BS)')
        raise NameError

    nUE = sum(nUE_BS)
    nTX = sum(nTX_BS)
    nBS = len(nUE_BS)

    Hmat = 1 / np.sqrt(2) * (np.random.randn(nUE, nTX) + 1j * np.random.randn(nUE, nTX))
    if chl_model == 'Ric':
        K = 10.0**(KdB / 10.0)
        Hmat = np.sqrt(K / (K + 1)) + np.sqrt(1 / (K + 1)) * Hmat
    start = time.clock()
    H_abs_ZF = gen_H_ZF(Hmat, nUE_BS, nTX_BS)
    Popt_BS = WMMSE_algo(H_abs_ZF, nUE_BS, Pmax_BS, var_noise)
    print(time.clock()-start)
    sum_rate = cal_sum_rate(H_abs_ZF, Popt_BS, var_noise, nUE_BS)

    H_UEBS = np.zeros([nUE, nBS])
    for iUE in range(nUE):
        for iBS in range(nBS):
            H_UEBS[iUE, iBS] = np.sqrt(np.sum(np.square(H_abs_ZF[iUE, sum(nUE_BS[0:iBS]): sum(nUE_BS[0:iBS+1])])))

    return H_UEBS, Popt_BS, sum_rate


def gen_sample(N_s, nUE_BS, nTX_BS, Pmax_BS, var_noise, chl_model='Ray', KdB=0):
    nUE = sum(nUE_BS)
    nBS = len(nUE_BS)

    X_H = np.zeros([N_s, nBS, nUE])
    Y_P = np.zeros([N_s, nBS])
    SR = np.zeros(N_s)

    for i in range(N_s):
        H, P, R = gen_channel(nUE_BS, nTX_BS, Pmax_BS, var_noise, chl_model, KdB)
        X_H[i] = np.transpose(H)
        Y_P[i] = P
        SR[i] = R

    X_H = np.reshape(X_H, [N_s, nBS, nUE, 1])
    X_P = np.ones([N_s, nBS, 1]) * np.reshape(Pmax_BS, [1, nBS, 1])
    X_UE = np.zeros([N_s, nUE, 1])
    Y_P = np.reshape(Y_P, [N_s, nBS, 1])

    return X_H, X_P, X_UE, Y_P, SR


def gen_sample_scalability(N_s, nUE_BS, nTX_BS, Pmax_BS, var_noise, nUE_max, nBS_max, chl_model='Ray', KdB=0):
    X_H = np.zeros([N_s, nBS_max, nUE_max])
    ACS = np.zeros([N_s, nBS_max, nUE_max])
    Y_P = np.zeros([N_s, nBS_max])
    SR = np.zeros(N_s)

    for i in range(N_s):
        nBS = sum(nUE_BS[i, :]!=0)
        nUE = sum(nUE_BS[i, :])
        for iBS in range(nBS):
            ACS[i, iBS, sum(nUE_BS[i, 0:iBS]):sum(nUE_BS[i, 0:iBS+1])] = 1
        H, P, R = gen_channel(nUE_BS[i, 0:nBS], nTX_BS[i, 0:nBS], Pmax_BS[i, 0:nBS], var_noise, chl_model, KdB)
        X_H[i, 0:nBS, 0:nUE] = np.transpose(H)
        Y_P[i, 0:nBS] = P
        SR[i] = R

    X_H = np.reshape(X_H, [N_s, nBS_max, nUE_max, 1])
    X_P = np.reshape(Pmax_BS, [N_s, nBS_max, 1])
    X_UE = np.zeros([N_s, nUE_max, 1])
    Y_P = np.reshape(Y_P, [N_s, nBS_max, 1])

    return X_H, X_P, X_UE, Y_P, ACS, SR


def preprocess_GeoYeLi(n_marco, n_pico, nUE_marco, nUE_pico, nUE_BS, N_s, data='Train'):
    filename = 'Dataset/'+ data + '_' + str(n_marco) + 'MBS_' + str(nUE_marco) + 'UE_' + \
                                   str(n_pico) + 'PBS_' + str(nUE_pico) + 'UE.mat'
    file = sio.loadmat(filename)
    A_BS_UE = file['A_BS_UE']; X_BS = file['X_BS']; X_UE = file['X_UE'];
    Y_BS = file['Y_BS']; SR = file['SR']
    A_BS_UE = A_BS_UE[0: N_s]; X_BS = X_BS[0: N_s]; X_UE = X_UE[0: N_s];
    Y_BS = Y_BS[0: N_s]; SR = SR[0: N_s]
    Cak = int(A_BS_UE.shape[3])
    nUE_max = max(nUE_marco, nUE_pico)
    nBS = n_marco + n_pico
    E = np.zeros([N_s, nBS, nBS*nUE_max, 1])
    H_BS = np.zeros([N_s, nBS, nUE_max*Cak])
    for iBS in range(nBS):
        E[:,:, iBS*nUE_max:iBS*nUE_max+nUE_BS[iBS], :] = \
            A_BS_UE[:, :, sum(nUE_BS[0:iBS]):sum(nUE_BS[0:iBS+1]), :]
    E = np.reshape(E, [N_s, nBS, nBS, nUE_max*Cak])
    for iBS in range(nBS):
        H_BS[:, iBS, :] = E[:, iBS, iBS, :]
        E[:, iBS, iBS, :] = 0.0
    E = np.concatenate((E, np.transpose(E, axes=[0,2,1,3])), axis=3)

    return nUE_max, A_BS_UE, E, X_BS, H_BS, Y_BS, SR


if __name__ == '__main__':
    nUE_BS = np.array([3, 3, 3, 4])
    nTX_BS = np.array([4, 4, 4, 6])
    Pmax_BS = np.array([1, 1, 1, 3])
    var_noise = 1
    H_UEBS, _, _, Popt_BS, sum_rate = gen_channel(nUE_BS, nTX_BS, Pmax_BS, var_noise)