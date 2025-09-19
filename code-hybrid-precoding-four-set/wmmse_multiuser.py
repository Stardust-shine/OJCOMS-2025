import numpy as np
from scipy.io import loadmat
import os
import scipy.io as sio


def get_mu(Pmax, interf, interf_k):
    Lambda, D = np.linalg.eig(interf)
    D_H = (D.conj()).T
    Phi = np.matmul(np.matmul(D_H, interf_k), D)
    Phi = np.real(np.diag(Phi))
    Lambda = np.real(Lambda)
    mu_upp = 1.0
    mu_low = 0.0
    mu = (mu_upp + mu_low) / 2
    P = np.sum(Phi / (Lambda + mu) ** 2)
    while P > Pmax:
        mu_low = mu_upp
        mu_upp = mu_upp * 2
        P = np.sum(Phi / (Lambda + mu_upp) ** 2)
    while abs(mu_upp - mu_low) > 1e-4:
        mu = (mu_upp + mu_low) / 2
        P = np.sum(Phi / (Lambda + mu) ** 2)
        if P < Pmax:
            mu_upp = mu
        else:
            mu_low = mu
    return mu


def WMMSE_BF(H, nUE_BS, nTX_BS, nTX_UE, Pmax_BS, var_noise):
    nBS = len(nUE_BS)
    nUE = sum(nUE_BS)

    V = np.zeros(sum(nUE_BS * nTX_BS), dtype=complex)

    iTX = 0
    for iBS in range(nBS):
        V[iTX: iTX + nUE_BS[iBS] * nTX_BS[iBS]] = np.sqrt(Pmax_BS[iBS] / (nTX_BS[iBS] * nUE_BS[iBS]))
        iTX = iTX + nUE_BS[iBS] * nTX_BS[iBS]

    U = np.zeros(nUE * nTX_UE, dtype=complex)
    W = np.zeros(nUE, dtype=complex)

    for iter in range(100):
        W_old = np.copy(W)
        for iBS in range(nBS):
            for iUE in range(nUE_BS[iBS]):
                iUE_idx = sum(nUE_BS[0:iBS]) + iUE
                interf = 0
                for jBS in range(nBS):
                    for jUE in range(nUE_BS[jBS]):
                        h = H[iUE_idx * nTX_UE: (iUE_idx + 1) * nTX_UE, sum(nTX_BS[0:jBS]): sum(nTX_BS[0:jBS + 1])]
                        v = V[sum(nUE_BS[0:jBS] * nTX_BS[0:jBS]) + nTX_BS[jBS] * jUE:
                              sum(nUE_BS[0:jBS] * nTX_BS[0:jBS]) + nTX_BS[jBS] * (jUE + 1)]
                        v = np.reshape(v, [-1, 1])
                        interf = interf + np.matmul(np.matmul(np.matmul(h, v), (v.conj()).T), (h.conj()).T)
                interf = interf + var_noise * np.eye(nTX_UE)
                h = H[iUE_idx * nTX_UE: (iUE_idx + 1) * nTX_UE, sum(nTX_BS[0:iBS]): sum(nTX_BS[0:iBS + 1])]
                v = V[sum(nUE_BS[0:iBS] * nTX_BS[0:iBS]) + nTX_BS[iBS] * iUE:
                      sum(nUE_BS[0:iBS] * nTX_BS[0:iBS]) + nTX_BS[iBS] * (iUE + 1)]
                U[iUE_idx * nTX_UE: (iUE_idx + 1) * nTX_UE] = np.matmul(np.matmul(np.linalg.inv(interf), h), v)

        for iBS in range(nBS):
            for iUE in range(nUE_BS[iBS]):
                iUE_idx = sum(nUE_BS[0:iBS]) + iUE
                h = H[iUE_idx * nTX_UE: (iUE_idx + 1) * nTX_UE, sum(nTX_BS[0:iBS]): sum(nTX_BS[0:iBS + 1])]
                v = V[sum(nUE_BS[0:iBS] * nTX_BS[0:iBS]) + nTX_BS[iBS] * iUE:
                      sum(nUE_BS[0:iBS] * nTX_BS[0:iBS]) + nTX_BS[iBS] * (iUE + 1)]
                u = U[iUE_idx * nTX_UE: (iUE_idx + 1) * nTX_UE]
                W[iUE_idx] = 1 / (1 - np.matmul(np.matmul((u.conj()).T, h), v))

        for iBS in range(nBS):
            interf = 0
            interf_k = 0
            for jBS in range(nBS):
                for jUE in range(nUE_BS[jBS]):
                    jUE_idx = sum(nUE_BS[0:jBS]) + jUE
                    h = H[jUE_idx * nTX_UE: (jUE_idx + 1) * nTX_UE, sum(nTX_BS[0:iBS]): sum(nTX_BS[0:iBS + 1])]
                    h_H = (h.conj()).T
                    u = U[jUE_idx * nTX_UE: (jUE_idx + 1) * nTX_UE]
                    u = np.reshape(u, [-1, 1])
                    u_H = (u.conj()).T
                    w = W[jUE_idx]
                    interf = interf + np.matmul(np.matmul(np.matmul(h_H, u) * w, u_H), h)
                    if jBS == iBS:
                        interf_k = interf_k + np.matmul(np.matmul(np.matmul(h_H, u) * (w ** 2), u_H), h)
            interf = interf + get_mu(Pmax_BS[iBS], interf, interf_k) * np.eye(nTX_BS[iBS])
            for iUE in range(nUE_BS[iBS]):
                iUE_idx = sum(nUE_BS[0:iBS]) + iUE
                h = H[iUE_idx * nTX_UE: (iUE_idx + 1) * nTX_UE, sum(nTX_BS[0:iBS]): sum(nTX_BS[0:iBS + 1])]
                h_H = (h.conj()).T
                u = U[iUE_idx * nTX_UE: (iUE_idx + 1) * nTX_UE]
                w = W[iUE_idx]
                V[sum(nUE_BS[0:iBS] * nTX_BS[0:iBS]) + nTX_BS[iBS] * iUE:
                  sum(nUE_BS[0:iBS] * nTX_BS[0:iBS]) + nTX_BS[iBS] * (iUE + 1)] = \
                    np.matmul(np.matmul(np.linalg.inv(interf), h_H), u) * w
        if abs(np.sum(np.log(W_old)) - np.sum(np.log(W))) < 1e-4:
            break

    return U, V


num_users = 3
user_num_an = 4
bs_num_an = 16
Ncl = 4
Nray = 5

file_dir = str(num_users) + "_N" + str(user_num_an) + "X" + str(bs_num_an) + "_Ncl" + str(Ncl) + "_Nray" + str(Nray)

data_test = loadmat("./data/setH_K" + file_dir + "_number" + str(10000) + ".mat")['H'].astype(np.float32)
data_test = np.transpose(data_test, axes=(0, 2, 3, 1))

H = data_test[:, :, :, 0] + np.complex('j') * data_test[:, :, :, 1]

num_sample = np.shape(H)[0]

U_all = np.zeros((num_sample, num_users * user_num_an), dtype=complex)
V_all = np.zeros((num_sample, num_users * bs_num_an), dtype=complex)
for num in range(num_sample):
    if num % 1000 == 0:
        print(num)
    h = H[num, :, :].reshape(num_users * user_num_an, bs_num_an)
    U, V = WMMSE_BF(h, np.array([num_users]), np.array([bs_num_an]), user_num_an, np.array([1]), 0.1)
    U_all[num, :] = U.reshape(1, -1)
    V_all[num, :] = V.reshape(1, -1)

save_dir = "./WMMSE-result/K" + file_dir
os.mkdir(save_dir)
sio.savemat(save_dir + "/wmmse.mat", {'H': H, 'U': U_all, 'V': V_all})
