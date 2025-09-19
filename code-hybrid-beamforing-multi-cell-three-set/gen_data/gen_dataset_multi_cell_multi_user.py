import numpy as np
import sys
import time
import scipy.io as sio
import os
import h5py
import random

# SEED = 42
# np.random.seed(SEED)
# random.seed(SEED)


def spreadAoD(mu, std):
    b = std / np.sqrt(2)
    a = np.random.rand() - 0.5
    x = mu - b * np.sign(a) * np.log(1 - 2 * abs(a))
    return x


def generate_H_SV(K, Nt, Ncl, Nray, d_lamda=0.5, beta=1, std=10 / 180 * np.pi):
    Ct = np.arange(Nt)
    H = np.zeros([K, Nt], dtype=complex)
    for k in range(K):
        Htemp = np.zeros([1, Nt], dtype=complex)
        for ii in range(Ncl):
            fhi_i = np.random.uniform(0, 2 * np.pi)
            for jj in range(Nray):
                a = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
                fhi_ij = spreadAoD(fhi_i, std)
                ft = 1 / np.sqrt(Nt) * np.exp(Ct * 1j * 2 * np.pi * d_lamda * np.sin(fhi_ij))
                Htemp = Htemp + a * ft
        H[k] = Htemp
    H = H * np.sqrt(Nt / Ncl / Nray)
    return H


def spreadAoD_vectorized(mu, std, size):
    b = std / np.sqrt(2)
    a = np.random.rand(*size) - 0.5
    return mu - b * np.sign(a) * np.log(1 - 2 * np.abs(a))


def generate_H_SV_vectorized(K, Nt, Ncl, Nray, d_lamda=0.5, beta=1, std=10 / 180 * np.pi):
    Ct = np.arange(Nt)  # (Nt,)

    # Step 1: cluster中心角度 AoD_i ∈ [0, 2π) → shape: (K, Ncl)
    fhi_i = np.random.uniform(0, 2 * np.pi, size=(K, Ncl))  # (K, Ncl)

    # Step 2: 每个cluster中心角度扩展为Nray条 → shape: (K, Ncl, Nray)
    fhi_ij = spreadAoD_vectorized(fhi_i[..., np.newaxis], std, size=(K, Ncl, Nray))  # (K, Ncl, Nray)

    # Step 3: 复高斯系数 a → shape: (K, Ncl, Nray)
    a = (np.random.randn(K, Ncl, Nray) + 1j * np.random.randn(K, Ncl, Nray)) / np.sqrt(2)

    # Step 4: 阵列响应向量 ft → 需要 shape (K, Ncl, Nray, Nt)
    # 角度 shape: (K, Ncl, Nray, 1)，Ct shape: (1, 1, 1, Nt)
    sin_fhi = np.sin(fhi_ij)[..., np.newaxis]  # (K, Ncl, Nray, 1)
    Ct_exp = Ct[np.newaxis, np.newaxis, np.newaxis, :]  # (1, 1, 1, Nt)
    exp_arg = 1j * 2 * np.pi * d_lamda * Ct_exp * sin_fhi
    ft = np.exp(exp_arg) / np.sqrt(Nt)  # (K, Ncl, Nray, Nt)

    # Step 5: a * ft → 需要广播 a[..., np.newaxis]
    contrib = a[..., np.newaxis] * ft  # shape: (K, Ncl, Nray, Nt)

    # Step 6: 聚合所有 path
    H = np.sum(contrib, axis=(1, 2))  # sum over Ncl and Nray → shape: (K, Nt)

    # Step 7: normalize
    H = H * np.sqrt(Nt / (Ncl * Nray))

    return H


def gen_pathloss(n_pico, nUE_pico, nTX_pico, R_pico, shadow_db, alpha=22.7, beta=36.7, h_p=5.):
    theta_pBS = np.linspace(0, n_pico - 1, n_pico) / n_pico * 2 * np.pi  # [n_pico]
    # posi_pico = np.reshape((np.cos(theta_pBS) + 1j * np.sin(theta_pBS)) * (R_pico/np.sqrt(2)), [-1, 1]) # 距离更远
    # posi_pico = np.reshape((np.cos(theta_pBS) + 1j * np.sin(theta_pBS)) * R_pico/np.sqrt(2), [-1, 1])
    if n_pico == 1:
        R = R_pico
    else:
        R = R_pico / np.sin(np.pi / n_pico)  # 小区两两相切 !!
    posi_pico = np.reshape((np.cos(theta_pBS) + 1j * np.sin(theta_pBS)) * R, [-1, 1])  # 小区两两相切 !!

    dis_pico = np.random.rand(n_pico, nUE_pico) * (R_pico - h_p) + h_p  # [n_pico, nUE_pico]
    theta_pico = np.random.rand(n_pico, nUE_pico) * 2 * np.pi  # [n_pico, nUE_pico]
    posi_pUE = (np.cos(theta_pico) + 1j * np.sin(theta_pico)) * dis_pico + posi_pico  # [n_pico, nUE_pico]

    posi_allUE = np.reshape(posi_pUE, [1, -1])  # [1, n_pico * nUE_pico]
    posi_allBS = np.reshape(posi_pico * np.ones([1, nTX_pico]), [-1, 1])  # [n_pico * nTX_pico, 1]
    dist = abs(posi_allBS - posi_allUE)  # [n_pico * nTX_pico, n_pico * nUE_pico]

    alpha0 = np.ones([n_pico*nTX_pico, 1])*alpha
    beta0 = np.ones([n_pico*nTX_pico, 1])*beta
    alpha_dB = -1.0 * (alpha0 + beta0 * np.log10(dist) + 26 * np.log10(2))  # [n_pico*nTX_pico, n_pico*nUE_pico]
    # Diagonal elements are the channels of each cell

    path_loss = np.power(10.0, alpha_dB / 10.0)

    # log_shadowing = np.random.normal(0, shadow_db, size=np.shape(path_loss))
    # shadowing = np.power(10.0, -log_shadowing / 10.0)

    # large_scale = np.multiply(path_loss, shadowing)
    large_scale_h = np.copy(path_loss)  # [n_pico*nTX_pico, n_pico*nUE_pico]
    large_scale = np.copy(large_scale_h)

    for k in range(n_pico):
        large_scale[k * nTX_pico: (k + 1) * nTX_pico, k * nUE_pico: (k + 1) * nUE_pico] = np.ones(
            (nTX_pico, nUE_pico)) * -1000.

    return dist, large_scale_h, large_scale


def gen_channel(nUE_BS, nTX_BS, n_pico, nUE_pico, nTX_pico, R_pico, shadow_db, channel, h_p):
    nUE = sum(nUE_BS)
    nTX = sum(nTX_BS)

    Distance, Large_scale_h, Large_scale = gen_pathloss(n_pico, nUE_pico, nTX_pico, R_pico, shadow_db, h_p)
    if channel == 'SV':
        Hmat = np.zeros([n_pico*nTX_pico, n_pico*nUE_pico], dtype=complex)
        for c in range(n_pico):
            for m in range(n_pico):
                H_SV = generate_H_SV_vectorized(nUE_pico, nTX_pico, Ncl=8, Nray=10, d_lamda=0.5, beta=1, std=10 / 180 * np.pi)
                H_SV = np.transpose(H_SV, [1, 0])
                Hmat[c*nTX_pico:(c+1)*nTX_pico, m*nUE_pico:(m+1)*nUE_pico] = H_SV
    elif channel == 'Ray':
        Hmat = 1 / np.sqrt(2) * (np.random.randn(n_pico*nTX_pico, n_pico*nUE_pico) +
                                 1j * np.random.randn(n_pico*nTX_pico, n_pico*nUE_pico))
    else:
        Hmat = None

    Hmat = Hmat * np.sqrt(Large_scale_h)

    Hmat_cell = np.zeros((n_pico, nTX_pico, nUE_pico), dtype=complex)
    for c in range(n_pico):
        Hmat_cell[c, :, :] = Hmat[c * nTX_pico: (c + 1) * nTX_pico, c * nUE_pico: (c + 1) * nUE_pico]

    Hmat = np.reshape(Hmat, [n_pico, nTX_pico, n_pico, nUE_pico])
    return Distance, Large_scale_h, Large_scale, Hmat, Hmat_cell


def gen_sample(Num, nUE_BS, nTX_BS, n_pico, nUE_pico, nTX_pico, R_pico, shadow_db, h_p):
    Distance_all = np.zeros([Num, n_pico*nTX_pico, n_pico*nUE_pico], dtype=np.float32)
    Large_scale_all = np.zeros([Num, n_pico*nTX_pico, n_pico*nUE_pico, 2], dtype=np.float32)
    Large_scale_h_all = np.zeros([Num, n_pico*nTX_pico, n_pico*nUE_pico, 2], dtype=np.float32)
    H_all = np.zeros([Num, n_pico, nTX_pico, n_pico, nUE_pico, 2], dtype=np.float32)
    H_cell = np.zeros([Num, n_pico, nTX_pico, nUE_pico, 2], dtype=np.float32)

    start_time = time.time()

    for i in range(Num):
        # np.random.seed(SEED + i)
        # random.seed(SEED + i)
        Distance, Large_scale_h, Large_scale, H, H_c = gen_channel(nUE_BS, nTX_BS, n_pico, nUE_pico, nTX_pico, R_pico,
                                                                   shadow_db, channel, h_p)
        Distance_all[i, :, :] = Distance
        Large_scale_all[i, :, :, 0] = Large_scale
        Large_scale_all[i, :, :, 1] = Large_scale
        Large_scale_h_all[i, :, :, 0] = Large_scale_h
        Large_scale_h_all[i, :, :, 1] = Large_scale_h
        H_all[i, :, :, :, :, 0] = np.real(H)
        H_all[i, :, :, :, :, 1] = np.imag(H)
        H_cell[i, :, :, :, 0] = np.real(H_c)
        H_cell[i, :, :, :, 1] = np.imag(H_c)

        if i % 10000 == 0:
            print('Generate' + str(i) + 'samples', 'total time:', time.time() - start_time)

    return Distance_all, Large_scale_h_all, Large_scale_all, H_all, H_cell


n_marco = 0              # number of marco BSs
n_pico = 10                # number of pico BSs
nUE_marco = 0          # number of UEs associate to marco BSs
nUE_pico = 5            # number of UEs associate to pico BSs
nTX_marco = 0          # number of antennas of marco BSs
nTX_pico = 16            # number of antennas of pico BSs
R_pico = 120
N_TRAIN = 300000
N_TEST = 1000
shadow_db = 8
channel = 'SV'  # 'Ray'
distance_min = 60

nUE_BS = np.int32(np.concatenate([np.ones(n_pico) * nUE_pico, np.ones(n_marco) * nUE_marco]))
nTX_BS = np.int32(np.concatenate([np.ones(n_pico) * nTX_pico, np.ones(n_marco) * nTX_marco]))

Distance_all_train, Large_scale_h_train, Large_scale_train, H_all_train, H_cell_train = gen_sample(N_TRAIN, nUE_BS,
                                                                                                   nTX_BS, n_pico,
                                                                                                   nUE_pico, nTX_pico,
                                                                                                   R_pico, shadow_db,
                                                                                                   distance_min)
Distance_all_test, Large_scale_h_test, Large_scale_test, H_all_test, H_cell_test = gen_sample(N_TEST, nUE_BS, nTX_BS,
                                                                                              n_pico, nUE_pico, nTX_pico,
                                                                                              R_pico, shadow_db,
                                                                                              distance_min)

num_users_train = np.ones((N_TRAIN, n_pico, 1)) * nUE_pico
num_users_test = np.ones((N_TEST, n_pico, 1)) * nUE_pico

current_file_path = os.path.abspath(__file__)
parent_path = os.path.dirname(os.path.dirname(current_file_path))
save_path = os.path.join(parent_path, "data", "Cell" + str(n_pico) + "_TX" + str(nTX_pico) +
                         "_UE" + str(nUE_pico) + ".mat")

print(save_path)

with h5py.File(save_path, 'w') as f:
    f.create_dataset('Large_scale_train', data=Large_scale_train)
    f.create_dataset('Large_scale_h_train', data=Large_scale_h_train)
    f.create_dataset('H_all_train', data=H_all_train)
    f.create_dataset('H_cell_train', data=H_cell_train)
    f.create_dataset('Large_scale_test', data=Large_scale_test)
    f.create_dataset('Large_scale_h_test', data=Large_scale_h_test)
    f.create_dataset('H_all_test', data=H_all_test)
    f.create_dataset('H_cell_test', data=H_cell_test)
    f.create_dataset('num_users_train', data=num_users_train)
    f.create_dataset('num_users_test', data=num_users_test)
    f.create_dataset('Distance_all_train', data=Distance_all_train)
    f.create_dataset('Distance_all_test', data=Distance_all_test)
print('generated')

