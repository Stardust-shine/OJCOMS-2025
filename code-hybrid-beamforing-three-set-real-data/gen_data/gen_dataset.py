# coding=UTF-8
import numpy as np
import sys
import time
import scipy.io as sio
import os
import h5py


def generate_H_rayleigh(K, N):
    H = 1 / np.sqrt(2) * (np.random.randn(K, N).astype(np.float32) + 1j * np.random.randn(K, N).astype(np.float32))
    return H


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


def generate_dft(N):
    Ncb = N
    Nt_set = np.arange(N)
    angles = np.linspace(0, 2 * np.pi, Ncb + 1)
    angles = angles[:-1]
    dft_codebook_mtx = np.exp(-1j * np.outer(Nt_set, angles))
    dft_codebook_mtx = dft_codebook_mtx / np.sqrt(Nt)  # [N, Ncb]

    DFT = np.zeros([1, 2, N, Ncb])
    DFT[0, 0, :, :] = np.real(dft_codebook_mtx)
    DFT[0, 1, :, :] = np.imag(dft_codebook_mtx)

    return DFT


def generate_transform(N):
    row_indices = np.arange(N).reshape(N, 1)  # 形状: (N, 1)
    col_indices = np.arange(N).reshape(1, N)  # 形状: (1, N)

    A = row_indices * col_indices  # 广播乘法，结果形状: (N, N)

    transform = np.exp(-1j * A * 2 * np.pi/N) / np.sqrt(N)

    T = np.zeros([1, N, N, 2])

    T[0, :, :, 0] = np.real(transform)
    T[0, :, :, 1] = np.imag(transform)

    return T


def generate_H_dataset(number, K, N, Ncl, Nray, item):
    start = time.time()
    H = np.zeros([number, 2, K, N], dtype=np.float32)
    shade = np.zeros([number, K, 1], dtype=np.float32)
    num_users = np.zeros([number, 1], dtype=np.float32)
    # dft = generate_dft(N)

    for i in range(number):
        if i % 2000 == 0:
            end = time.time()
            print(i, end - start)
            sys.stdout.flush()
            start = time.time()
        if Ncl > 0:
            Htemp = generate_H_SV(K, N, Ncl, Nray)
        else:
            Htemp = generate_H_rayleigh(K, N)

        H[i, 0, :, :] = np.real(Htemp)
        H[i, 1, :, :] = np.imag(Htemp)

        shade[i, :, :] = 1
        num_users[i, :] = K

    current_file_path = os.path.abspath(__file__)
    parent_path = os.path.dirname(os.path.dirname(current_file_path))
    save_path = os.path.join(parent_path, "data", "setH_K" + str(K) + "_N" + str(N) +
                             "_Ncl" + str(Ncl) + "_Nray" + str(Nray) + item + ".mat")
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('H', data=H)
        f.create_dataset('shade', data=shade)
        f.create_dataset('num_users', data=num_users)
    print('generated')
    return H


def generate_gen_H_dataset(number, K_min, K_max, N, Ncl, Nray, item='_train'):
    start = time.time()
    H = np.zeros([number, 2, K_max, N], dtype=np.float32)
    shade = np.zeros([number, K_max, 1], dtype=np.float32)
    num_users = np.zeros([number, 1], dtype=np.float32)

    for i in range(number):
        if i % 2000 == 0:
            end = time.time()
            print(i, end - start)
            sys.stdout.flush()
            start = time.time()
        if item == '_train':
            nUE = np.random.randint(K_min, K_max + 1)
        elif item == '_test':
            nUE = np.random.randint(K_min, K_max + 1)
        # nUE = max(min(int(1 + np.random.exponential(2)), K_max), 2)
        if Ncl > 0:
            Htemp = generate_H_SV(nUE, N, Ncl, Nray)
        else:
            Htemp = generate_H_rayleigh(nUE, N)

        H[i, 0, 0:nUE, 0:N] = np.real(Htemp)
        H[i, 1, 0:nUE, 0:N] = np.imag(Htemp)
        num_users[i, :] = nUE

        shade[i, 0:nUE, :] = 1

    current_file_path = os.path.abspath(__file__)
    parent_path = os.path.dirname(os.path.dirname(current_file_path))
    save_path = os.path.join(parent_path, "data", "genH_K" + str(K_max) + "_N" + str(N) +
                             "_Ncl" + str(Ncl) + "_Nray" + str(Nray) + item + ".mat")
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('H', data=H)
        f.create_dataset('shade', data=shade)
        f.create_dataset('num_users', data=num_users)
    print('generated')
    return H


if __name__ == '__main__':
    # Gen data for Gen_train
    # K_min = 2
    # K_max_test = 16
    # K_max_train = 10
    # Nt = 64
    #
    #
    # Ncl = 8
    # Nray = 10
    #
    # generate_gen_H_dataset(50000, K_min, K_max_train, Nt, Ncl=Ncl, Nray=Nray, item='_train')
    # generate_gen_H_dataset(2000, K_min, K_max_test, Nt, Ncl=Ncl, Nray=Nray, item='_test')

    # Gen data for train
    K = 2
    Nt = 8

    Ncl = 8
    Nray = 10
    generate_H_dataset(50000, K, Nt, Ncl=Ncl, Nray=Nray, item='_train')
    generate_H_dataset(2000, K, Nt, Ncl=Ncl, Nray=Nray, item='_test')
