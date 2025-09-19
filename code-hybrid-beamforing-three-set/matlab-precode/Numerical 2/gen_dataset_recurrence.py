# coding=UTF-8
import numpy as np
import sys
import time
import scipy.io as sio


def generate_H_rayleigh(K, N):
    H = 1 / np.sqrt(2) * (np.random.randn(K, N).astype(np.float32) + 1j * np.random.randn(K, N).astype(np.float32))
    return H


def spreadAoD(mu, std):
    b=std/np.sqrt(2)
    a=np.random.rand()-0.5
    x=mu-b*np.sign(a)*np.log(1-2*abs(a))
    return x


def generate_H_SV(K, N_r, Nt, L, d_lamda=0.5, beta=1, std=10/180*np.pi):
    Ct = np.arange(Nt)
    Dt = np.arange(N_r)
    H = np.zeros([K*N_r, Nt], dtype=complex)
    TT = np.zeros([K*L, N_r], dtype=complex)
    D = np.zeros([K, L], dtype=complex)

    for k in range(K):
        Htemp = np.zeros([N_r, Nt],dtype=complex)
        for ii in range(L):
            fhi_i = np.random.uniform(0,2*np.pi)
            theta_i = np.random.uniform(0,2*np.pi)
            a = (np.random.randn() + 1j * np.random.randn())
            ft = np.reshape(1 / np.sqrt(Nt) * np.exp(Ct * 1j * 2 * np.pi * d_lamda * np.cos(fhi_i)), (1, Nt))
            tt = np.reshape(1 / np.sqrt(N_r) * np.exp(Dt * 1j * 2 * np.pi * d_lamda * np.cos(theta_i)), (1, N_r))
            Htemp = Htemp + a * np.matmul(np.transpose(tt), ft.conj())

            TT[(k-1)*L + ii, :] = np.reshape(np.transpose(tt), (N_r))
            D[k, ii] = a
        H[k*N_r:(k+1)*N_r, :] = Htemp
    H = H * np.sqrt(N_r*Nt)
    return H, TT, D


def generate_H_dataset(number, K, N_r, N, L):
    start = time.time()
    H = np.zeros([number, 2, K*N_r, N],dtype=np.float32)
    TT_all = np.zeros([number, K*L, N_r],dtype=complex)
    D_all = np.zeros([number, K, L],dtype=complex)

    for i in range(number):
        if i % 2000 == 0:
            end = time.time()
            print(i,end-start)
            sys.stdout.flush()
            start = time.time()

        Htemp, TT, D = generate_H_SV(K, N_r, N, L)
        H[i,0,:,:] = np.real(Htemp)
        H[i,1,:,:] = np.imag(Htemp)

        TT_all[i, :, :] = TT
        D_all[i, :, :] = D

    H = np.transpose(H, axes=(0, 2, 3, 1))
    H = H[:, :, :, 0] + np.complex('j') * H[:, :, :, 1]

    # np.save("./data/setH_K" + str(K) + "_N" + str(N) + "_Ncl" + str(Ncl) + "_Nray" + str(Nray) + "_number" + str(number) + ".npy", H)
    sio.savemat("./data/test-"  + str(K) + ".mat", {'H_com': H, 'TT': TT_all, 'D': D_all})
    print('generated')
    return H


if __name__ == '__main__':
    K = 18
    N_r = 16
    Nt = 128
    L = 2
    # need a 'data' folder
    generate_H_dataset(1000, K, N_r, Nt, L)
    # generate_H_dataset(10000, K, N_r, Nt, L)
