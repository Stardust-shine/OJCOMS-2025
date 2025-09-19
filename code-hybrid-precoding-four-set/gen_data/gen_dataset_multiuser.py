# coding=UTF-8
import numpy as np
import sys
import time
import scipy.io as sio
import os


def generate_H_rayleigh(K, N):
    H = 1 / np.sqrt(2) * (np.random.randn(K, N).astype(np.float32) + 1j * np.random.randn(K, N).astype(np.float32))
    return H


def spreadAoD(mu, std):
    b=std/np.sqrt(2)
    a=np.random.rand()-0.5
    x=mu-b*np.sign(a)*np.log(1-2*abs(a))
    return x


def generate_H_SV(K, N_r, Nt, Ncl, Nray, d_lamda=0.5, beta=1, std=10/180*np.pi):
    Ct = np.arange(Nt)
    Dt = np.arange(N_r)
    H = np.zeros([K*N_r, Nt], dtype=complex)
    for k in range(K):
        Htemp = np.zeros([N_r, Nt],dtype=complex)
        for ii in range(Ncl):
            fhi_i = np.random.uniform(0,2*np.pi)
            theta_i = np.random.uniform(0,2*np.pi)
            for jj in range(Nray):
                a = (np.random.randn()+1j*np.random.randn())/np.sqrt(2)
                fhi_ij = spreadAoD(fhi_i, std)
                theta_ij = spreadAoD(theta_i, std)
                ft = np.reshape(1 / np.sqrt(Nt) * np.exp(Ct * 1j * 2 * np.pi * d_lamda * np.sin(fhi_ij)), (1, Nt))
                tt = np.reshape(1 / np.sqrt(N_r) * np.exp(Dt * 1j * 2 * np.pi * d_lamda * np.sin(theta_ij)), (1, N_r))
                Htemp = Htemp+ a*np.matmul(np.transpose(tt), ft.conj())
        H[k*N_r:(k+1)*N_r, :] = Htemp
    H = H * np.sqrt(N_r*Nt/Ncl/Nray)
    return H


def generate_H_dataset(number, K, N_r, N, Ncl, Nray):
    start = time.time()
    H = np.zeros([number, 2, K*N_r, N],dtype=np.float32)

    for i in range(number):
        if i % 2000 == 0:
            end = time.time()
            print(i, end - start)
            sys.stdout.flush()
            start = time.time()
        if Ncl > 0:
            Htemp = generate_H_SV(K, N_r, N, Ncl, Nray)
        else:
            Htemp = generate_H_rayleigh(K, N)

        H[i, 0, :, :] = np.real(Htemp)
        H[i, 1, :, :] = np.imag(Htemp)

    current_file_path = os.path.abspath(__file__)
    parent_path = os.path.dirname(os.path.dirname(current_file_path))

    # np.save("./data/setH_K" + str(K) + "_N" + str(N) + "_Ncl" + str(Ncl) + "_Nray" + str(Nray) +
    # "_number" + str(number) + ".npy", H)
    sio.savemat(parent_path + "/data/setH_K" + str(K) + "_N" + str(N_r) + "X"+ str(N) + "_Ncl" + str(Ncl) + "_Nray" + str(Nray) + "_number" + str(500000) + ".mat",
                {'H': H})
    print('generated')
    return H


if __name__ == '__main__':
    K = 2
    N_r = 2
    Nt = 256

    Ncl = 8
    Nray = 10
    # need a 'data' folder
    generate_H_dataset(100000, K, N_r, Nt, Ncl=Ncl, Nray=Nray)
    generate_H_dataset(10000, K, N_r, Nt, Ncl=Ncl, Nray=Nray)
