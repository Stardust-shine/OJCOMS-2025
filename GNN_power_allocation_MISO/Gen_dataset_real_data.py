import numpy as np
from powerallocation_wmmse import gen_sample_real, gen_pathloss
import json
import scipy.io as sio

Configfile = open('Config_real.json', 'r', encoding='utf8')
config = json.load(Configfile)
APPEND_MOD = False

N_TRAIN = 8000
N_TEST = 1000
model_location = config['model_location']

n_marco = config['n_marco']
n_pico = config['n_pico']
nUE_marco = config['nUE_marco']
nUE_pico = config['nUE_pico']
nTX_marco = config['nTX_marco']
nTX_pico = config['nTX_pico']
p_marco = config['p_marco']
p_pico = config['p_pico']
var_noise = config['var_noise']

n_hidden = config['n_hidden']
n_output = config['n_output']

p_BS = np.concatenate([np.ones(n_pico) * p_pico, np.ones(n_marco) * p_marco])
nUE_BS = np.int64(np.concatenate([np.ones(n_pico) * nUE_pico, np.ones(n_marco) * nUE_marco]))
nTX_BS = np.int64(np.concatenate([np.ones(n_pico) * nTX_pico, np.ones(n_marco) * nTX_marco]))

file_real = 'Dataset_real/Channel_' + str(n_marco) + 'MBS_' + str(nUE_marco) + 'UE_' + \
                                   str(n_pico) + 'PBS_' + str(nUE_pico) + 'UE.mat'

H_file = sio.loadmat(file_real)
Hmat = H_file['H']

Hmat_train = Hmat[0:N_TRAIN, :, :]
Hmat_test = Hmat[N_TRAIN:N_TRAIN + N_TEST, :, :]

A_BS_UE, X_BS, X_UE, Y_BS, SR, SR_UE, X_D = gen_sample_real(Hmat_train, N_TRAIN, nUE_BS, nTX_BS, p_BS, var_noise, config)
A_BS_UE_test, X_BS_test, X_UE_test, Y_BS_test, SR_test, SR_UE_test, X_D_test \
    = gen_sample_real(Hmat_test, N_TEST, nUE_BS, nTX_BS, p_BS, var_noise, config)

trainfilename = 'Dataset_new/Train_' + str(n_marco) + 'MBS_' + str(nUE_marco) + 'UE_' + \
                                   str(n_pico) + 'PBS_' + str(nUE_pico) + 'UE_real.mat'
testfilename = 'Dataset_new/Test_' + str(n_marco) + 'MBS_' + str(nUE_marco) + 'UE_' + \
                                   str(n_pico) + 'PBS_' + str(nUE_pico) + 'UE_real.mat'

if APPEND_MOD is True:
    trfile = sio.loadmat(trainfilename); tefile = sio.loadmat(testfilename)

    A_BS_UEo = trfile['A_BS_UE']; X_BSo = trfile['X_BS']; X_UEo = trfile['X_UE'];
    Y_BSo = trfile['Y_BS']; SRo = trfile['SR']; X_Do = trfile['X_D']

    A_BS_UE_testo = tefile['A_BS_UE']; X_BS_testo = tefile['X_BS']; X_UE_testo = tefile['X_UE'];
    Y_BS_testo = tefile['Y_BS']; SR_testo = tefile['SR']; X_D_testo = tefile['X_D']

    A_BS_UE = np.concatenate((A_BS_UEo, A_BS_UE), axis=0)
    X_BS = np.concatenate((X_BSo, X_BS), axis=0)
    X_UE = np.concatenate((X_UEo, X_UE), axis=0)
    Y_BS = np.concatenate((Y_BSo, Y_BS), axis=0)
    SR = np.concatenate((SRo[0], SR), axis=0)
    X_D = np.concatenate((X_Do, X_D), axis=0)

    A_BS_UE_test = np.concatenate((A_BS_UE_testo, A_BS_UE_test), axis=0)
    X_BS_test = np.concatenate((X_BS_testo, X_BS_test), axis=0)
    X_UE_test = np.concatenate((X_UE_testo, X_UE_test), axis=0)
    Y_BS_test = np.concatenate((Y_BS_testo, Y_BS_test), axis=0)
    SR_test = np.concatenate((SR_testo[0], SR_test), axis=0)
    X_D_test = np.concatenate((X_D_testo, X_D_test), axis=0)

sio.savemat(trainfilename, {'A_BS_UE': A_BS_UE, 'X_BS': X_BS, 'X_UE': X_UE,
                            'Y_BS': Y_BS, 'SR': SR, 'SR_UE': SR_UE, 'X_D': X_D})
sio.savemat(testfilename, {'A_BS_UE': A_BS_UE_test, 'X_BS': X_BS_test, 'X_UE': X_UE_test,
                           'Y_BS': Y_BS_test, 'SR': SR_test, 'SR_UE': SR_UE_test, 'X_D': X_D_test})