import numpy as np
import math
import scipy.io as sio
import time


def ZF_precoding(H):
    HT = (H.conj()).T
    W = np.matmul(HT, np.linalg.inv(np.matmul(H, HT)))  # 求伪逆
    W = W / np.sqrt(np.sum(np.abs(W) ** 2))
    H_abs_ZF = np.diagonal(np.matmul(H, W))

    return W


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

    return V


def WMMSE_1BS1UETX(H, Pmax, var_noise):
    nUE = H.shape[0]
    nTX = H.shape[1]

    V = np.ones([nTX, nUE], dtype=complex) * np.sqrt(Pmax / (nTX * nUE))
    W = np.zeros(nUE, dtype=complex)

    for iter in range(100):
        W_old = np.copy(W)

        H_tmp = np.expand_dims(H, axis=1)
        V_tmp = np.expand_dims(V, axis=0)

        I = np.sum(np.power(np.abs(np.matmul(H_tmp, V_tmp)), 2), axis=2)
        V_tmp = np.expand_dims(np.transpose(V), axis=2)
        U = np.matmul(H_tmp, V_tmp)[:, 0, 0] / (I[:, 0] + var_noise)

        U_tmp = np.expand_dims(np.expand_dims(U, axis=1), axis=1)
        W = 1 / (1 - np.matmul(np.matmul(U_tmp.conj(), H_tmp), V_tmp))

        UHH = np.matmul(U_tmp.conj(), H_tmp)
        W_tmp = np.reshape(W, [-1, 1, 1])
        I = np.sum(np.matmul(np.matmul(np.transpose(UHH.conj(), [0, 2, 1]), W_tmp), UHH), axis=0)
        Ik = np.sum(np.matmul(np.matmul(np.transpose(UHH.conj(), [0, 2, 1]), W_tmp ** 2), UHH), axis=0)
        I = I + get_mu(Pmax, I, Ik) * np.eye(nTX)
        V = np.matmul(np.matmul(np.expand_dims(np.linalg.inv(I), axis=0), np.transpose(UHH.conj(), [0, 2, 1])), W_tmp)
        V = np.transpose(V[:, :, 0])

        if iter >= 1 and abs(np.sum(np.log(W_old)) - np.sum(np.log(W))) < 1e-4:
            break

    return V


def EE_max_Precoding(H, Pmax, var_noise, rau, rmin, P0, Pc):
    nUE = H.shape[0]
    nTX = H.shape[1]
    flag = 0
    # Initialize
    HHh = np.matmul(H, np.conj(np.transpose(H)))
    HHh_inv = np.linalg.inv(HHh)
    V0 = np.matmul(np.conj(np.transpose(H)), HHh_inv)
    V0 = V0 / np.sqrt(np.sum(np.power(np.abs(V0), 2)))
    G = np.sum(np.eye(nUE) * np.matmul(H, V0), axis=0)
    Pini = power_opt(nUE, Pmax / nUE, Pc, rmin, G, rau, var_noise)
    V = np.matmul(V0, np.diag(np.sqrt(Pini)))

    # V = np.ones([nTX, nUE], dtype=complex) * np.sqrt(Pmax / (nTX * nUE))
    W = np.zeros(nUE, dtype=complex)
    EE_old = 2022

    # Update
    for iter in range(100):
        W_old = np.copy(W)

        H_tmp = np.expand_dims(H, axis=1)
        V_tmp = np.expand_dims(V, axis=0)

        I = np.sum(np.power(np.abs(np.matmul(H_tmp, V_tmp)), 2), axis=2)
        V_tmp = np.expand_dims(np.transpose(V), axis=2)
        U = np.matmul(H_tmp, V_tmp)[:, 0, 0] / (I[:, 0] + var_noise)

        U_tmp = np.expand_dims(np.expand_dims(U, axis=1), axis=1)
        W = 1 / (1 - np.matmul(np.matmul(U_tmp.conj(), H_tmp), V_tmp))
        EE, _ = cal_ee_precoding(H, V, P0, Pc, rau, nTX, var_noise)
        V, alpha = bisearch(H, U, W[:, 0, 0], rau, var_noise, Pc, P0, rmin, nTX, nUE, EE)
        if V is np.nan:
            flag = 1
            break
        V = np.transpose(V)
        EE_new, _ = cal_ee_precoding(H, V, P0, Pc, rau, nTX, var_noise)
        # print('EE is '+ str(EE_new))
        if iter >= 1 and abs(np.sum(np.log(W) - np.log(W_old))) < 5e-3:
            break
        EE_old = EE_new

    return V, flag


def bisearch(H, U, W, rau, sigma2, Pc, P0, r0, Ntx, K, EE0):
    low = 0.0;
    high = 2 * EE0
    while (1):
        alpha = (low + high) / 2
        V_real, V_imag = check_feasibility(alpha, H, U, W, rau, sigma2, Pc, P0, r0, Ntx, K)
        if V_real is None:
            high = alpha
        else:
            low = alpha
            V_new_re = V_real;
            V_new_im = V_imag
        if abs(high - low) < 1e-2:
            break

    V = V_new_re + 1j * V_new_im

    try:
        return V, alpha
    except UnboundLocalError:
        V = np.nan;
        alpha = np.nan
        return V, alpha


def check_feasibility(alpha, H, U, W, rau, sigma2, Pc, P0, r0, Ntx, K):
    import cvxpy as cp
    H_real = np.real(H);
    H_imag = np.imag(H);
    U_real = np.real(U);
    U_imag = np.imag(U);
    W = np.real(W)
    V_real = cp.Variable((K, Ntx));
    V_imag = cp.Variable((K, Ntx))
    # V_real = np.transpose(np.real(V0)); V_imag = np.transpose(np.imag(V0))
    HV_real, HV_imag = cplx_multiply(H_real, H_imag, V_real, V_imag)
    HV_real = cp.sum(HV_real, axis=1);
    HV_imag = cp.sum(HV_imag, axis=1)
    UhHV_real, UhHV_imag = cplx_multiply(U_real, cp.Constant(-1) * U_imag, HV_real, HV_imag)

    G_real, G_imag = cplx_matmul(H_real, H_imag, cp.transpose(V_real), cp.transpose(V_imag))
    U1_real = cp.reshape(U_real, [K, 1]);
    U1_imag = cp.reshape(U_imag, [K, 1])
    I_real, I_imag = cplx_multiply(U1_real, -U1_imag, G_real, G_imag)

    E1_real = cp.power(1 - UhHV_real, 2) + cp.power(UhHV_imag, 2)
    E2_real = cp.power(I_real, 2) + cp.power(I_imag, 2)
    E3_real = cp.sum(cp.multiply(1 - np.eye(K), E2_real), axis=1)
    E4_real = sigma2 * (cp.power(U_real, 2) + cp.power(U_imag, 2))
    E = E1_real + E3_real + E4_real

    f0 = np.log2(np.e) * (cp.log(W) - cp.multiply(W, E) + 1.0)
    f1 = rau * (cp.sum(cp.power(V_real, 2) + cp.power(V_imag, 2))) + Ntx * Pc + P0

    objective = cp.Minimize(0.0)
    constraints = [cp.sum(f0) - alpha * f1 >= 0, f0 >= r0 * np.ones(K)]  # + [f0[i] >= r0 for i in range(K)]
    prob = cp.Problem(objective, constraints)
    try:
        result = prob.solve()
    except cp.error.SolverError:
        result = prob.solve(solver='SCS')
    # result = prob.solve(solver='SCS')
    return V_real.value, V_imag.value


def cplx_multiply(a_real, a_imag, b_real, b_imag):
    import cvxpy as cp
    ab_real = cp.multiply(a_real, b_real) - cp.multiply(a_imag, b_imag)
    ab_imag = cp.multiply(a_real, b_imag) + cp.multiply(a_imag, b_real)
    return ab_real, ab_imag


def cplx_matmul(a_real, a_imag, b_real, b_imag):
    import cvxpy as cp
    ab_real = cp.matmul(a_real, b_real) - cp.matmul(a_imag, b_imag)
    ab_imag = cp.matmul(a_real, b_imag) + cp.matmul(a_imag, b_real)
    return ab_real, ab_imag


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


def cal_sum_rate_BF_UE1TX(H, V, var_noise):
    if len(np.shape(H)) < 3:
        H = np.expand_dims(H, axis=0)
        V = np.expand_dims(V, axis=0)

    nUE = np.shape(H)[1]
    G = np.power(np.abs(np.matmul(H, V)), 2)
    A = np.reshape(np.eye(nUE), [1, nUE, nUE])
    rate = np.log2(1 + np.sum(G * A, axis=2) / (np.sum(G * (1 - A), axis=2) + var_noise))
    sumrate = np.sum(rate, axis=1)

    return sumrate, rate


def gen_chl_est_err1(Alp_mat, Ttr, ul_SNR):
    SNR = 10 ** (ul_SNR / 10) * Alp_mat * Ttr
    E = Alp_mat / (1.0 + SNR)

    return E


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


def gen_sample(nUE_BS, nTX_BS, nTX_UE, Pmax_BS, var_noise, chl_model='Ray', KdB=0, is_gen_label=True, factor=1.0,
               is_large_scale=False, is_w_chl_err=False, var_esterr=0.0):
    nTX_allBS = sum(nTX_BS)
    nTX_allUE = sum(nUE_BS) * nTX_UE
    Ncell = len(nUE_BS)

    # scalability
    # if nTX_allUE % 2 == 0:
    #     Hmat = 1 / np.sqrt(2) * (np.random.randn(1, 2, nTX_allBS) + 1j * np.random.randn(1, 2, nTX_allBS))
    #     Hmat = Hmat * np.ones([nTX_allUE//2, 1, 1])
    #     Hmat = np.reshape(Hmat, [nTX_allUE, nTX_allBS])
    # else:
    #     Hmat = 1 / np.sqrt(2) * (np.random.randn(nTX_allUE, nTX_allBS) + 1j * np.random.randn(nTX_allUE, nTX_allBS))
    Hmat = 1 / np.sqrt(2) * (
                np.random.randn(nTX_allUE, nTX_allBS) + 1j * np.random.randn(nTX_allUE, nTX_allBS)) * factor
    if chl_model == 'Ric':
        K = 10.0 ** (KdB / 10.0)
        Hmat = np.sqrt(K / (K + 1)) + np.sqrt(1 / (K + 1)) * Hmat
    if chl_model == 'SV':
        Hmat = generate_H_SV(nTX_allUE, nTX_allBS, Ncl=4, Nray=5)

    if is_large_scale is True:
        Alpha, _ = gen_pathloss(1, 0, nTX_allUE, 1, nTX_allBS, 1, is_shadow=True)
    else:
        Alpha = np.ones([nTX_allUE, nTX_allBS], dtype=complex)
    Hmat = Hmat * np.sqrt(Alpha)
    ul_SNR = 10 * np.log10(1 / var_noise) - 5
    if var_esterr > 0.0:
        var_err = gen_chl_est_err1(Alpha, nTX_allUE, ul_SNR)
    else:
        var_err = 0.0
    Hemat = Hmat + (np.random.randn(nTX_allUE, nTX_allBS) + 1j * np.random.randn(nTX_allUE, nTX_allBS)) * np.sqrt(
        var_err) / np.sqrt(2)

    if (nTX_UE == 1) and (len(nUE_BS) == 1):  # 单基站、用户单天线的场景
        if is_gen_label is True:
            start = time.perf_counter()
            Vmat = WMMSE_1BS1UETX(Hmat, Pmax_BS[0], var_noise)
            cost_time = time.perf_counter() - start
            rate = cal_sum_rate_BF_UE1TX(Hmat, Vmat, var_noise)
        else:
            Vmat = np.zeros(Hmat.transpose().shape)
            rate = np.zeros(1)

    elif (nTX_UE == 1) and (len(nUE_BS) > 1):  # 多基站、用户单天线的场景
        Vmat = np.zeros([nTX_allBS, nTX_allUE], dtype=complex)
        if is_gen_label is True:
            Vopt = WMMSE_BF(Hmat, nUE_BS, nTX_BS, nTX_UE, Pmax_BS, var_noise)
            for iBS in range(len(nUE_BS)):
                Vtmp = np.transpose(
                    np.reshape(Vopt[sum(nUE_BS[0:iBS] * nTX_BS[0:iBS]):sum(nUE_BS[0:iBS + 1] * nTX_BS[0:iBS + 1])],
                               [nUE_BS[iBS], nTX_BS[iBS]]))
                Vmat[sum(nTX_BS[0:iBS]):sum(nTX_BS[0:iBS + 1]), sum(nUE_BS[0:iBS]):sum(nUE_BS[0:iBS + 1])] = Vtmp
        rate = cal_sum_rate_BF_UE1TX(Hmat, Vmat, var_noise)

    else:  # 用户多天线场景
        Vopt = WMMSE_BF(Hmat, nUE_BS, nTX_BS, nTX_UE, Pmax_BS, var_noise)
        Vmat = Vopt
        rate = np.nan

    if is_w_chl_err is False:
        return Hmat, Vmat, rate
    else:
        return Hmat, Hemat, Vmat, rate


def gen_sample_maxee(nUE_BS, nTX_BS, nTX_UE, Pmax_BS, var_noise, rau, rmin, P0, Pc, chl_model='Ray',
                     KdB=0, is_gen_label=True, best='EEmax'):
    nTX_allBS = sum(nTX_BS)
    nTX_allUE = sum(nUE_BS) * nTX_UE

    Hmat = 1 / np.sqrt(2) * (np.random.randn(nTX_allUE, nTX_allBS) + 1j * np.random.randn(nTX_allUE, nTX_allBS))

    if chl_model == 'Ric':
        K = 10.0 ** (KdB / 10.0)
        Hmat = np.sqrt(K / (K + 1)) + np.sqrt(1 / (K + 1)) * Hmat

    if is_gen_label is True:
        if best == 'EEmax':
            start = time.perf_counter()
            Vmat, flag = EE_max_Precoding(Hmat, Pmax_BS[0], var_noise, rau, rmin, P0, Pc)
            print(str(time.perf_counter() - start))
        else:
            Vmat = WMMSE_1BS1UETX(Hmat, Pmax_BS[0] * 0.0 + 1.0, var_noise)
            flag = 0
        rate, _ = cal_sum_rate_BF_UE1TX(Hmat, Vmat, var_noise)
        power = rau * np.sum(np.power(np.abs(Vmat), 2)) + nTX_allBS * Pc + P0
        ee = rate / power
    else:
        Vmat = np.zeros(Hmat.transpose().shape);
        flag = 0
        rate = np.zeros(1)
        power = np.zeros(1)
        ee = np.zeros(1)

    return Hmat, Vmat, rate, power, ee, flag


def cal_ratio(pyrate_mat, nnrate_mat, n_spl_oneuser, method='div'):
    pyrate_mat = np.reshape(pyrate_mat, [-1, n_spl_oneuser])
    nnrate_mat = np.reshape(nnrate_mat, [-1, n_spl_oneuser])
    pyrate = np.mean(pyrate_mat, axis=1)
    nnrate = np.mean(nnrate_mat, axis=1)
    if method == 'div':
        ratio = nnrate / pyrate * 100
    elif method == 'subtract':
        ratio = (pyrate - nnrate) * 100
    else:
        print('wrong method!')
    return ratio


def cal_corr(H, V):
    X = H;
    Y = V
    X_Re = X[..., 0];
    X_Im = X[..., 1]
    Y_Re = Y[..., 0];
    Y_Im = Y[..., 1]
    XY_Re = X_Re * Y_Re - X_Im * Y_Im
    XY_Im = X_Re * Y_Im + X_Im * Y_Re
    XY = np.concatenate((np.expand_dims(XY_Re, axis=-1), np.expand_dims(XY_Im, axis=-1)), axis=-1)
    temp = np.sum(XY, axis=2, keepdims=True) / (np.sqrt(np.sum(np.power(X, 2), axis=(2, 3), keepdims=True)) *
                                                np.sqrt(np.sum(np.power(Y, 2), axis=(2, 3), keepdims=True)))
    temp1 = np.sqrt(np.power(temp[..., 0], 2) + np.power(temp[..., 1], 2))
    corr = np.mean(temp1)

    return temp1[..., 0], corr


def cal_approx_pseodo_inv(H, var_noise=1):
    # H 用户数 X 天线数
    K = H.shape[1]
    H = H[:, :, :, 0] + 1j * H[:, :, :, 1]
    hHh = np.matmul(H, np.transpose(H, [0, 2, 1]).conj()) + var_noise * np.reshape(np.eye(K), [1, K, K])
    # approx_inv = np.reshape(np.eye(K), [1, K, K]) * 1 / np.real(hHh) + 1j * 0.0
    approx_inv = np.linalg.inv(hHh)
    approx_pinv = np.matmul(np.transpose(H, [0, 2, 1]).conj(), approx_inv)
    pinv_real = np.real(approx_pinv);
    pinv_imag = np.imag(approx_pinv)
    pinv = np.concatenate((np.expand_dims(pinv_real, axis=3), np.expand_dims(pinv_imag, axis=3)), axis=3)
    pinv = pinv / np.sqrt(np.sum(np.power(pinv, 2), axis=(1, 2, 3), keepdims=True))
    pinv = np.transpose(pinv, [0, 2, 1, 3])

    return pinv


def power_opt(K, Pmax, Pc, R0, H, rho, var_noise):
    P = Pmax
    H = np.abs(H)
    ee_old, _ = cal_ee(H, P, Pc, rho, var_noise)
    for iter in range(100):
        R = np.log2(1.0 + H * H * P / var_noise)
        P = (np.maximum(np.power(2, R0),
                        np.sum(rho * P + Pc) * H * H / var_noise / (rho * np.sum(R) * np.log(2))) - 1) * \
            var_noise / ((H * H) + 1e-6)
        P = np.maximum(np.minimum(P, Pmax), 0)
        ee_new, _ = cal_ee(H, P, Pc, rho, var_noise)
        if abs(ee_new - ee_old) <= 1e-3:
            # print('converge at the ' + str(iter + 1) + ' th iteration!')
            break
        ee_old = ee_new

    return P


def cal_ee(H, P, Pc, rho, var_noise):
    H = np.abs(H)
    R = np.log2(1.0 + H * H * P / var_noise)
    P_all = rho * P + Pc
    EE = np.sum(R) / np.sum(P_all)

    return EE, R


def cal_ee_precoding(H, V, P0, Pc, rau, nTX, var_noise):
    sumrate, rate = cal_sum_rate_BF_UE1TX(H, V, var_noise)
    power = rau * np.sum(np.power(np.abs(V), 2), axis=(-1, -2)) + nTX * Pc + P0
    ee = sumrate / power

    return ee, rate


def gen_adj_mat(Ncell, N, K):
    temp = np.reshape(np.eye(Ncell), [Ncell, Ncell, 1]) * np.ones([1, 1, N])
    Adj1 = np.reshape(temp, [Ncell, Ncell * N])
    temp = np.reshape(Adj1, [Ncell, 1, Ncell * N]) * np.ones([1, K, 1])
    Adj = np.reshape(temp, [Ncell * K, Ncell * N])

    return Adj


shadow_map = sio.loadmat('Dataset/ShadowMap20Cells.mat')['SHADOW_MAP']


def gen_pathloss(n_marco, n_pico, nUE_marco, nUE_pico, nTX_marco, nTX_pico, R_marco=240.0, R_pico=50.0,
                 alpha_mBS=13.54, beta_mBS=39.08, alpha_pBS=32.4, beta_pBS=31.9, h_p=10.0, h_m=30.0, is_shadow=False,
                 d_region=50):
    R_marco = R_pico;
    alpha_mBS = alpha_pBS;
    beta_mBS = beta_pBS

    # D_marco = 2 * R_marco; D_pico = 2 * R_pico
    # posi_marco = np.reshape(np.linspace(0, (n_marco - 1) * D_marco, n_marco), [-1, 1])
    # # posi_pico = np.reshape(np.linspace(0, (n_pico - 1) * D_pico, n_pico) + 1j * 120.0, [-1, 1])
    # theta_pBS = np.linspace(0, n_pico - 1, n_pico) / n_pico * 2 * np.pi
    # posi_pico = np.reshape((np.cos(theta_pBS) + 1j * np.sin(theta_pBS)) * 2 * D_pico, [-1, 1])

    posi_bs = gen_ppp_posi(5e-5, d_region, d_region, R_pico)
    posi_marco = np.reshape(posi_bs[0:n_marco], [-1, 1])
    posi_pico = np.reshape(posi_bs[n_marco:n_marco + n_pico], [-1, 1])

    dis_pico = np.random.rand(n_pico, nUE_pico) * (R_pico - h_p) + h_p
    theta_pico = np.random.rand(n_pico, nUE_pico) * 2 * np.pi
    posi_pUE = (np.cos(theta_pico) + 1j * np.sin(theta_pico)) * dis_pico + posi_pico

    dis_marco = np.random.rand(n_marco, nUE_marco) * (R_marco - h_m) + h_m
    theta_marco = np.random.rand(n_marco, nUE_marco) * 2 * np.pi
    posi_mUE = (np.cos(theta_marco) + 1j * np.sin(theta_marco)) * dis_marco + posi_marco

    posi_allUE = np.concatenate((np.reshape(posi_pUE, [1, -1]), np.reshape(posi_mUE, [1, -1])), axis=1)
    posi_allBS = np.concatenate((np.reshape(posi_pico, [-1, 1]), np.reshape(posi_marco, [-1, 1])), axis=0)
    dist0 = abs(posi_allBS - posi_allUE)

    posi_allBS = np.concatenate((np.reshape(posi_pico * np.ones([1, nTX_pico]), [-1, 1]),
                                 np.reshape(posi_marco * np.ones([1, nTX_marco]), [-1, 1])), axis=0)

    dist = abs(posi_allBS - posi_allUE)

    # alpha0 = np.concatenate((np.ones([n_pico*nTX_pico, 1])*alpha_pBS, np.ones([n_marco*nTX_marco, 1])*alpha_mBS), axis=0)
    # beta = np.concatenate((np.ones([n_pico*nTX_pico, 1]) * beta_pBS, np.ones([n_marco*nTX_marco, 1])*beta_mBS), axis=0)
    # alpha_dB = -1.0*(alpha0 + beta * np.log10(dist))

    alpha_dB = cal_3slope_pathloss(dist)

    if is_shadow is True:
        x_cord = np.int64(np.real(posi_allUE + R_marco + h_m + 1j * (R_marco + h_m)))
        y_cord = np.int64(np.imag(posi_allUE + R_marco + h_m + 1j * (R_marco + h_m)))
        shadow_allUE = np.zeros([nTX_pico * n_pico + nTX_marco * n_marco, nUE_pico * n_pico + nUE_marco * n_marco])
        index = 0
        for i in range(n_pico):
            shadow_allUE[index: index + nTX_pico] = shadow_map[i, x_cord, y_cord] * np.ones([nTX_pico, 1])
            index = index + nTX_pico
        for i in range(n_pico, n_pico + n_marco):
            shadow_allUE[index: index + nTX_marco] = shadow_map[i, x_cord, y_cord] * np.ones([nTX_marco, 1])
            index = index + nTX_marco
        alpha_dB = alpha_dB - shadow_allUE

    alpha = np.power(10.0, alpha_dB / 10.0)

    return np.transpose(alpha), dist0


def cal_3slope_pathloss(dist, freq=0.2, d0=10.0, d1=50.0, h_bs=15.0, h_u=1.65):
    L = 46.3 + 33.9 * np.log10(freq) - 13.82 * np.log10(h_bs) - (1.1 * np.log10(freq) - 0.7) * h_u + (
                1.56 * np.log10(freq) - 0.8)
    PL = (dist > d1) * (-L - 35 * np.log10(dist)) + \
         (dist <= d1) * (dist > d0) * (-L - 15 * np.log10(d1) - 20 * np.log10(dist)) + \
         (dist <= d0) * (-L - 15 * np.log10(d1) - 20 * np.log10(d0))
    return PL


def gen_ppp_posi(lamda, xrange, yrange, R_cell):
    n_bs = int(np.random.poisson(lamda * xrange * yrange))
    n_bs = 10
    posi_bs = np.zeros(n_bs, dtype=complex)
    for i in range(n_bs):
        while True:
            posi = np.random.uniform(-xrange / 2, xrange / 2) + 1j * np.random.uniform(-yrange / 2, yrange / 2)
            if i == 0 or min(abs(posi_bs[0:i] - posi)) >= 0:
                posi_bs[i] = posi
                break
    return posi_bs


def water_filling(H, var_noise, P_all=1.0):
    low = 10e-6
    high = 10e2
    error = 10e-5
    for iter in range(100):
        medium = (low + high) / 2.0
        P = 1.0 / medium - var_noise / H
        P = np.maximum(P, 0.0)
        sum_P = np.sum(P)
        if np.abs(P_all - sum_P) <= error:
            lamda = medium
            break
        elif P_all - sum_P <= 0:
            low = medium
        else:
            high = medium

    return P


if __name__ == '__main__':
    # nUE_BS = np.array([30])
    # nTX_BS = np.array([64])
    # nTX_UE = 1
    # Pmax_BS = np.array([10])
    # var_noise = 1
    # H = np.random.randn(sum(nUE_BS)*nTX_UE, sum(nTX_BS)) + 1j * np.random.randn(sum(nUE_BS)*nTX_UE, sum(nTX_BS))
    # V = WMMSE_BF(H, nUE_BS, nTX_BS, nTX_UE, Pmax_BS, var_noise)

    K = 25;
    Ntx = 32;
    Pmax = 100000;
    var_noise = 0.1;
    rau = 1 / 0.311;
    rmin = 0.0;
    P0 = 43.3;
    Pc = 17.6
    H = 1 / np.sqrt(2) * (np.random.randn(K, Ntx) + 1j * np.random.randn(K, Ntx))
    V = EE_max_Precoding(H, Pmax, var_noise, rau, rmin, P0, Pc)
    # H_UEBS, _, _, Popt_BS, sum_rate = gen_channel(nUE_BS, nTX_BS, Pmax_BS, var_noise)
