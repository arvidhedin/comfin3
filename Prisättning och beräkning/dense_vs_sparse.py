import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from assignment2_fdm import payoff, ai, bi, ci, boundary_operator
from bsexact import bsexact


def difference_operator_sparse(M, S_vec, sigma, gamma, r):
    dS = S_vec[-1] / M
    S = S_vec[1:-1]

    a = ai(S,dS, sigma, gamma, r)
    b = bi(S,dS, sigma, gamma, r)
    c = ci(S,dS, sigma, gamma, r)

    D_sparse = sp.diags([a[1:], b, c[:-1]], [-1, 0, 1], format = 'csr')

    return D_sparse


def explicit_euler(M, N, T, S_max, sigma, gamma, r, K):
    tau = 0
    dtau = T / N

    S_vec = np.linspace(0, S_max, M + 1)
    V_vec = payoff(S_vec, K)

    D_interior = difference_operator_sparse(M, S_vec, sigma, gamma, r)

    for n in range(1, N + 1):
        b_interior = boundary_operator(M, S_max, sigma, gamma, r, K, tau)
        V_vec[1:-1] = V_vec[1:-1] + dtau*(D_interior@V_vec[1:-1] + b_interior)

        tau += dtau

        V_vec[0] = 0
        V_vec[-1] = S_max - K * np.exp(-r * tau)

    return V_vec

def implicit_euler(M, N, T, S_max, sigma, gamma, r, K):
    tau = 0
    dtau = T / N

    S_vec = np.linspace(0, S_max, M + 1)
    V_vec = payoff(S_vec, K)

    D_interior = difference_operator_sparse(M, S_vec, sigma, gamma, r)
    I = sp.eye(M - 1)
    inv_mat = np.linalg.inv(I - dtau * D_interior)

    for n in range(N):
        tau += dtau
        b_interior = boundary_operator(M, S_max, sigma, gamma, r, K, tau)
        V_vec[1:-1] = inv_mat @ (V_vec[1:-1] + dtau * b_interior)

        V_vec[0] = 0.0
        V_vec[-1] = S_max - K * np.exp(-r * tau)

    return  V_vec

