import numpy as np
import matplotlib.pyplot as plt
from bsexact import bsexact

def payoff(S, K: float):
    return  np.maximum(S-K, 0)

def ai(S, dS, sigma, gamma, r):
    return (sigma**2 * S**(2*gamma))/(2 * dS**2) - r * S / (2 * dS)

def bi(S, dS, sigma, gamma, r):
    return  -(sigma**2 * S**(2*gamma)/(dS**2) + r)

def ci(S, dS, sigma, gamma, r):
    return (sigma**2 * S**(2*gamma))/(2 * dS**2) + r * S / (2 * dS)

def difference_operator(M, S_vec, sigma, gamma, r):
    dS = S_vec[-1] / M
    S = S_vec[1:-1]

    a = ai(S,dS, sigma, gamma, r)
    b = bi(S,dS, sigma, gamma, r)
    c = ci(S,dS, sigma, gamma, r)

    D = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)

    return D

def boundary_operator(M, S_max, sigma, gamma, r, K, tau):
    dS = S_max / M

    c = ci(S_max, dS, sigma, gamma, r)

    V_upper_boundary = S_max - K * np.exp(-r * tau)

    b = np.zeros(M - 1)
    b[-1] = c * V_upper_boundary

    return b

def explicit_euler(M, N, T, S_max, sigma, gamma, r, K):
    tau = 0
    dtau = T / N

    S_vec = np.linspace(0, S_max, M + 1)
    V_vec = payoff(S_vec, K)

    D_interior = difference_operator(M, S_vec, sigma, gamma, r)

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

    D_interior = difference_operator(M, S_vec, sigma, gamma, r)
    I = np.eye(M - 1)
    inv_mat = np.linalg.inv(I - dtau * D_interior)

    for n in range(N):
        tau += dtau
        b_interior = boundary_operator(M, S_max, sigma, gamma, r, K, tau)
        V_vec[1:-1] = inv_mat @ (V_vec[1:-1] + dtau * b_interior)

        V_vec[0] = 0.0
        V_vec[-1] = S_max - K * np.exp(-r * tau)

    return  V_vec


def plot_option_values_for_gammas(M, N, T, S_max, sigma, r, K):
    S_vec = np.linspace(0, S_max, M + 1)
    gamma_values = np.linspace(0.0, 1.0, 6)

    plt.figure()
    for gamma in gamma_values:
        option_values = implicit_euler(M, N, T, S_max, sigma, gamma, r, K)
        plt.plot(S_vec, option_values, label=f'gamma = {gamma:.1f}')

    plt.xlabel('Underlying price S')
    plt.ylabel('Option value V(S)')
    plt.title('Option value for different gamma values')
    plt.xlim(5, 25)
    plt.ylim(0, 10)
    plt.legend()
    plt.grid(True)


def plot_comparison_gamma_one(M, N, T, S_max, sigma, r, K):
    gamma = 1.0
    S_vec = np.linspace(0, S_max, M + 1)

    explicit_values = explicit_euler(M, N, T, S_max, sigma, gamma, r, K)
    implicit_values = implicit_euler(M, N, T, S_max, sigma, gamma, r, K)
    exact_values = np.zeros_like(S_vec)
    exact_values[1:] = [
        bsexact(sigma, r, K, T, S)
        for S in S_vec[1:]
    ]

    plt.figure()
    plt.plot(S_vec, exact_values, label='Black-Scholes exact')
    plt.plot(S_vec, explicit_values, '--', label='Explicit Euler')
    plt.plot(S_vec, implicit_values, '-.', label='Implicit Euler')
    plt.xlabel('Underlying price S')
    plt.ylabel('Option value V(S)')
    plt.title('Comparison for gamma = 1')
    plt.legend()
    plt.grid(True)



def main():
    M = 20
    N = 30
    T = 0.5
    K = 15
    r = 0.1
    sigma = 0.25
    gamma = 0.8
    S_max = 4 * K
    S_vec = np.linspace(0, S_max, M + 1)

    V1 = explicit_euler(M, N, T, S_max, sigma, gamma, r, K)
    plt.figure()
    plt.plot(S_vec, V1)
    plt.show()

    V2 = implicit_euler(M,N, T, S_max, sigma, gamma, r, K)
    plt.figure()
    plt.plot(S_vec, V2)
    plt.show()

    plot_option_values_for_gammas(M, N, T, S_max, sigma, r, K)
    plt.show()

    plot_comparison_gamma_one(M, N, T, S_max, sigma, r, K)
    plt.show()



if __name__ == '__main__':
    main()

