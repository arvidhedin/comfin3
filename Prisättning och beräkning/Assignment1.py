import numpy as np
from bsexact import bsexact
import matplotlib.pyplot as plt
S0 = 14
K = 15
r = 0.1
sigma = 0.25
T = 0.5
gamma = 1
N = 100 # number of samples
M = 100  # number of time steps
dt = T / M
V = 0
for i in range(N):
    S = S0
    for j in range(M):
        dW = np.random.normal(0, np.sqrt(dt))
        S+= r * S * dt + sigma * S**gamma * dW
    V += max(S - K, 0)
V /= N
discounted_payoff = np.exp(-r * T) * V

# Calculating the exact price
exact_price = bsexact(sigma, r, K, T, S0)
#sample error as a function of the number of sample paths
sample_errors = []
for n in [100, 500, 1000, 2000, 5000, 10000]:
    V = 0
    for i in range(n):
        S = S0
        for j in range(M):
            dW = np.random.normal(0, np.sqrt(dt))
            S+= r * S * dt + sigma * S**gamma * dW
        V += max(S - K, 0)
    V /= n
    discounted_payoff = np.exp(-r * T) * V
    sample_errors.append(abs(discounted_payoff - exact_price))

#the discretization error as a function of the time step
discretization_errors = []
for m in [10, 50, 100, 200, 500, 1000,2000]:
    dt = T / m
    V = 0
    for i in range(N):
        S = S0
        for j in range(m):
            dW = np.random.normal(0, np.sqrt(dt))
            S+= r * S * dt + sigma * S**gamma * dW
        V += max(S - K, 0)
    V /= N
    discounted_payoff = np.exp(-r * T) * V
    discretization_errors.append(abs(discounted_payoff - exact_price))

#reducing sample error using antithetic variates
def antithetic_price(n_pairs, m):
    dt = T / m
    payoff_sum = 0.0
    for _ in range(n_pairs):
        S_plus = S0
        S_minus = S0

        for _ in range(m):
            dW = np.random.normal(0, np.sqrt(dt))

            S_plus += r * S_plus * dt + sigma * S_plus**gamma * dW
            S_minus += r * S_minus * dt - sigma * S_minus**gamma * dW

        payoff_plus = max(S_plus - K, 0)
        payoff_minus = max(S_minus - K, 0)

        payoff_sum += 0.5 * (payoff_plus + payoff_minus)

    return np.exp(-r * T) * payoff_sum / n_pairs

def milstein_price(n_paths, m):
    dt = T / m
    payoff_sum = 0.0

    for _ in range(n_paths):
        S = S0

        for _ in range(m):
            dW = np.random.normal(0, np.sqrt(dt))
            diffusion = sigma * S**gamma
            milstein_correction = 0.5 * sigma**2 * gamma * S**(2 * gamma - 1) * (dW**2 - dt)
            S += r * S * dt + diffusion * dW + milstein_correction

        payoff_sum += max(S - K, 0)

    return np.exp(-r * T) * payoff_sum / n_paths

def euler_price_for_gamma(n_paths, m, gamma_value):
    dt = T / m
    S = np.full(n_paths, S0, dtype=float)
    sqrt_dt = np.sqrt(dt)

    for _ in range(m):
        dW = np.random.normal(0, sqrt_dt, n_paths)
        S += r * S * dt + sigma * S**gamma_value * dW

    payoff = np.maximum(S - K, 0)
    return np.exp(-r * T) * np.mean(payoff)

gamma_values = np.linspace(0.5, 1.0, 11)
gamma_plot_paths = 10000
gamma_prices = [
    euler_price_for_gamma(gamma_plot_paths, M, gamma_value)
    for gamma_value in gamma_values
]

#sample error as func of the no of sample paths for antithetic variates
sample_errors_antithetic = []
for n in [100, 500, 1000, 2000, 5000, 10000]:
    discounted_payoff = antithetic_price(n // 2, M)
    sample_errors_antithetic.append(abs(discounted_payoff - exact_price))

#reducing discretization error using the Milstein scheme
milstein_discretization_errors = []
for m in [10, 50, 100, 200, 500, 1000, 2000]:
    milstein_estimate = milstein_price(N, m)
    milstein_discretization_errors.append(abs(milstein_estimate - exact_price))


plt.figure(figsize=(10, 6))
plt.loglog([100, 500, 1000, 2000, 5000, 10000], sample_errors, 'o-')
plt.loglog([100, 500, 1000, 2000, 5000, 10000], sample_errors_antithetic, 's-')
plt.xlabel('Number of Sample Paths')
plt.ylabel('Absolute Sample Error')
plt.title('Convergence of Monte Carlo Estimate')
plt.legend(['Sample Error', 'Antithetic Variates'])
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.loglog([10, 50, 100, 200, 500, 1000, 2000], discretization_errors, 'o-')
plt.loglog([10, 50, 100, 200, 500, 1000, 2000], milstein_discretization_errors, 's-')
plt.xlabel('Number of Time Steps')  
plt.ylabel('Absolute Discretization Error')
plt.title('Convergence of Discretization')
plt.legend(['Euler-Maruyama', 'Milstein'])
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(gamma_values, gamma_prices, 'o-')
plt.xlabel('Gamma')
plt.ylabel('Option Value')
plt.title('European Call Option Value as a Function of Gamma')
plt.grid(True)
plt.show()


print(f"The estimated price of the European call option is: {discounted_payoff:.4f}")
print(f"The difference is: {abs(discounted_payoff - exact_price):.4f}")
print(f"The price using antithetic variates is: {antithetic_price(5000, M):.4f}")
print(f"The sample error using antithetic variates is: {abs(antithetic_price(5000, M) - exact_price):.4f}")
print(f"The price using Milstein is: {milstein_price(N, M):.4f}")
for gamma_value, gamma_price in zip(gamma_values, gamma_prices):
    print(f"Gamma = {gamma_value:.2f}, option value = {gamma_price:.4f}")

