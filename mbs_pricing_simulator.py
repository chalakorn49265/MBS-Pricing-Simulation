
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Set parameters
notional = 100000
WAC = 0.08
r_monthly = WAC / 12
N_months = 360
kappa = 0.6
r_bar = 0.08
sigma = 0.12
r0 = 0.078
dt = 1 / 12
n_paths = 10000
seasonality = np.array([0.94, 0.76, 0.74, 0.95, 0.98, 0.92, 0.98, 1.10, 1.18, 1.22, 1.23, 0.98])

# CIR simulation
def simulate_CIR_paths(r0, kappa, r_bar, sigma, T, dt, n_paths):
    n_steps = int(T / dt)
    r = np.zeros((n_paths, n_steps + 1))
    r[:, 0] = r0
    for t in range(1, n_steps + 1):
        sqrt_rt = np.sqrt(np.maximum(r[:, t - 1], 0))
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        dr = kappa * (r_bar - r[:, t - 1]) * dt + sigma * sqrt_rt * dW
        r[:, t] = r[:, t - 1] + dr
    return r

# Monthly Payment
def compute_MP(PV, r, n):
    return PV * r / (1 - (1 + r)**(-n))

# Discount Factors
def discount_factors(rate_path, dt):
    return np.exp(-np.cumsum(rate_path * dt))

# 10-year rate approximation
def compute_r10(rate_path, dt, T=10):
    n_steps = int(T / dt)
    integrals = np.cumsum(rate_path[:, 1:n_steps+1] * dt, axis=1)
    return integrals[:, -1] / T

# CPR model
def compute_CPR(t, PV_t, PV0, R, r10, month):
    try:
        RI = 0.28 + 0.14 * np.arctan(-8.57 + 430 * (R - r10))
    except:
        RI = 0.28
    BU = 0.3 + 0.7 * (PV_t / PV0)
    SG = min(1, t / 30)
    SY = seasonality[month % 12]
    CPR = RI * BU * SG * SY
    return np.clip(CPR, 0, 1)

# Price MBS
def price_MBS():
    rate_paths = simulate_CIR_paths(r0, kappa, r_bar, sigma, 30, dt, n_paths)
    payments = np.zeros((n_paths, N_months))
    discount = np.zeros_like(payments)

    for i in range(n_paths):
        PV = notional
        r10_estimate = compute_r10(rate_paths[i:i+1], dt)[0]

        for t in range(N_months):
            month = t % 12
            MP = compute_MP(PV, r_monthly, N_months - t)
            IP = PV * r_monthly
            SP = MP - IP

            CPR = compute_CPR(t + 1, PV, notional, WAC, r10_estimate, month)
            SMM = 1 - (1 - CPR) ** (1 / 12)
            PP = (PV - SP) * SMM

            payments[i, t] = MP + PP
            PV -= (SP + PP)

        discount[i, :] = discount_factors(rate_paths[i, :-1], dt)

    return np.mean(np.sum(payments * discount, axis=1))

# OAS pricing
def price_MBS_with_OAS(OAS_spread):
    rate_paths = simulate_CIR_paths(r0, kappa, r_bar, sigma, 30, dt, n_paths)
    payments = np.zeros((n_paths, N_months))
    discount = np.zeros_like(payments)

    for i in range(n_paths):
        PV = notional
        r10_estimate = compute_r10(rate_paths[i:i+1], dt)[0]

        for t in range(N_months):
            month = t % 12
            MP = compute_MP(PV, r_monthly, N_months - t)
            IP = PV * r_monthly
            SP = MP - IP

            CPR = compute_CPR(t + 1, PV, notional, WAC, r10_estimate, month)
            SMM = 1 - (1 - CPR) ** (1 / 12)
            PP = (PV - SP) * SMM

            payments[i, t] = MP + PP
            PV -= (SP + PP)

        discount[i, :] = np.exp(-np.cumsum((rate_paths[i, :-1] + OAS_spread) * dt))

    return np.mean(np.sum(payments * discount, axis=1))

# Target function for OAS
def target_OAS(oas_guess, market_price):
    return price_MBS_with_OAS(oas_guess) - market_price

# IO/PO pricing
def price_IO_PO_tranches():
    rate_paths = simulate_CIR_paths(r0, kappa, r_bar, sigma, 30, dt, n_paths)
    IO_payments = np.zeros((n_paths, N_months))
    PO_payments = np.zeros((n_paths, N_months))
    discount = np.zeros_like(IO_payments)

    for i in range(n_paths):
        PV = notional
        r10_estimate = compute_r10(rate_paths[i:i+1], dt)[0]

        for t in range(N_months):
            month = t % 12
            MP = compute_MP(PV, r_monthly, N_months - t)
            IP = PV * r_monthly
            SP = MP - IP

            CPR = compute_CPR(t + 1, PV, notional, WAC, r10_estimate, month)
            SMM = 1 - (1 - CPR) ** (1 / 12)
            PP = (PV - SP) * SMM

            IO_payments[i, t] = IP
            PO_payments[i, t] = PP + SP
            PV -= (SP + PP)

        discount[i, :] = discount_factors(rate_paths[i, :-1], dt)

    IO_price = np.mean(np.sum(IO_payments * discount, axis=1))
    PO_price = np.mean(np.sum(PO_payments * discount, axis=1))

    return IO_price, PO_price

# Price vs r_bar plot
def price_vs_r_bar():
    r_bars = np.arange(0.04, 0.11, 0.01)
    prices = []
    global r_bar
    for r in r_bars:
        r_bar = r
        prices.append(price_MBS())

    plt.figure()
    plt.plot(r_bars, prices, marker='o')
    plt.xlabel('Mean Reversion Level (r̄)')
    plt.ylabel('MBS Price')
    plt.title('MBS Price Sensitivity to Long-Term Rate (r̄)')
    plt.grid(True)
    plt.show()
