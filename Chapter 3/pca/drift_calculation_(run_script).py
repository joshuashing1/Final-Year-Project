# drift_calculation_(run_script).py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fwd_plot_historical import parse_tenor
from pca_fn import PCAFactors
from volatility_fn import VolatilityFitter, PolynomialInterpolator

# -------------------------------
# Load & align data
# -------------------------------
df = pd.read_csv(r"data\GLC_fwd_curve_raw.csv")
df = df.set_index("time")

tenor_years = pd.Series({c: parse_tenor(c) for c in df.columns})
tenor_years = tenor_years.sort_values()
df = df.loc[:, tenor_years.index]

T = df.index.to_numpy()          # time axis (rows)
Tau = tenor_years.to_numpy()     # tenor in years (columns)
Z = df.to_numpy(dtype=float)     # rates matrix [n_time, n_tenor]

# -------------------------------
# PCA: diff -> cov -> eig -> top-k
# -------------------------------
factors = 3
pca = PCAFactors(k=factors, annualize=252.0)

# 1) Differences along time: df(t, τ)
diff_rates = pca.difference(Z, axis=0)

# 2) Covariance across tenors (annualized)
sigma = pca.covariance(diff_rates)
print("Sigma shape:", sigma.shape)

# 3) Eigenvalues / eigenvectors (unsorted)
eigval_all = pca.eigenvalues(sigma)
eigvec_all = pca.eigenvectors(sigma)
print("All eigenvalues:")
print(eigval_all)

# 4) Top-k principal components (sorted by descending eigenvalue)
princ_eigval, princ_comp, order = pca.pca(sigma, k=factors)
print("\nPrincipal eigenvalues (top-k):")
print(princ_eigval)

print("\nPrincipal components (columns = PCs):")
print(princ_comp)

# --- Plot principal components vs tenor
plt.figure(figsize=(9, 5))
for i in range(princ_comp.shape[1]):
    plt.plot(Tau, princ_comp[:, i], marker='.', label=f'PC{i+1}')
plt.title('Principal components (eigenvectors over tenor)')
plt.xlabel(r'Tenor $\tau$ (years)')
plt.ylabel('Loading')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# Discretized volatility functions from PCs
# vols[:, i] = sqrt(lambda_i) * PC_i(tenor)
# -------------------------------
vols = VolatilityFitter.from_pca(princ_eigval, princ_comp)  # shape [n_tenor, k]
print("vols shape:", vols.shape)

plt.figure(figsize=(10, 4))
for i in range(vols.shape[1]):
    plt.plot(Tau, vols[:, i], marker='.', label=f'Vol {i+1}')
plt.xlabel(r'Tenor $\tau$ (years)')
plt.ylabel(r'Volatility $\sigma$')
plt.title('Discretized volatilities')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# Fit volatility polynomials per factor
# (match notebook: degrees 0, 3, 3 for 3 factors)
# -------------------------------
vf = VolatilityFitter(Tau, vols)
vf.fit([0, 3, 3])  # broadcast or supply any list/tuple of length k

# Visualize fit on original tenors
plt.figure(figsize=(15, 4))
for i in range(vf.k):
    plt.subplot(1, vf.k, i + 1)
    plt.plot(Tau, vols[:, i], marker='.', label='Discretized')
    plt.plot(Tau, vf.models[i](Tau), label='Fitted')
    plt.title(f'Factor {i+1} fit')
    plt.xlabel(r'Tenor $\tau$ (years)')
    if i == 0:
        plt.ylabel('Vol')
    plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# Discretize fitted volatilities on a grid for drift calculation
# -------------------------------
mc_tenors = np.linspace(0.0, 25.0, 51)  # 0, 0.5, ..., 25
mc_vols = vf.predict(mc_tenors)         # shape [len(mc_tenors), k]

plt.figure(figsize=(10, 4))
plt.plot(mc_tenors, mc_vols, marker='.')
plt.xlabel(r'Tenor $\tau$ (years)')
plt.title('Volatilities (fitted, discretized)')
plt.tight_layout()
plt.show()

# -------------------------------
# Drift calculation (HJM risk-neutral drift)
# m(τ) = Σ_i [ (∫_0^τ σ_i(u) du) * σ_i(τ) ]
# -------------------------------
def integrate(f, x0: float, x1: float, dx: float) -> float:
    """Uniform-step trapezoid rule."""
    n = int((x1 - x0) / dx) + 1
    xs = np.linspace(x0, x1, n)
    ys = f(xs) if callable(getattr(f, "__call__", None)) else np.array([f(x) for x in xs])
    return float(np.trapz(ys, xs))

def m_tau(tau: float, models) -> float:
    out = 0.0
    for fitted in models:
        # Each 'fitted' is a PolynomialInterpolator; it is vectorized via __call__
        out += integrate(fitted, 0.0, tau, 0.01) * float(fitted(tau))
    return out

mc_drift = np.array([m_tau(tau, vf.models) for tau in mc_tenors])

plt.figure(figsize=(10, 4))
plt.plot(mc_tenors, mc_drift, marker='.')
plt.xlabel(r'Tenor $\tau$ (years)')
plt.title('Risk-neutral drift')
plt.tight_layout()
plt.show()
