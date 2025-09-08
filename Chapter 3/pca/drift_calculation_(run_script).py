"""Current script is very seasoned to the GLC dataset. Not generalized."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import normal
from pca_fn import PCAFactors
from volatility_fn import VolatilityFitter, PolynomialInterpolator

df = pd.read_csv(r"Chapter 3\data\GLC_fwd_curve_raw.csv")
df = df / 100.0
df = df.set_index("time")

def parse_tenor(s: str) -> float:
    """Return tenor in years from labels like '1M','6M','1.0Y','10.0Y'."""
    s = s.strip().upper()
    if s.endswith("M"):
        return float(s[:-1]) / 12.0
    if s.endswith("Y"):
        return float(s[:-1])
    return float(s)

tenor_years = pd.Series({c: parse_tenor(c) for c in df.columns})
tenor_years = tenor_years.sort_values()
df = df.loc[:, tenor_years.index]

T = df.index.to_numpy()          # time axis (rows)
Tau = tenor_years.to_numpy()     # tenor in years (columns)
Z = df.to_numpy(dtype=float)     # rates matrix [n_time, n_tenor]

labels   = ['1M','6M','1.0Y','2.0Y','3.0Y','5.0Y','10.0Y','20.0Y','25.0Y']
pick_tau = np.array([parse_tenor(x) for x in labels])

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

mc_tenors = np.linspace(0.0, 25.0, 51)  # based on GLC tenor structure
mc_vols = vf.predict(mc_tenors)         

def integrate(f, x0, x1, dx=0.01):
    n = int((x1 - x0) / dx) + 1
    xs = np.linspace(x0, x1, n)
    return float(np.trapz(f(xs), xs))

def m_tau(tau, models):
    return sum(integrate(m, 0.0, tau) * float(m(tau)) for m in models)

mc_drift = np.array([m_tau(tau, vf.models) for tau in mc_tenors])  # shape (9,)
mc_vols  = vf.predict(pick_tau)                                # shape (9, k)

plt.figure(figsize=(10, 4))
plt.plot(mc_tenors, mc_drift, marker='.')
plt.xlabel(r'Tenor $\tau$ (years)')
plt.title('Risk-neutral drift')
plt.tight_layout()
plt.show()

curve_spot = np.array(Z[0,:].flatten())[0]

# --- Align to requested tenors
label_to_tau  = np.array([parse_tenor(x) for x in labels])
label_col_idx = [int(np.where(np.isclose(Tau, t))[0][0]) for t in label_to_tau]
curve_spot_vec = Z[0, label_col_idx].astype(float)                 # (9,)
drift_at_labels = np.array([m_tau(t, vf.models) for t in label_to_tau])  # (9,)
vols_at_labels  = vf.predict(label_to_tau)                         # (9, k)
n_days = len(T)
timeline_years = np.arange(n_days) / 252.0
hist_path = df[labels].to_numpy(dtype=float)                       # (n_days, 9)

# --- Simulation (one path)
def simulate_forward_path(f0, tau, drift, vols, tgrid, seed=123):
    f = f0.copy()
    N, K = len(tau), vols.shape[1]
    path = np.empty((len(tgrid), N), float); path[0] = f
    rng = np.random.default_rng(seed); vols_T = vols.T
    for it in range(1, len(tgrid)):
        dt = tgrid[it] - tgrid[it-1]; z = rng.normal(size=K); fprev = f.copy()
        for i in range(N):
            i1 = i+1 if i < N-1 else i-1
            f[i] = (fprev[i] + drift[i]*dt
                    + np.dot(vols_T[:, i], z)*np.sqrt(dt)
                    + (fprev[i1]-fprev[i])/(tau[i1]-tau[i]) * dt)
        path[it] = f
    return path

sim_path = simulate_forward_path(curve_spot_vec, label_to_tau, drift_at_labels, vols_at_labels, timeline_years)

# --- Plots: Simulated vs Historical for 9 tenors
fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
for ax, j in zip(axes.ravel(), range(len(labels))):
    ax.plot(timeline_years, hist_path[:, j], label='Historical', lw=1.5)
    ax.plot(timeline_years, sim_path[:, j],  label='Simulated',  lw=1.0, ls='--')
    ax.set_title(labels[j]); ax.set_xlabel('Time t (years)'); ax.set_ylabel('f(t, τ)')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
plt.suptitle('Forward Rates: Simulated vs Historical (by Tenor)', y=0.98)
plt.tight_layout(); plt.show()
