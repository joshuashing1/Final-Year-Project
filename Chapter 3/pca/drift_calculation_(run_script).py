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
print(eigval_all.shape)
print(eigvec_all.shape)

# 4) Top-k principal components (sorted by descending eigenvalue)
princ_eigval, princ_comp, order = pca.pca(sigma, k=factors)
print("\nPrincipal eigenvalues (top-3):")
print(princ_eigval)

print("\nPrincipal components (columns = PCs):")
print(princ_comp)

DPI = 100
W_PX = 1573
H_PX = 750
W_IN = W_PX / DPI
H_IN = H_PX / DPI

fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)
for i in range(princ_comp.shape[1]):
    ax.plot(Tau, princ_comp[:, i], marker='.', label=f'PC{i+1}')
ax.set_title('Principal eigenvectors', fontsize=37, fontweight="bold", pad=12)
ax.set_xlabel(r'Tenor $T$ (years)', fontsize=32)
ax.legend(fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=27)
ax.tick_params(axis='both', which='minor', labelsize=27)
fig.tight_layout()
save_path = "principal_components.png"
plt.savefig(save_path, dpi=DPI)
plt.show()

vols = VolatilityFitter.from_pca(princ_eigval, princ_comp)
print("vols shape:", vols.shape)

fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)
for i in range(vols.shape[1]):
    ax.plot(Tau, vols[:, i], marker='.', label=f'Vol {i+1}')
ax.set_title('Discretized Volatilities', fontsize=37, fontweight="bold", pad=12)
ax.set_xlabel(r'Tenor $T$ (years)', fontsize=32)
ax.set_ylabel("Volatility $\sigma$", fontsize=32)
ax.tick_params(axis='both', which='major', labelsize=27)
ax.tick_params(axis='both', which='minor', labelsize=27)
fig.tight_layout()
save_path = "discretized_volatility.png"
plt.savefig(save_path, dpi=DPI)
plt.show()

vf = VolatilityFitter(Tau, vols)
vf.fit([0, 3, 3])  # polynomial interpolation of degrees k for principal components 1, 2, 3 resp.

fig, axes = plt.subplots(1, vf.k, figsize=(18, 6), dpi=DPI)
if vf.k == 1:
    axes = [axes]
for i, ax in enumerate(axes):
    ax.plot(Tau, vols[:, i], marker='.', label='Discretized')
    ax.plot(Tau, vf.models[i](Tau), label='Fitted')
    ax.set_title(f'Factor {i+1} fit', fontsize=22, fontweight="bold")
    ax.set_xlabel(r'Tenor $T$ (years)', fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.legend(fontsize=14)
fig.tight_layout(pad=3.0, w_pad=2.0, h_pad=2.0)
save_path = "interpolated_volatility.png"
fig.savefig(save_path, dpi=DPI)
plt.show()

print(f"Saved figure to {save_path}")

mc_tenors = np.linspace(0.0, 25.0, 51)  # based on GLC tenor structure
mc_vols = vf.predict(mc_tenors)         

def integrate(f, x0, x1, dx=0.01):
    n = int((x1 - x0) / dx) + 1
    xs = np.linspace(x0, x1, n)
    return float(np.trapz(f(xs), xs))

def m_tau(tau, models):
    return sum(integrate(m, 0.0, tau) * float(m(tau)) for m in models)

mc_drift = np.array([m_tau(tau, vf.models) for tau in mc_tenors])  
mc_vols  = vf.predict(pick_tau)                                

DPI = 100
W_PX = 1573
H_PX = 750
W_IN = W_PX / DPI
H_IN = H_PX / DPI

fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)
ax.plot(mc_tenors, mc_drift, marker='.')
ax.set_title("Risk-neutral drift", fontsize=37, fontweight="bold", pad=12)
ax.set_xlabel(r"Tenor $T$ (years)", fontsize=32)
ax.set_ylabel(r"Drift $\mu(t,T)$", fontsize=32)
ax.tick_params(axis="both", which="major", labelsize=27)
ax.tick_params(axis="both", which="minor", labelsize=27)
fig.tight_layout()
save_path = "risk_neutral_drift.png"
fig.savefig(save_path, dpi=DPI)
plt.show()

curve_spot = np.array(Z[0,:].flatten())[0]

label_to_tau  = np.array([parse_tenor(x) for x in labels])
label_col_idx = [int(np.where(np.isclose(Tau, t))[0][0]) for t in label_to_tau]
curve_spot_vec = Z[0, label_col_idx].astype(float)                 
drift_at_labels = np.array([m_tau(t, vf.models) for t in label_to_tau])  
vols_at_labels  = vf.predict(label_to_tau)                         
n_days = len(T)
timeline_years = np.arange(n_days) / 252.0
hist_path = df[labels].to_numpy(dtype=float)                       

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

fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
for j, ax in enumerate(axes.ravel()):
    ax.plot(timeline_years, hist_path[:, j], label="Historical", lw=1.5)
    ax.plot(timeline_years, sim_path[:, j],  label="Simulated", ls="--", lw=1.0)
    ax.set_title(labels[j], fontsize=16, fontweight="bold")
    ax.set_xlabel("Time t (years)", fontsize=14)
    ax.set_ylabel(r"$f(t, T)$", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(fontsize=12)
fig.suptitle("Forward Rates: Simulated vs Historical", fontsize=20, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96]) 
save_path = "forward_rates_simulated_curves.png"
fig.savefig(save_path, dpi=150)
plt.show()