"""Current script is very seasoned to the GLC dataset. Not generalized."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import normal
from pca_fn import PCAFactors
from volatility_fn import VolatilityFitter, PolynomialInterpolator

df = pd.read_csv(r"Chapter 3\data\GLC_fwd_curve_raw.csv")
# df = df / 100.0
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

labels   = ['1M','3M','6M','1Y','3Y','5Y','10Y','20Y','25Y']
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

mc_drift = np.array([m_tau(t, vf.models) for t in pick_tau])  # shape (9,)
mc_vols  = vf.predict(pick_tau)                                # shape (9, k)

# ---------------- Timeline: use CSV index directly ----------------
timeline = (T - T[0]).astype(float)   # start at 0; units = CSV index units
TRADING_DAYS = 252.0

# ---------------- Stable τ-gradient ----------------
def tau_gradient(y: np.ndarray, tau: np.ndarray) -> np.ndarray:
    g = np.empty_like(y)
    if len(tau) == 1:
        g[:] = 0.0
    else:
        g[1:-1] = (y[2:] - y[:-2]) / (tau[2:] - tau[:-2])
        g[0]    = (y[1]  - y[0])   / (tau[1]  - tau[0])
        g[-1]   = (y[-1] - y[-2])  / (tau[-1] - tau[-2])
    return g

def simulate_forward_curve(f0: np.ndarray,
                           tenors: np.ndarray,
                           drift:  np.ndarray,
                           vols_grid: np.ndarray,   # [nTenor, k]
                           timeline: np.ndarray) -> np.ndarray:
    """
    Euler (time in YEARS):
      f_{t+dt}(τ) = f_t(τ) + (∂f/∂τ + m(τ)) * dt_years + Σ_i σ_i(τ) * dW_i,
    where σ are annualized (per √year) and dW_i ~ N(0, dt_years).
    """
    k = vols_grid.shape[1]
    vols_Tk = np.asarray(vols_grid).T                # k × nTenor
    f = f0.copy()
    out = [f.copy()]
    for it in range(1, len(timeline)):
        dt_rows   = float(timeline[it] - timeline[it-1])   # row steps (e.g., days)
        if dt_rows <= 0:
            out.append(f.copy()); continue
        dt_years  = dt_rows / TRADING_DAYS                 # <<< key fix
        dW        = np.random.normal(size=k) * np.sqrt(dt_years)
        conv      = tau_gradient(f, tenors)                # ∂f/∂τ
        diff      = vols_Tk.T @ dW                         # (nTenor,)
        f         = f + (conv + drift) * dt_years + diff

        # numerical hygiene in case of rare overshoots
        if not np.all(np.isfinite(f)):
            f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        out.append(f.copy())
    return np.vstack(out)                                  # [n_time, n_tenor]


np.random.seed(0)
f0 = np.interp(pick_tau, Tau, Z[0])                 # start from first CSV row on 9 maturities
sim_mat = simulate_forward_curve(f0, pick_tau, mc_drift, mc_vols, timeline)   # [n_time, 9]

# ---------------- Historical series at the same 9 maturities ----------------
hist_cols = [int(np.argmin(np.abs(Tau - tau))) for tau in pick_tau]
hist_series = Z[:, hist_cols]                        # [n_time, 9]

# ---------------- Plot: Historical vs Expected (9-panel) ----------------
fig, axs = plt.subplots(3, 3, figsize=(14, 10)); axs = axs.ravel()
for i, (ax, lbl) in enumerate(zip(axs, labels)):
    ax.plot(timeline, hist_series[:, i], label='Historical', lw=1.4, color='green')
    ax.plot(timeline, sim_mat[:, i],     label='Expected',   lw=1.2, color='blue')
    ax.set_title(f"Maturity: {lbl}")
    ax.grid(True, alpha=0.3)
    if i % 3 == 0: ax.set_ylabel(r"Forward rate $f(t,\tau)$")
    if i >= 6:     ax.set_xlabel("Time")
handles, labs = axs[-1].get_legend_handles_labels()
fig.legend(handles, labs, loc="lower center", ncol=2)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()