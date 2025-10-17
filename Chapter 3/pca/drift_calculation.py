import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pca_fn import PCAFactors

def parse_tenor(s: str) -> float:
    s = s.strip().upper()
    if s.endswith("M"): return float(s[:-1]) / 12.0
    if s.endswith("Y"): return float(s[:-1])
    return float(s)

def vols_from_pca(princ_eigval: np.ndarray, princ_comp: np.ndarray) -> np.ndarray:
    """Discretized volatility functions: vols[:, i] = sqrt(lambda_i) * PC_i(T)."""
    return np.asarray(princ_comp, float) * np.sqrt(np.asarray(princ_eigval, float))[None, :]

def fit_poly_per_factor(T: np.ndarray, V: np.ndarray, degrees) -> list[np.ndarray]:
    """Return list of coeff arrays (descending powers), one per factor."""
    if isinstance(degrees, int): degrees = [degrees] * V.shape[1]
    elif len(degrees) == 1:     degrees = [degrees[0]] * V.shape[1]
    return [np.polyfit(T, V[:, i], deg) for i, deg in enumerate(degrees)]

def eval_polys(coeff_list: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    """Stacked evaluation -> shape [len(x), k]."""
    x = np.asarray(x, float)
    return np.column_stack([np.polyval(c, x) for c in coeff_list])

def drift_m_tau(tau: float, coeff_list: list[np.ndarray], n_points: int = 500) -> float:
    """
    μ(t, τ) = Σ_k (∫_0^τ σ_k(u) du) * σ_k(τ),
    where ∫ is evaluated numerically with np.trapz.
    
    n_points: resolution of the integration grid.
    """
    s = 0.0
    grid = np.linspace(0.0, tau, n_points)
    for c in coeff_list:
        vals = np.polyval(c, grid)
        integral = np.trapz(vals, grid)   # approximate ∫_0^τ σ(u) du
        s += integral * np.polyval(c, tau)
    return float(s)


def simulate_fwd_path(f0, tau, drift_vals, vols_at_tau, tgrid, seed=123):
    """
    Euler–Maruyama with Musiela shift. vols_at_tau: [N, K] vol columns per factor.
    Drift supplied at grid 'tau' as drift_vals (length N).
    """
    f = f0.copy()
    N, K = vols_at_tau.shape
    path = np.empty((len(tgrid), N), float)
    path[0] = f
    rng = np.random.default_rng(seed)

    for it in range(1, len(tgrid)):
        dt = tgrid[it] - tgrid[it - 1]
        fprev = f.copy()
        dfdtau = np.gradient(fprev, tau)               # Musiela shift term
        z = rng.normal(size=K)
        diffusion = vols_at_tau @ (z * np.sqrt(dt))    # shape (N,)
        f = fprev + (drift_vals + dfdtau) * dt + diffusion
        path[it] = f
    return path

df = pd.read_csv(r"Chapter 3\data\GLC_fwd_curve_raw.csv")
df = df / 100.0
df = df.set_index("t")

tenor_years = pd.Series({c: parse_tenor(c) for c in df.columns}).sort_values()
df = df.loc[:, tenor_years.index]

T = df.index.to_numpy()              # time axis (rows)
Tau = tenor_years.to_numpy()         # tenor in years (columns)
Z = df.to_numpy(dtype=float)         # [n_time, n_tenor]

labels   = ['1M','6M','1.0Y','2.0Y','3.0Y','5.0Y','10.0Y','20.0Y','25.0Y']
pick_tau = np.array([parse_tenor(x) for x in labels])

factors = 3
pca = PCAFactors(k=factors, annualize=252.0)
diff_rates = pca.difference(Z, axis=0)
sigma = pca.covariance(diff_rates)
princ_eigval, princ_comp, order = pca.pca(sigma, k=factors)   # eigvals desc, PCs columns

vols = vols_from_pca(princ_eigval, princ_comp)                # shape [n_tenor, k]
print(vols)

DPI, W_PX, H_PX = 100, 1573, 750
W_IN, H_IN = W_PX / DPI, H_PX / DPI

fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)
for i in range(princ_comp.shape[1]):
    ax.plot(Tau, princ_comp[:, i], marker='.', label=f'PC{i+1}')
ax.set_title('Principal eigenvectors', fontsize=37, fontweight="bold", pad=12)
ax.set_xlabel(r'Tenor $T$ (years)', fontsize=32)
ax.legend(fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=27)
ax.tick_params(axis='both', which='minor', labelsize=27)
fig.tight_layout()
plt.savefig("principal_components.png", dpi=DPI)
plt.show()

fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)
for i in range(vols.shape[1]):
    ax.plot(Tau, vols[:, i], marker='.', label=f'Vol {i+1}')
ax.set_title('Discretized Volatilities', fontsize=37, fontweight="bold", pad=12)
ax.set_xlabel(r'Tenor $T$ (years)', fontsize=32)
ax.set_ylabel("Volatility $\sigma$", fontsize=32)
ax.tick_params(axis='both', which='major', labelsize=27)
ax.tick_params(axis='both', which='minor', labelsize=27)
fig.tight_layout()
plt.savefig("discretized_volatility.png", dpi=DPI)
plt.show()

coeff_list = fit_poly_per_factor(Tau, vols, [0, 3, 3]) # degree of interpolant

fig, axes = plt.subplots(1, len(coeff_list), figsize=(18, 6), dpi=DPI)
if len(coeff_list) == 1: axes = [axes]
for i, ax in enumerate(axes):
    ax.plot(Tau, vols[:, i], marker='.', label='Discretized')
    ax.plot(Tau, np.polyval(coeff_list[i], Tau), label='Fitted')
    ax.set_title(f'Factor {i+1} fit', fontsize=22, fontweight="bold")
    ax.set_xlabel(r'Tenor $T$ (years)', fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.legend(fontsize=14)
fig.tight_layout(pad=3.0, w_pad=2.0, h_pad=2.0)
plt.savefig("interpolated_volatility.png", dpi=DPI)
plt.show()

mc_tenors = np.linspace(0.0, 25.0, 51)
mc_drift = np.array([drift_m_tau(tau, coeff_list) for tau in mc_tenors])

curve_spot_vec = df.loc[df.index[0], labels].to_numpy(dtype=float)

label_to_tau  = pick_tau
label_col_idx = [int(np.where(np.isclose(Tau, t))[0][0]) for t in label_to_tau]
hist_path = df[labels].to_numpy(dtype=float)

drift_at_labels = np.array([drift_m_tau(t, coeff_list) for t in label_to_tau])
vols_at_labels  = eval_polys(coeff_list, label_to_tau)          # shape [N, K]

timeline_years = np.arange(len(T)) / 252.0
sim_path = simulate_fwd_path(curve_spot_vec, label_to_tau, drift_at_labels, vols_at_labels, timeline_years)

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
plt.savefig("fwd_rates_simulated_q.png", dpi=150)
plt.show()
