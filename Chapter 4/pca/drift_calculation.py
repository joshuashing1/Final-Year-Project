import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config / helpers
# -----------------------------
LABELS = ['1M','6M','1.0Y','2.0Y','3.0Y','5.0Y','10.0Y','20.0Y','25.0Y']
DAY2YR = 252.0  # interpret integer t as trading days

def parse_tenor(s: str) -> float:
    s = s.strip().upper()
    if s.endswith("M"): return float(s[:-1]) / 12.0
    if s.endswith("Y"): return float(s[:-1])
    return float(s)

def fit_poly_per_factor(T: np.ndarray, V: np.ndarray, degrees) -> list[np.ndarray]:
    if isinstance(degrees, int): degrees = [degrees] * V.shape[1]
    elif len(degrees) == 1:     degrees = [degrees[0]] * V.shape[1]
    return [np.polyfit(T, V[:, i], deg) for i, deg in enumerate(degrees)]

def eval_polys(coeff_list: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return np.column_stack([np.polyval(c, x) for c in coeff_list])

def drift_alpha_of_tau(tau: float, coeff_list: list[np.ndarray], n_points: int = 400) -> float:
    # α(t,t+τ) = Σ_k (∫_0^τ σ_k(u) du) σ_k(τ)
    s = 0.0
    grid = np.linspace(0.0, tau, n_points)
    for c in coeff_list:
        vals = np.polyval(c, grid)
        integ = np.trapz(vals, grid)
        s += integ * np.polyval(c, tau)
    return float(s)

def simulate_path_EM(
    f0: np.ndarray,
    taus: np.ndarray,
    tgrid_years: np.ndarray,
    alpha_at_taus: np.ndarray,       # length N
    vols_at_taus: np.ndarray,        # [N, K]
    r_series: np.ndarray,            # [T]
    fQ_series: np.ndarray,           # [T, N]
    seed: int = 123
):
    """Euler–Maruyama with Musiela shift; drift(t,τ) = α(τ) + (r_t - fQ(t,τ))/(t+τ) + ∂_τ f_t(τ)."""
    rng = np.random.default_rng(seed)
    N = len(taus); K = vols_at_taus.shape[1]
    path = np.empty((len(tgrid_years), N), float)
    f = f0.astype(float).copy()
    path[0] = f
    for it in range(1, len(tgrid_years)):
        t_prev = tgrid_years[it-1]
        dt = tgrid_years[it] - t_prev
        dfdtau = np.gradient(f, taus)                                # ∂_τ f
        extra  = (r_series[it-1] - fQ_series[it-1]) / (t_prev + np.asarray(taus))
        drift_vec = alpha_at_taus + extra + dfdtau
        z = rng.normal(size=K)
        diffusion = vols_at_taus @ (z * np.sqrt(dt))                 # Σ_k σ_k(τ) ΔW_k
        f = f + drift_vec * dt + diffusion
        path[it] = f
    return path

# -----------------------------
# Load data
# -----------------------------
# Q-measure forward surface (already in decimals)
fQ_df = pd.read_csv(r"Chapter 4\data\simulated_fwd_rates_Q_msr.csv").set_index("t").sort_index()

# Historical short rate r_t (decimals)
rHist = pd.read_csv(r"Chapter 4\data\short_rate.csv").set_index("t").sort_index()["r_t"]

# PCA vol term structures
vol_tab = pd.read_csv(r"Chapter 4\data\discretized_volatility.csv")
Tau_vol = vol_tab["Tenor (Years)"].to_numpy(float)
Vols    = vol_tab[["Vol1","Vol2","Vol3"]].to_numpy(float)  # [M,3]

# Historical GBP forward rates (P data) -- sample you provided is in PERCENT, convert to decimals
gbp_hist_df = pd.read_csv(r"Chapter 4\data\GLC_fwd_curve_raw.csv").set_index("t").sort_index()
gbp_hist_df = gbp_hist_df / 100.0

# -----------------------------
# Prepare tenors & coefficients
# -----------------------------
taus = np.array([parse_tenor(x) for x in LABELS], float)              # 9 tenors
coeff_list    = fit_poly_per_factor(Tau_vol, Vols, degrees=[0, 3, 3])
vols_at_taus  = eval_polys(coeff_list, taus)                          # [N,3]
alpha_at_taus = np.array([drift_alpha_of_tau(t, coeff_list) for t in taus])

# Align timelines across ALL THREE series: Q, r_t, and historical GBP
common_t = fQ_df.index.intersection(rHist.index).intersection(gbp_hist_df.index)

# Slice matrices on the common timeline
fQ_sel    = fQ_df.loc[common_t, LABELS].to_numpy(float)               # [T,N], decimals
r_sel     = rHist.loc[common_t].to_numpy(float)                       # [T], decimals
gbp_sel   = gbp_hist_df.loc[common_t, LABELS].to_numpy(float)         # [T,N], decimals

# Time grid in years
tgrid_yr = (common_t.to_numpy(float) - common_t.min()) / DAY2YR

# Initial curve f(0, τ): take first Q row (you can also choose first GBP row if desired)
f0 = fQ_sel[0]

# -----------------------------
# Simulate under the specified SDE
# -----------------------------
path = simulate_path_EM(
    f0=f0,
    taus=taus,
    tgrid_years=tgrid_yr,
    alpha_at_taus=alpha_at_taus,
    vols_at_taus=vols_at_taus,
    r_series=r_sel,
    fQ_series=fQ_sel,
    seed=42
)

# Save simulated path
out = pd.DataFrame(path, index=common_t, columns=LABELS)
out.index.name = "t"
out.to_csv("simulated_forward_P_measure.csv")
print("Saved simulated path -> simulated_forward_P_measure.csv")

# -----------------------------
# 3×3 subplot: Historical GBP vs Simulated
# -----------------------------
timeline_years = tgrid_yr
hist_path = gbp_sel    # use historical GBP forward rates as the reference
sim_path  = path
labels    = LABELS

fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
for j, ax in enumerate(axes.ravel()):
    ax.plot(timeline_years, hist_path[:, j], label="Historical GBP", lw=1.5)
    ax.plot(timeline_years, sim_path[:, j],  label="Simulated", ls="--", lw=1.0)
    ax.set_title(labels[j], fontsize=16, fontweight="bold")
    ax.set_xlabel("Time t (years)", fontsize=14)
    ax.set_ylabel(r"$f(t, T)$", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(fontsize=10)
fig.suptitle("Forward Rates: Simulated vs Historical GBP", fontsize=20, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("forward_rates_simulated_vs_historical_gbp.png", dpi=150)
plt.show()
