import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ config ------------------
LABELS = ['1M','6M','1.0Y','2.0Y','3.0Y','5.0Y','10.0Y','20.0Y','25.0Y']
DAY2YR = 252.0

# ------------------ helpers -----------------
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
    g = np.linspace(0.0, tau, n_points)
    s = 0.0
    for c in coeff_list:
        vals = np.polyval(c, g)
        s += np.trapz(vals, g) * np.polyval(c, tau)
    return float(s)

def simulate_path_EM(f0, taus, tgrid_years, alpha_at_taus, vols_at_taus, r_series, fQ_series, seed=42):
    rng = np.random.default_rng(seed)
    N, K = len(taus), vols_at_taus.shape[1]
    path = np.empty((len(tgrid_years), N), float); path[0] = f = f0.astype(float).copy()
    eps = 1e-12
    for it in range(1, len(tgrid_years)):
        t_prev, dt = tgrid_years[it-1], tgrid_years[it]-tgrid_years[it-1]
        dfdtau = np.gradient(f, taus)
        extra  = (r_series[it-1] - fQ_series[it-1]) / np.maximum(t_prev + taus, eps)
        drift  = alpha_at_taus + extra + dfdtau
        z = rng.normal(size=K)
        f = f + drift * dt + vols_at_taus @ (z * np.sqrt(dt))
        path[it] = f
    return path

# ------------------ load data ----------------
fQ_df = pd.read_csv(r"Chapter 4\data\simulated_fwd_rates_Q_msr.csv").set_index("t").sort_index()
rHist = pd.read_csv(r"Chapter 4\data\short_rate.csv").set_index("t").sort_index()["r_t"]
vol_tab = pd.read_csv(r"Chapter 4\data\discretized_volatility.csv")
gbp_hist_df = pd.read_csv(r"Chapter 4\data\GLC_fwd_curve_raw.csv").set_index("t").sort_index() / 100.0

Tau_vol = vol_tab["Tenor (Years)"].to_numpy(float)
Vols    = vol_tab[["Vol1","Vol2","Vol3"]].to_numpy(float)

# ------------------ prep ---------------------
taus = np.array([parse_tenor(x) for x in LABELS], float)
tmin, tmax = Tau_vol.min(), Tau_vol.max()
if not (taus.min() >= tmin - 1e-12 and taus.max() <= tmax + 1e-12):
    raise ValueError("Requested tenors outside volatility data range.")

coeff_list    = fit_poly_per_factor(Tau_vol, Vols, degrees=[0, 3, 3])
vols_at_taus  = eval_polys(coeff_list, taus)
alpha_at_taus = np.array([drift_alpha_of_tau(t, coeff_list) for t in taus])

common_t = fQ_df.index.intersection(rHist.index).intersection(gbp_hist_df.index)
fQ_sel   = fQ_df.loc[common_t, LABELS].to_numpy(float)
r_sel    = rHist.loc[common_t].to_numpy(float)
gbp_sel  = gbp_hist_df.loc[common_t, LABELS].to_numpy(float)
tgrid_yr = (common_t.to_numpy(float) - common_t.min()) / DAY2YR
f0 = fQ_sel[0]  # or gbp_sel[0] if you prefer to start from historical

# ------------------ simulate -----------------
path = simulate_path_EM(f0, taus, tgrid_yr, alpha_at_taus, vols_at_taus, r_sel, fQ_sel, seed=42)
pd.DataFrame(path, index=common_t, columns=LABELS).to_csv("simulated_forward_P_measure.csv")

# ------------------ plot 3x3 -----------------
fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
for j, ax in enumerate(axes.ravel()):
    ax.plot(tgrid_yr, gbp_sel[:, j], label="Historical GBP", lw=1.5)
    ax.plot(tgrid_yr, path[:, j],     label="Simulated", ls="--", lw=1.0)
    ax.set_title(LABELS[j], fontsize=16, fontweight="bold")
    ax.set_xlabel("Time t (years)", fontsize=14)
    ax.set_ylabel(r"$f(t, T)$", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(fontsize=10)
fig.suptitle("Forward Rates: Simulated vs Historical GBP", fontsize=20, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("forward_rates_simulated_vs_historical_gbp.png", dpi=150)
plt.show()
