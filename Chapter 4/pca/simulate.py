import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_tenor(s: str) -> float:
    s = s.strip().upper()
    if s.endswith("M"): return float(s[:-1]) / 12.0
    if s.endswith("Y"): return float(s[:-1])
    return float(s)

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
    """
    s = 0.0
    grid = np.linspace(0.0, tau, n_points)
    for c in coeff_list:
        vals = np.polyval(c, grid)
        integral = np.trapz(vals, grid)   # approximate ∫_0^τ σ(u) du
        s += integral * np.polyval(c, tau)
    return float(s)

def simulate_fwd_path(
    f0: np.ndarray,
    tau: np.ndarray,
    drift_vals: np.ndarray,
    Sigma: np.ndarray,
    tgrid: np.ndarray,
    r_mean: float,
    f_mean: np.ndarray,
    seed: int = 123
):
    """
    Euler–Maruyama with Musiela shift.
    Drift at time t: α(τ) + (r_t - f^Q(t, t+τ))/(t+τ) + ∂f/∂τ.
    vols_at_tau: [N, K] volatility columns per factor evaluated at tau.
    drift_vals: α(τ) vector (length N), time-homogeneous HJM drift from σ-polys.
    r_series, fQ_series must be aligned to tgrid (same length).
    """
    f = f0.copy()
    N, K = Sigma.shape
    path = np.empty((len(tgrid), N), float)
    path[0] = f
    rng = np.random.default_rng(seed)
    eps = 1e-12

    for it in range(1, len(tgrid)):
        t_prev = tgrid[it - 1]
        dt     = tgrid[it] - tgrid[it - 1]

        fprev  = f.copy()
        dfdtau = np.gradient(fprev, tau)

        # Real-world correction term (componentwise over τ grid)
        risk_prem = (r_mean - f_mean) / np.maximum(t_prev + tau, eps)
        
        z = rng.normal(size=K)
        diffusion = Sigma @ (z * np.sqrt(dt))  # shape (N,)

        f = fprev + (drift_vals + risk_prem + dfdtau) * dt + diffusion
        path[it] = f
    return path

LABELS = ['1M','6M','1.0Y','2.0Y','3.0Y','5.0Y','10.0Y','20.0Y','25.0Y']
DAY2YR = 252.0

# Q-measure forward surface (decimals), and historical short rate r_t (decimals)
fQ_df   = pd.read_csv(r"Chapter 4\data\simulated_fwd_rates_Q_msr.csv").set_index("t").sort_index()
rHist   = pd.read_csv(r"Chapter 4\data\short_rate.csv").set_index("t").sort_index()["r_t"]

# Historical GBP forward rates (P data) in PERCENT -> convert to decimals
df_hist = pd.read_csv(r"Chapter 4\data\GLC_fwd_curve_raw.csv").set_index("t").sort_index() / 100.0

# Volatility term-structures table: Tenor (Years), Vol1, Vol2, Vol3
vol_tab  = pd.read_csv(r"Chapter 4\data\discretized_volatility.csv")
Tau_vol  = vol_tab["Tenor (Years)"].to_numpy(float)
Vols_tab = vol_tab[["Vol1","Vol2","Vol3"]].to_numpy(float)

labels    = LABELS
pick_tau  = np.array([parse_tenor(x) for x in labels])  # the 9 kept tenors

tmin, tmax = Tau_vol.min(), Tau_vol.max()

coeff_list      = fit_poly_per_factor(Tau_vol, Vols_tab, degrees=[0, 3, 3])
vols_at_labels  = eval_polys(coeff_list, pick_tau)  # [N, K]
drift_at_labels = np.array([drift_m_tau(t, coeff_list) for t in pick_tau])

t   = fQ_df.index.intersection(rHist.index).intersection(df_hist.index)
fQ_sel     = fQ_df.loc[t, labels].to_numpy(float)   # [T, N]
r_sel      = rHist.loc[t].to_numpy(float)           # [T]
hist_path  = df_hist.loc[t, labels].to_numpy(float) # [T, N]

print("\nQ-measure forward rates parsed for each tenor (first few rows):")
print(pd.DataFrame(fQ_sel, index=t, columns=labels).head())

timeline_years = (t.to_numpy(float) - t.min()) / DAY2YR

r_mean = float(rHist.loc[t].mean())
f_mean = df_hist.loc[t, labels].to_numpy(float).mean(axis=0) # take mean wrt time dimension

DPI = 100
W_IN = 1573 / DPI
H_IN = 750  / DPI

fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)
rHist.loc[t].plot(ax=ax, lw=1.5)
ax.axhline(y=r_mean, linestyle=":", lw=2.0, color="green", label=fr"Mean $\overline{{r}}$ = {r_mean:.4f}")
ax.set_title("Short Rate $r_t$ (historical)", fontsize=37, fontweight="bold")

TICK_FS = 27
ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
ax.tick_params(axis="both", which="minor", labelsize=TICK_FS)
ax.xaxis.get_offset_text().set_size(TICK_FS)
ax.yaxis.get_offset_text().set_size(TICK_FS)
ax.set_xlabel("Time t (years)", fontsize=32)
ax.set_ylabel(r"$r_t$", fontsize=32)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("short_rate.png", dpi=DPI)
plt.show()

curve_spot_vec = fQ_sel[0] # initial forward curve

sim_path = simulate_fwd_path(
    f0=curve_spot_vec,
    tau=pick_tau,
    drift_vals=drift_at_labels,
    vols_at_tau=vols_at_labels,
    tgrid=timeline_years,
    r_mean=r_mean,
    f_mean=f_mean,
    seed=42
)

out = pd.DataFrame(sim_path, index=t, columns=labels)
out.index.name = "t"
out.to_csv("simulated_forward_P_measure.csv")

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
plt.savefig("fwd_rates_simulated_p.png", dpi=150)
plt.show()
