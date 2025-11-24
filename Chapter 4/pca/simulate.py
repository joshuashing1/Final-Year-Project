"""
This Python script simulates the Hull-Sokol-White SDE using volatility
derived from PCA methodology under P measure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_tenor(s: str) -> float:
    """
    Converts all tenors into years.
    """
    s = s.strip().upper()
    if s.endswith("M"): return float(s[:-1]) / 12.0
    if s.endswith("Y"): return float(s[:-1])
    return float(s)

def polynomial_fit_per_factor(T: np.ndarray, V: np.ndarray, degrees) -> list[np.ndarray]:
    """
    Conduct least squares polynomial fit of each factor's discretized volatility across the tenors.
    """
    if isinstance(degrees, int): degrees = [degrees] * V.shape[1]
    elif len(degrees) == 1:     degrees = [degrees[0]] * V.shape[1]
    return [np.polyfit(T, V[:, i], deg) for i, deg in enumerate(degrees)]

def eval_polys(coeff_list: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    """
    Evaluate the value of a polynomial given an input.
    """
    x = np.asarray(x, float)
    return np.column_stack([np.polyval(c, x) for c in coeff_list])

def drift_computation(tau: float, coeff_list: list[np.ndarray], n_points: int = 500) -> float:
    """
    Compute the drift of the HSW SDE.
    """
    s = 0.0
    grid = np.linspace(0.0, tau, n_points)
    for c in coeff_list:
        vals = np.polyval(c, grid)
        integral = np.trapz(vals, grid)   # approximate integral in drift
        s += integral * np.polyval(c, tau)
    return float(s)

def simulate_path(f0: np.ndarray, tau: np.ndarray, drift_vals: np.ndarray, Sigma: np.ndarray, tgrid: np.ndarray,
    r_mean: float, f_mean: np.ndarray, seed: int = 123):
    """
    Simulate forward rate pathwise using Euler–Maruyama process with Musiela shift with risk premium adjustment.
    """
    f = f0.copy()
    N, P = Sigma.shape
    path = np.empty((len(tgrid), N), float)
    path[0] = f
    rng = np.random.default_rng(seed)
    eps = 1e-12

    for t in range(1, len(tgrid)):
        t_prev = tgrid[t - 1]
        dt     = tgrid[t] - tgrid[t - 1]
        fprev  = f.copy()
        dfdtau = np.gradient(fprev, tau)
        risk_prem = (r_mean - f_mean) / np.maximum(t_prev + tau, eps)
        diffusion = Sigma @ (rng.normal(size=P) * np.sqrt(dt)) 
        f = fprev + (drift_vals + risk_prem + dfdtau) * dt + diffusion
        path[t] = f
    return path

labels  = ['1M','6M','1.0Y','2.0Y','3.0Y','5.0Y','10.0Y','20.0Y','25.0Y']

# simulated Q-measure forward surface and historical short rate
fQ_df = pd.read_csv(r"Chapter 4\data\simulated_fwd_rates_Q_msr.csv").set_index("t").sort_index()
rHist = pd.read_csv(r"Chapter 4\data\short_rate.csv").set_index("t").sort_index()["r_t"]

# Historical GBP forward rates
df_hist = pd.read_csv(r"Chapter 4\data\GLC_fwd_curve_raw.csv").set_index("t").sort_index() / 100.0

# Volatility term-structure (discretized from PCA)
vol_tab = pd.read_csv(r"Chapter 4\data\discretized_volatility.csv")
Tau_vol = vol_tab["Tenor (Years)"].to_numpy(float)
Vols_tab = vol_tab[["Vol1", "Vol2", "Vol3"]].to_numpy(float)

selected_tenors_years = np.array([parse_tenor(x) for x in labels])

# Fit a smooth polynomial over the volatility grid per factor 
coeff_list = polynomial_fit_per_factor(Tau_vol, Vols_tab, degrees=[0, 3, 3])
vols_at_labels  = eval_polys(coeff_list, selected_tenors_years)
drift_at_labels = np.array([drift_computation(tau, coeff_list) for tau in selected_tenors_years])

# align time indexes across datasets
t = fQ_df.index.intersection(rHist.index).intersection(df_hist.index)

fQ_sel = fQ_df.loc[t, labels].to_numpy(float)     
r_sel = rHist.loc[t].to_numpy(float)                   
hist_path = df_hist.loc[t, labels].to_numpy(float)   

print("\nQ-measure forward rates parsed for each tenor (first few rows):")
print(pd.DataFrame(fQ_sel, index=t, columns=labels).head())

annualization = 252.0
timeline_years = (t.to_numpy(float) - t.min()) / annualization

# risk-premium parameters
r_mean = float(r_sel.mean())
f_mean = df_hist.loc[t, labels].to_numpy(float).mean(axis=0)

DPI  = 100
W_IN = 1573 / DPI
H_IN = 750  / DPI

fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)
rHist.loc[t].plot(ax=ax, lw=1.5)
ax.axhline(y=r_mean, linestyle=":", lw=2.0, color="green",
           label=fr"Mean $\overline{{r}}$ = {r_mean:.4f}")
ax.legend(fontsize=20)

ax.set_title("Short Rate $r_t$ (historical)", fontsize=37, fontweight="bold")
TICK_FS = 27
ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
ax.tick_params(axis="both", which="minor", labelsize=TICK_FS)
ax.xaxis.get_offset_text().set_size(TICK_FS)
ax.yaxis.get_offset_text().set_size(TICK_FS)
ax.set_xlabel("Time t (days)", fontsize=32)
ax.set_ylabel(r"$r_t$", fontsize=32)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.savefig("short_rate.png", dpi=DPI)
plt.show()

curve_spot_vec = fQ_sel[0] # initial forward curve

sim_path = simulate_path(f0=curve_spot_vec, tau=selected_tenors_years, drift_vals=drift_at_labels, Sigma=vols_at_labels, tgrid=timeline_years,
    r_mean=r_mean, f_mean=f_mean, seed=42)

out = pd.DataFrame(sim_path, index=t, columns=labels)
out.index.name = "t"
out.to_csv("simulated_forward_P_measure.csv")

fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
axes = axes.ravel()

for j, ax in enumerate(axes):
    ax.plot(timeline_years, hist_path[:, j], label="Historical", lw=1.5)
    ax.plot(timeline_years, sim_path[:, j],  label="Simulated", ls="--", lw=1.0)
    ax.set_title(labels[j], fontsize=16, fontweight="bold")
    ax.set_xlabel("Time t (years)", fontsize=14)
    ax.set_ylabel(r"$f(t, T)$", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(fontsize=12)

fig.suptitle("Forward Rates: Simulated vs Historical (P measure)", fontsize=20,
             fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("fwd_rates_simulated_p.png", dpi=150)
plt.show()