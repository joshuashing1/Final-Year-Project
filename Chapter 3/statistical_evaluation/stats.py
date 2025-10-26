# minimal_stats_md_pathlib.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera

# ---------- Paths (using pathlib) ----------
ROOT = Path("Chapter 3") / "statistical_evaluation"
FWD_DIR  = ROOT / "simulated_fwd_rates_csv"
SR_DIR   = ROOT / "short_rates_csv"

HIST_PATH  = FWD_DIR / "GLC_fwd_curve_selected.csv"
PCA_PATH   = FWD_DIR / "pca_simulated_fwd_rates.csv"
VAE_PATH   = FWD_DIR / "vae_simulated_fwd_rates.csv"
PCA_R_PATH = SR_DIR  / "pca_short_rate.csv"
VAE_R_PATH = SR_DIR  / "vae_short_rate.csv"

TIME_COL = "t"
DT_YEARS = 1.0 / 252.0  # adjust if your time step is not daily-like

# ---------- Load (no merging) ----------
hist = pd.read_csv(HIST_PATH).sort_values(TIME_COL).reset_index(drop=True)
pca  = pd.read_csv(PCA_PATH ).sort_values(TIME_COL).reset_index(drop=True)
vae  = pd.read_csv(VAE_PATH ).sort_values(TIME_COL).reset_index(drop=True)
pca_r = pd.read_csv(PCA_R_PATH).sort_values(TIME_COL).reset_index(drop=True)
vae_r = pd.read_csv(VAE_R_PATH).sort_values(TIME_COL).reset_index(drop=True)

# Optional sanity checks (no aligning/merging done)
assert np.array_equal(hist[TIME_COL].to_numpy(), pca[TIME_COL].to_numpy())
assert np.array_equal(hist[TIME_COL].to_numpy(), vae[TIME_COL].to_numpy())
assert np.array_equal(hist[TIME_COL].to_numpy(), pca_r[TIME_COL].to_numpy())
assert np.array_equal(hist[TIME_COL].to_numpy(), vae_r[TIME_COL].to_numpy())

# ---------- Columns / basic transforms ----------
tenors = [c for c in hist.columns if c != TIME_COL]
assert tenors == list(pca.columns[1:]) == list(vae.columns[1:]), "Tenor columns mismatch."

# increments (drop first row later to align horizons)
hist_inc = hist[tenors].diff().iloc[1:].reset_index(drop=True)
pca_inc  =  pca[tenors].diff().iloc[1:].reset_index(drop=True)
vae_inc  =  vae[tenors].diff().iloc[1:].reset_index(drop=True)

# levels aligned for RMSE
hist_lvl = hist[tenors].iloc[1:].reset_index(drop=True)
pca_lvl  =  pca[tenors].iloc[1:].reset_index(drop=True)
vae_lvl  =  vae[tenors].iloc[1:].reset_index(drop=True)

def jb_table(df):
    rows = []
    for c in tenors:
        stat, pval = jarque_bera(df[c].to_numpy())
        rows.append((c, stat, pval))
    return pd.DataFrame(rows, columns=["tenor", "JB_stat", "JB_pvalue"])

jb_pca = jb_table(pca_inc)
jb_vae = jb_table(vae_inc)

rmse_pca_per_tenor = np.sqrt(((hist_lvl - pca_lvl)**2).mean(axis=0))
rmse_vae_per_tenor = np.sqrt(((hist_lvl - vae_lvl)**2).mean(axis=0))

rmse_tbl = pd.DataFrame({
    "tenor": tenors,
    "RMSE_PCA": rmse_pca_per_tenor.values,
    "RMSE_VAE": rmse_vae_per_tenor.values
})
overall_pca = float(rmse_pca_per_tenor.mean())
overall_vae = float(rmse_vae_per_tenor.mean())

print("\n=== Jarque–Bera on one-step increments (PCA model) ===")
print(jb_pca.to_string(index=False))
print("\n=== Jarque–Bera on one-step increments (VAE model) ===")
print(jb_vae.to_string(index=False))
print("\n=== Predictive RMSE per tenor (levels vs historical) ===")
print(rmse_tbl.to_string(index=False))
print("\nOverall Predictive RMSE:")
print(f"  PCA: {overall_pca:.6f}")
print(f"  VAE: {overall_vae:.6f}")

# ---------- Martingale defect ----------
def _tenor_to_years(c: str) -> float:
    s = c.strip().upper()
    if s.endswith("M"): return float(s[:-1]) / 12.0
    if s.endswith("Y"): return float(s[:-1])
    return float(s)

# tenor grid (years) and trapezoid widths Δτ_k
tenor_years = np.array([_tenor_to_years(c) for c in tenors], dtype=float)
dτ = np.diff(tenor_years)  # length J-1

# Discount factor D(0,t) from short-rate paths
A_pca = np.cumsum(pca_r["r_t"].to_numpy()) * DT_YEARS
A_vae = np.cumsum(vae_r["r_t"].to_numpy()) * DT_YEARS
D0t_pca = np.exp(-A_pca)[:, None]  # (N_time, 1)
D0t_vae = np.exp(-A_vae)[:, None]

# Trapezoid integral over *tenor* at fixed t:
# I_t(τ_j) = ∫_0^{τ_j} f(t, t+u) du, using the grid values f(t, τ_k)
def tenor_integrals_trapz(F_row: np.ndarray) -> np.ndarray:
    avg = 0.5 * (F_row[:-1] + F_row[1:])   # length J-1
    cum = np.concatenate([[0.0], np.cumsum(avg * dτ)])
    return cum  # length J

# P(0,T) from the initial *historical* curve (first row of hist)
f0 = hist[tenors].iloc[0].to_numpy(dtype=float)   # shape (J,)
I0 = tenor_integrals_trapz(f0)                    # (J,)
P0T = np.exp(-I0)[None, :]                        # (1, J)

# P(t,T) from simulated forward surfaces
F_pca = pca[tenors].to_numpy(dtype=float)         # (N, J)
F_vae = vae[tenors].to_numpy(dtype=float)         # (N, J)

def all_PtT(F: np.ndarray) -> np.ndarray:
    N, J = F.shape
    out = np.empty_like(F)
    for i in range(N):
        Ii = tenor_integrals_trapz(F[i])
        out[i] = np.exp(-Ii)
    return out

PtT_pca = all_PtT(F_pca)                          # (N, J)
PtT_vae = all_PtT(F_vae)                          # (N, J)

# Martingale processes and defects
M_pca = D0t_pca * (PtT_pca / P0T)                  # (N, J)
M_vae = D0t_vae * (PtT_vae / P0T)                  # (N, J)

MD_pca = 1.0 - M_pca                               # (N, J), tenor-dependent
MD_vae = 1.0 - M_vae

# Time axis (years)
t_vals = hist[TIME_COL].to_numpy()
timeline_years = (t_vals - t_vals[0]) * DT_YEARS

labels9 = ["1M","6M","1.0Y","2.0Y","3.0Y","5.0Y","10.0Y","20.0Y","25.0Y"]
j_sel = [tenors.index(lbl) for lbl in labels9]

def plot_md_grid_tenor(md_matrix, model_name, out_png):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
    for j, ax in enumerate(axes.ravel()):
        ax.plot(timeline_years, md_matrix[:, j_sel[j]], lw=1.3, label=f"{model_name} defect")
        ax.axhline(0.0, ls="--", lw=0.8)  # ideal martingale baseline
        ax.set_title(labels9[j], fontsize=13, fontweight="bold")
        ax.set_xlabel("Time t (years)", fontsize=11)
        ax.set_ylabel("Martingale defect", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    fig.suptitle(f"Martingale Defect vs Tenor — {model_name}", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_png, dpi=150)
    plt.show()

plot_md_grid_tenor(MD_pca, "PCA-HJM", "martingale_defect_pca.png")
plot_md_grid_tenor(MD_vae, "VAE-HJM", "martingale_defect_vae.png")