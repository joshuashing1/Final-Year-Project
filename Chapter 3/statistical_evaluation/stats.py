# minimal_stats_md_pathlib.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera

ROOT = Path("Chapter 3") / "statistical_evaluation"
FWD_DIR  = ROOT / "simulated_fwd_rates_csv"
HIST_PATH  = FWD_DIR / "GLC_fwd_curve_selected.csv"
PCA_PATH   = FWD_DIR / "pca_simulated_fwd_rates.csv"
VAE_PATH   = FWD_DIR / "vae_simulated_fwd_rates.csv"

TIME_COL = "t"
DT_YEARS = 1.0 / 252.0  

hist = pd.read_csv(HIST_PATH).sort_values(TIME_COL).reset_index(drop=True)
pca  = pd.read_csv(PCA_PATH ).sort_values(TIME_COL).reset_index(drop=True)
vae  = pd.read_csv(VAE_PATH ).sort_values(TIME_COL).reset_index(drop=True)

assert np.array_equal(hist[TIME_COL].to_numpy(), pca[TIME_COL].to_numpy())
assert np.array_equal(hist[TIME_COL].to_numpy(), vae[TIME_COL].to_numpy())

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

# # modified duration (annual compounding)
# def _tenor_to_years(c: str) -> float:
#     s = c.strip().upper()
#     if s.endswith("M"): return float(s[:-1]) / 12.0
#     if s.endswith("Y"): return float(s[:-1])
#     return float(s)

# tenor_years = np.array([_tenor_to_years(c) for c in tenors], dtype=float)
# dτ = np.diff(tenor_years)

# def _cum_int_over_tenor(F_row: np.ndarray) -> np.ndarray:
#     """Cumulative ∫_0^{τ_j} f(t, t+u) du via trapezoid over tenor grid."""
#     avg = 0.5 * (F_row[:-1] + F_row[1:])  # (J-1,)
#     return np.concatenate([[0.0], np.cumsum(avg * dτ)])  # (J,)

# def _prices_from_forwards(F: np.ndarray) -> np.ndarray:
#     """Return P(t, T_j) for all t,j from forward surface F[t, j]=f(t, τ_j)."""
#     N, J = F.shape
#     P = np.empty_like(F)
#     for i in range(N):
#         I = _cum_int_over_tenor(F[i])     # ∫_0^{τ_j} f(t, t+u) du
#         P[i] = np.exp(-I)
#     return P

# # Forward matrices
# F_pca = pca[tenors].to_numpy(float)  # (N,J)
# F_vae = vae[tenors].to_numpy(float)  # (N,J)

# # Prices from forward surfaces
# P_pca = _prices_from_forwards(F_pca)
# P_vae = _prices_from_forwards(F_vae)

# # Annual-compounded yields from price: P = (1+y)^(-τ)  => y = P^(-1/τ) - 1
# _tau = np.maximum(tenor_years, 1e-12)   # avoid divide-by-zero
# Y_pca = np.power(P_pca, -1.0 / _tau) - 1.0
# Y_vae = np.power(P_vae, -1.0 / _tau) - 1.0

# # Modified Duration (annual comp): D_mod = τ / (1 + y)
# Dmod_pca = tenor_years[None, :] / (1.0 + Y_pca)
# Dmod_vae = tenor_years[None, :] / (1.0 + Y_vae)

# labels9 = ["1M","6M","1.0Y","2.0Y","3.0Y","5.0Y","10.0Y","20.0Y","25.0Y"]
# idx9 = [tenors.index(lbl) for lbl in labels9]

# t_vals = hist[TIME_COL].to_numpy()
# timeline_years = (t_vals - t_vals[0]) * DT_YEARS

# def plot_dmod_side_by_side(D_pca, D_vae, out_png, dpi, w_px, h_px):
#     fig_w_in = (2 * w_px) / dpi
#     fig_h_in = h_px / dpi

#     fig, axes = plt.subplots(1, 2, figsize=(fig_w_in, fig_h_in), dpi=dpi, sharey=True)

#     # 9 distinct, consistent colors across both panels
#     cmap = plt.get_cmap("tab10")
#     tenor_colors = {lbl: cmap(i % 10) for i, lbl in enumerate(labels9)}

#     # Helper to draw one panel
#     def _draw(ax, D, title):
#         for lbl in labels9:
#             j = tenors.index(lbl)
#             ax.plot(
#                 timeline_years,
#                 D[:, j],
#                 lw=2.0,
#                 label=lbl,
#                 color=tenor_colors[lbl],
#             )
#         ax.set_title(title, fontsize=37, fontweight="bold", pad=12)
#         ax.set_xlabel("Time t (years)", fontsize=32)
#         ax.set_ylabel("Modified duration (years)", fontsize=32)
#         ax.grid(True, alpha=0.3)
#         ax.tick_params(axis="both", which="major", labelsize=27)
#         ax.tick_params(axis="both", which="minor", labelsize=27)

#     # Draw both panels
#     _draw(axes[0], D_pca, "Modified Duration — PCA")
#     _draw(axes[1], D_vae, "Modified Duration — VAE")

#     # Single shared legend below both panels
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(
#         handles,
#         labels,
#         loc="lower center",
#         bbox_to_anchor=(0.5, -0.12),  # lower and centered
#         ncol=5,
#         fontsize=20,
#         title="Tenor",
#         title_fontsize=22,
#         frameon=False,
#     )

#     # Adjust layout to reserve space for the legend
#     fig.tight_layout(rect=[0, 0.12, 1, 1])
#     plt.subplots_adjust(bottom=0.22)  # ensures legend area is fully visible

#     plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
#     plt.show()


# plot_dmod_side_by_side(Dmod_pca, Dmod_vae, "modified_duration.png", dpi=100, w_px=1300, h_px=750)
