import os
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera

ROOT = r"Chapter 3\statistical_evaluation\simulated_fwd_rates"
HIST_PATH = os.path.normpath(os.path.join(ROOT, "GLC_fwd_curve_selected.csv"))
PCA_PATH  = os.path.normpath(os.path.join(ROOT, "pca_simulated_fwd_rates.csv"))
VAE_PATH  = os.path.normpath(os.path.join(ROOT, "vae_simulated_fwd_rates.csv"))
TIME_COL = "t"

hist = pd.read_csv(HIST_PATH).sort_values(TIME_COL).reset_index(drop=True)
pca  = pd.read_csv(PCA_PATH ).sort_values(TIME_COL).reset_index(drop=True)
vae  = pd.read_csv(VAE_PATH ).sort_values(TIME_COL).reset_index(drop=True)

# Align on common time index
common_t = set(hist[TIME_COL]).intersection(pca[TIME_COL]).intersection(vae[TIME_COL])
hist = hist[hist[TIME_COL].isin(common_t)].sort_values(TIME_COL).reset_index(drop=True)
pca  =  pca[pca[TIME_COL].isin(common_t)].sort_values(TIME_COL).reset_index(drop=True)
vae  =  vae[vae[TIME_COL].isin(common_t)].sort_values(TIME_COL).reset_index(drop=True)

# Tenor columns (assumed identical & ordered)
tenors = [c for c in hist.columns if c != TIME_COL]
assert tenors == list(pca.columns[1:]) == list(vae.columns[1:]), "Tenor columns mismatch."

# Historical is in % -> convert to decimals
hist[tenors] = hist[tenors] / 100.0

# --------------------- Increments / Levels -------------------
hist_inc = hist[tenors].diff().iloc[1:].reset_index(drop=True)
pca_inc  =  pca[tenors].diff().iloc[1:].reset_index(drop=True)
vae_inc  =  vae[tenors].diff().iloc[1:].reset_index(drop=True)

# For RMSE we compare aligned levels (drop first row to match increments horizon)
hist_lvl = hist[tenors].iloc[1:].reset_index(drop=True)
pca_lvl  =  pca[tenors].iloc[1:].reset_index(drop=True)
vae_lvl  =  vae[tenors].iloc[1:].reset_index(drop=True)

def jb_table(inc_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in tenors:
        stat, pval = jarque_bera(inc_df[c].to_numpy())
        rows.append((c, stat, pval))
    return pd.DataFrame(rows, columns=["tenor", "JB_stat", "JB_pvalue"])

jb_pca = jb_table(pca_inc)
jb_vae = jb_table(vae_inc)

rmse_pca_per_tenor = np.sqrt(((hist_lvl - pca_lvl)**2).mean(axis=0)) # rmse
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
