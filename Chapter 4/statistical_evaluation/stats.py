from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera

ROOT = Path("Chapter 4") / "statistical_evaluation"
FWD_DIR  = ROOT / "simulated_rates_input_csv"
HIST_PATH  = FWD_DIR / "GLC_fwd_curve_selected.csv"
PCA_PATH   = FWD_DIR / "simulated_forward_P_measure.csv"

TIME_COL = "t"
DT_YEARS = 1.0 / 252.0  

hist = pd.read_csv(HIST_PATH).sort_values(TIME_COL).reset_index(drop=True)
pca  = pd.read_csv(PCA_PATH ).sort_values(TIME_COL).reset_index(drop=True)

assert np.array_equal(hist[TIME_COL].to_numpy(), pca[TIME_COL].to_numpy())

tenors = [c for c in hist.columns if c != TIME_COL]

# increments (drop first row later to align horizons)
hist_inc = hist[tenors].diff().iloc[1:].reset_index(drop=True)
pca_inc  =  pca[tenors].diff().iloc[1:].reset_index(drop=True)

# levels aligned for RMSE
hist_lvl = hist[tenors].iloc[1:].reset_index(drop=True)
pca_lvl  =  pca[tenors].iloc[1:].reset_index(drop=True)

def jb_table(df):
    rows = []
    for c in tenors:
        stat, pval = jarque_bera(df[c].to_numpy())
        rows.append((c, stat, pval))
    return pd.DataFrame(rows, columns=["tenor", "JB_stat", "JB_pvalue"])

jb_pca = jb_table(pca_inc)

rmse_pca_per_tenor = np.sqrt(((hist_lvl - pca_lvl)**2).mean(axis=0))

rmse_tbl = pd.DataFrame({
    "tenor": tenors,
    "RMSE_PCA": rmse_pca_per_tenor.values
})
overall_pca = float(rmse_pca_per_tenor.mean())

print("\n=== Jarque–Bera test statistics (PCA model) ===")
print(jb_pca.to_string(index=False))
print("\n=== Predictive RMSE per tenor (levels vs historical) ===")
print(rmse_tbl.to_string(index=False))
print("\nOverall Predictive RMSE:")
print(f"  PCA: {overall_pca:.6f}")
