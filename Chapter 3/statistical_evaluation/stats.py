# pip install numpy pandas scipy scikit-learn
import os
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from sklearn.metrics import mean_squared_error

# ========================= Config (EDIT ROOT) =========================
# Use a raw string and os.path.join to avoid backslash-escape issues on Windows.
ROOT = r"Chapter 3\statistical_evaluation\simulated_fwd_rates"
HIST_PATH = os.path.normpath(os.path.join(ROOT, "GLC_fwd_curve_raw.csv"))
PCA_PATH  = os.path.normpath(os.path.join(ROOT, "pca_simulated_fwd_rates.csv"))
VAE_PATH  = os.path.normpath(os.path.join(ROOT, "vae_simulated_fwd_rates.csv"))
TIME_COL = "t"

# ============================== Helpers ================================
def tenor_to_years(label: str) -> float:
    s = label.strip().upper()
    if s.endswith("M"):  return float(s[:-1]) / 12.0
    if s.endswith("Y"):  return float(s[:-1])
    return float(s)

def sorted_common_tenors(*dfs, time_col="t"):
    cols = None
    for df in dfs:
        c = set(df.columns) - {time_col}
        cols = c if cols is None else (cols & c)
    if not cols:
        raise ValueError("No common tenor columns across files.")
    return sorted(cols, key=tenor_to_years)

def make_increments(df: pd.DataFrame, cols):
    inc = df[cols].diff()
    return inc.iloc[1:].reset_index(drop=True)

def jb_per_tenor(inc_df: pd.DataFrame, tenors):
    rows = []
    for c in tenors:
        x = inc_df[c].to_numpy()
        x = x[np.isfinite(x)]
        if x.size < 8:
            rows.append((c, np.nan, np.nan))
        else:
            stat, p = jarque_bera(x)
            rows.append((c, stat, p))
    return pd.DataFrame(rows, columns=["tenor", "JB_stat", "JB_pvalue"])

def rmse_per_tenor(true_df: pd.DataFrame, pred_df: pd.DataFrame, tenors):
    rows = []
    for c in tenors:
        y_true = true_df[c].to_numpy()
        y_pred = pred_df[c].to_numpy()
        m = np.isfinite(y_true) & np.isfinite(y_pred)
        r = np.sqrt(mean_squared_error(y_true[m], y_pred[m])) if m.any() else np.nan
        rows.append((c, r))
    out = pd.DataFrame(rows, columns=["tenor", "RMSE"])
    return out, float(np.nanmean(out["RMSE"].to_numpy()))

# ============================== Load ===================================
hist = pd.read_csv(HIST_PATH)
pca  = pd.read_csv(PCA_PATH)
vae  = pd.read_csv(VAE_PATH)

for df, name in ((hist,"historical"), (pca,"pca"), (vae,"vae")):
    if TIME_COL not in df.columns:
        raise ValueError(f"Missing '{TIME_COL}' in {name} file.")
    df.sort_values(TIME_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)

# Align on common time index
common_t = set(hist[TIME_COL]).intersection(pca[TIME_COL]).intersection(vae[TIME_COL])
hist = hist[hist[TIME_COL].isin(common_t)].sort_values(TIME_COL).reset_index(drop=True)
pca  = pca [pca [TIME_COL].isin(common_t)].sort_values(TIME_COL).reset_index(drop=True)
vae  = vae [vae [TIME_COL].isin(common_t)].sort_values(TIME_COL).reset_index(drop=True)

# Common tenors (sorted by maturity)
tenors = sorted_common_tenors(hist, pca, vae, time_col=TIME_COL)
hist = hist[[TIME_COL] + tenors]
pca  = pca [[TIME_COL] + tenors]
vae  = vae [[TIME_COL] + tenors]

# ===================== Normalize (divide historical by 100) =====================
hist[tenors] = hist[tenors] / 100.0

# ===================== Increments & aligned levels =====================
hist_inc = make_increments(hist, tenors)
pca_inc  = make_increments(pca, tenors)
vae_inc  = make_increments(vae, tenors)

# For RMSE we compare levels at t (after diff drop)
hist_lvl = hist[tenors].iloc[1:].reset_index(drop=True)
pca_lvl  = pca [tenors].iloc[1:].reset_index(drop=True)
vae_lvl  = vae [tenors].iloc[1:].reset_index(drop=True)

# ===================== Jarque–Bera (per tenor) =====================
jb_pca = jb_per_tenor(pca_inc, tenors)
jb_vae = jb_per_tenor(vae_inc, tenors)

# ===================== Predictive RMSE (per tenor & overall) =====================
rmse_pca, overall_pca = rmse_per_tenor(hist_lvl, pca_lvl, tenors)
rmse_vae, overall_vae = rmse_per_tenor(hist_lvl, vae_lvl, tenors)

# ============================== Output ===============================
print("\n=== Jarque–Bera on one-step increments (PCA model) ===")
print(jb_pca.to_string(index=False))

print("\n=== Jarque–Bera on one-step increments (VAE model) ===")
print(jb_vae.to_string(index=False))

rmse_tbl = rmse_pca.merge(rmse_vae, on="tenor", suffixes=("_PCA","_VAE")) \
                   .sort_values("tenor", key=lambda s: s.map(tenor_to_years))
print("\n=== Predictive RMSE per tenor (levels vs historical) ===")
print(rmse_tbl.to_string(index=False))

print("\nOverall Predictive RMSE:")
print(f"  PCA: {overall_pca:.6f}")
print(f"  VAE: {overall_vae:.6f}")
