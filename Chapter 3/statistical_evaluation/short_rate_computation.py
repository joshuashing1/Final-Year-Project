import numpy as np
import pandas as pd
from pathlib import Path

PCA_PATH = Path(r"Chapter 3\statistical_evaluation\simulated_fwd_rates\pca_simulated_fwd_rates.csv")
VAE_PATH = Path(r"Chapter 3\statistical_evaluation\simulated_fwd_rates\vae_simulated_fwd_rates.csv")

def tenor_to_years(label: str) -> float:
    """Convert tenor labels like '6M' or '1.5Y' to year fractions."""
    s = str(label).upper().strip()
    if s.endswith("M"):
        return float(s[:-1]) / 12.0
    if s.endswith("Y"):
        return float(s[:-1])
    return float(s)

def short_rate_numpy(csv_path: Path, time_col: str = "t") -> pd.DataFrame:
    """Compute short rate from forward curve CSV using finite differences."""
    df = pd.read_csv(csv_path)
    tenor_cols = [c for c in df.columns if c != time_col]
    T = np.array([tenor_to_years(c) for c in tenor_cols])
    idx = np.argsort(T)
    T, F = T[idx], df[tenor_cols].to_numpy()[:, idx]
    dfdT = np.gradient(F, T, axis=1)
    r_s = F[:, 0] - T[0] * dfdT[:, 0] # linear approx: f(0) = f(T1) - T1f'(T1)

    return pd.DataFrame({time_col: df[time_col], "r_t": r_s})

if __name__ == "__main__":
    pca_short = short_rate_numpy(PCA_PATH)
    pca_short.to_csv("pca_short_rate.csv", index=False)

    vae_short = short_rate_numpy(VAE_PATH)
    vae_short.to_csv("vae_short_rate.csv", index=False)
