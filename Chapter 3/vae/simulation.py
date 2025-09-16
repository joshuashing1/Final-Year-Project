import numpy as np
import pandas as pd

POLYNOM_COEF = "vae_poly_coeffs_scaled.csv"
FWD_RATES = "vae_reconstructed_fwd_rates_scaled.csv"
LATENT_FACTORS = "vae_fwd_latent_space.csv"

def _scale_csv(path: str, out_path: str) -> pd.DataFrame:
    """Helper: read CSV with 't' column, scale numeric cols by 1/100, save."""
    df = pd.read_csv(path)
    if "t" not in df.columns:
        raise ValueError(f"Expected 't' column in {path}")
    scaled = df.copy()
    scaled.iloc[:, 1:] = scaled.iloc[:, 1:] / 100.0
    scaled.to_csv(out_path, index=False)
    print(f"Saved scaled data to {out_path}")
    return scaled

def jacobian(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Derivative
    """
    if values.shape != x.shape:
        raise ValueError("values and x must have the same shape")
    deriv = np.gradient(values, x)
    return deriv

