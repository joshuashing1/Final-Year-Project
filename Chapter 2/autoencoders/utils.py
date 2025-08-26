import os, sys
import numpy as np
import pandas as pd

THIS = os.path.abspath(os.path.dirname(__file__))            
PRJ_ROOT = os.path.abspath(os.path.join(THIS, ".."))
if PRJ_ROOT not in sys.path:
    sys.path.insert(0, PRJ_ROOT)
    
from parametric_models.yplot_historical import parse_maturities, LinearInterpolant

def detect_and_split_dates(df: pd.DataFrame):
    cols = df.columns.tolist()

    def is_tenor(c):
        s = str(c).strip().upper()
        if s.endswith("M") or s.endswith("Y"):
            return True
        try:
            float(s)
            return True
        except Exception:
            return False

    if len(cols) > 0 and not is_tenor(cols[0]):
        return df.iloc[:, 0].astype(str).tolist(), df.iloc[:, 1:].copy()
    return None, df.copy()


def build_dense_matrix(values_df: pd.DataFrame):
    tenor_labels = [str(c) for c in values_df.columns]
    maturities_years = parse_maturities(tenor_labels)
    X_list = []
    for _, row in values_df.iterrows():
        y = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(y)

        if mask.sum() >= 2:
            interp = LinearInterpolant(maturities_years[mask], y[mask])
            y_filled = interp(maturities_years)
        elif mask.sum() == 1:
            y_filled = np.full_like(maturities_years, y[mask][0], dtype=float)
        else:
            y_filled = np.zeros_like(maturities_years, dtype=float)

        X_list.append(y_filled)
    X = np.vstack(X_list).astype(np.float32)
    return X, maturities_years, tenor_labels


def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (X - mu) / sd


def standardize_inverse(Z: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return Z * sd + mu