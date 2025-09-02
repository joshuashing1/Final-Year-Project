import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def yield_curves_plot(maturities_years, fitted_curves, title, save_path):
    """Yield curve plot in accordance to tenor structure.
    """
    x_min = float(np.nanmin(maturities_years))
    x_max = float(np.nanmax(maturities_years))
    x_grid = np.linspace(x_min, x_max, 300)

    DPI = 100
    W_IN = 1573 / DPI
    H_IN = 750  / DPI
    fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)
    for curve in fitted_curves:
        ax.plot(x_grid, curve(x_grid), linewidth=0.8)

    TICK_FS = 27
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_FS)

    ax.xaxis.get_offset_text().set_size(TICK_FS)
    ax.yaxis.get_offset_text().set_size(TICK_FS)

    ax.set_xlabel("Maturity (Years)", fontsize=32)
    ax.set_ylabel("Interest Rate (%)", fontsize=32)
    ax.set_title(title, fontsize=37, fontweight="bold", pad=12)
    ax.set_ylim(-2, 10)
    ax.set_xlim(left=0, right=x_max)
    
    info = (
        r"• Network: AE" # change description accordingly
        "\n"
        r"• Latent Dim: 13"
    )
    ax.text(
        0.76, 0.80, info,
        transform=ax.transAxes,
        fontsize=25,
        bbox=dict(boxstyle="square", facecolor="white", edgecolor="darkblue", linewidth=1.5)
    )

    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved figure to {save_path}")