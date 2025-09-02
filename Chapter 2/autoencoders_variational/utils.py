import os, sys
import numpy as np
import pandas as pd
import torch
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
    """
    Returns:
      X: torch.FloatTensor [n_curves, n_maturities]
      maturities_years: torch.FloatTensor [n_maturities]
      tenor_labels: list[str]
    """
    tenor_labels = [str(c) for c in values_df.columns]
    maturities_years_np = parse_maturities(tenor_labels)  # np.ndarray
    X_list = []

    for _, row in values_df.iterrows():
        y = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(y)

        if mask.sum() >= 2:
            # LinearInterpolant works with numpy; keep it and convert later
            interp = LinearInterpolant(maturities_years_np[mask], y[mask])
            y_filled = interp(maturities_years_np)
        elif mask.sum() == 1:
            y_filled = np.full_like(maturities_years_np, y[mask][0], dtype=float)
        else:
            y_filled = np.zeros_like(maturities_years_np, dtype=float)

        X_list.append(y_filled)

    X_np = np.vstack(X_list).astype(np.float32)
    X = torch.from_numpy(X_np)  # [N, M]
    maturities_years = torch.from_numpy(maturities_years_np.astype(np.float32))  # [M]
    return X, maturities_years, tenor_labels


def standardize_fit(X: torch.Tensor):
    """
    X: torch.Tensor [N, M]
    Returns:
      mu: torch.Tensor [1, M]
      sd: torch.Tensor [1, M] (with floor at 1e-8 -> set to 1.0 to avoid divide-by-zero)
    """
    mu = X.mean(dim=0, keepdim=True)
    sd = X.std(dim=0, unbiased=False, keepdim=True)
    sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
    return mu, sd


def standardize_apply(X: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor):
    return (X - mu) / sd


def standardize_inverse(Z: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor):
    return Z * sd + mu


def yield_curves_plot(maturities_years, fitted_curves, title, save_path):
    """Yield curve plot in accordance to tenor structure (PyTorch-friendly)."""
    # maturities_years can be torch.Tensor or np.ndarray; normalize to numpy for plotting
    if isinstance(maturities_years, torch.Tensor):
        maturities_years_np = maturities_years.detach().cpu().numpy()
    else:
        maturities_years_np = np.asarray(maturities_years)

    x_min = float(np.nanmin(maturities_years_np))
    x_max = float(np.nanmax(maturities_years_np))

    # Build x_grid in torch (so torch-based curves can run on GPU if needed), then convert for plotting
    x_grid_t = torch.linspace(x_min, x_max, steps=300)

    DPI = 100
    W_IN = 1573 / DPI
    H_IN = 750  / DPI
    fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)

    for curve in fitted_curves:
        # Support both numpy-based callables (e.g., LinearInterpolant) and torch-based callables
        try:
            # First try torch path
            y_t = curve(x_grid_t)  # expect torch.Tensor
            if isinstance(y_t, torch.Tensor):
                y_np = y_t.detach().cpu().numpy()
                x_np = x_grid_t.detach().cpu().numpy()
            else:
                # If it returned numpy already
                y_np = np.asarray(y_t)
                x_np = x_grid_t.detach().cpu().numpy()
        except Exception:
            # Fallback: assume curve expects numpy
            x_np = x_grid_t.detach().cpu().numpy()
            y_np = curve(x_np)

        ax.plot(x_np, y_np, linewidth=0.8)

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
        r"• Network: AE"
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
