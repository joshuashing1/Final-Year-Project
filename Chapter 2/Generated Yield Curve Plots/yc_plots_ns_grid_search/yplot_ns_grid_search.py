import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from calibration import calibrate_ns_grid, betas_ns_ols

def parse_maturities(labels):
    """Convert labels 'M', 'Y' to years as floats.
    """
    out = []
    for s in labels:
        s = str(s).strip().upper()
        if s.endswith('M'):
            out.append(float(s[:-1]) / 12.0)
        elif s.endswith('Y'):
            out.append(float(s[:-1]))
        else:
            # fallback: try as-is (already numeric years)
            out.append(float(s))
    return np.array(out, dtype=float)


def yield_curves_plot(maturities_years, fitted_curves, rmse_values, title, save_path):
    """Yield curve plot in accordance to tenor structure.
    """
    x_min = float(np.nanmin(maturities_years))
    x_max = float(np.nanmax(maturities_years))
    x_grid = np.linspace(x_min, x_max, 300)

    fig, ax = plt.subplots(figsize=(12, 5))
    for curve in fitted_curves:
        ax.plot(x_grid, curve(x_grid), linewidth=0.8)

    ax.set_xlabel("Maturity (Years)", fontsize=14)
    ax.set_ylabel("Interest Rate (%)", fontsize=14)
    ax.set_title(title, fontsize=24, fontweight="bold", pad=12)
    ax.set_ylim(-2, 10)
    ax.set_xlim(left=0, right=x_max)

    avg_rmse = float(np.nanmean(rmse_values))
    info = (
        r"• Grid Search; $\lambda_{i} \in [0.05,5.0], i=1000$"
        "\n"
        f"• Avg. RMSE = {avg_rmse:.4f}"
    )
    ax.text(
        0.70, 0.80, info,
        transform=ax.transAxes,
        fontsize=14,
        bbox=dict(boxstyle="square", facecolor="white", edgecolor="red", linewidth=1.5)
    )

    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved figure to {save_path}")
    

def rmse_over_lambdas(t: np.ndarray, y: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """RMSE(λ) for a single curve (t, y), refitting betas via OLS at each λ."""
    out = np.empty(lambdas.size, dtype=float)
    for j, lambd in enumerate(lambdas):
        curve, _ = betas_ns_ols(lambd, t, y)
        yhat = curve(t)
        out[j] = np.sqrt(np.mean((yhat - y) ** 2))
    return out


def plot_lambda_error_heatmap(lambdas: np.ndarray, err_mat: np.ndarray, title: str, save_path: str):
    """
    2D heatmap:
      x-axis: lambdas (size n_grid)
      y-axis: dataset row (size n_rows)
      color: RMSE
    err_mat must be shape (n_rows, n_grid).
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(
        err_mat,
        aspect="auto",
        origin="lower",
        extent=[lambdas[0], lambdas[-1], 1, err_mat.shape[0]]
    )
    ax.set_xlabel("λ (Decay)")
    ax.set_ylabel("Yield Curve Index")
    ax.set_title(f"{title}: Root Mean Squared Error over λ")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("RMSE")
    ax.grid(True, color="white", alpha=0.6, linewidth=0.7)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved heatmap to {save_path}")


def process_yield_csv(csv_path: str, title: str, out_dir: str, lambd0: float = 1.0):
    """Load a single CSV, calibrate per row, save params & plot."""
    df = pd.read_csv(csv_path, header=0)
    maturities_years = parse_maturities(df.columns.tolist())

    results = []
    fitted_curves = []
    lambdas_for_heatmap = np.linspace(0.05, 5.0, 200)
    rmse_lambda_rows = []

    for i, row in df.iterrows():
        y = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(y)
        t = maturities_years[mask]
        y_valid = y[mask]

        curve, _, _ = calibrate_ns_grid(t, y)

        fitted_curves.append(curve)
        yhat = curve(t)
        rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))

        results.append({
            "row_index": i,
            "beta1": float(curve.beta1),
            "beta2": float(curve.beta2),
            "beta3": float(curve.beta3),
            "lambd": float(curve.lambd),
            "rmse": rmse
        })
        
        rmse_lambda_rows.append(rmse_over_lambdas(t, y_valid, lambdas_for_heatmap))

    out = pd.DataFrame(results)
    params_path = f"{title}_ns_grid_search_betas.csv"
    out.to_csv(params_path, index=False)
    print(f"Saved {len(out)} rows to {params_path}")

    fig_path = f"{title}_ns_grid_search_curve.png"
    yield_curves_plot(maturities_years, fitted_curves, out["rmse"], title=title, save_path=str(fig_path))
    
    err_heatmap = np.vstack(rmse_lambda_rows)  # shape (n_rows, n_grid)
    heatmap_path = f"{title}_ns_lambda_rmse_heatmap.png"
    plot_lambda_error_heatmap(lambdas_for_heatmap, err_heatmap, title=title, save_path=heatmap_path)


def main():
    """Configure the currency datasets to process.
    """
    lists = [
        {"csv_path": r"Chapter 2\Data\USTreasury_Yield_Final.csv", "title": "USD"},
        {"csv_path": r"Chapter 2\Data\CGB_Yield_Final.csv", "title": "CNY"},
        {"csv_path": r"Chapter 2\Data\GLC_Yield_Final.csv", "title": "GBP"},
        {"csv_path": r"Chapter 2\Data\SGS_Yield_Final.csv", "title": "SGD"},
        {"csv_path": r"Chapter 2\Data\ECB_Yield_Final.csv", "title": "EUR"}
    ]

    out_dir = "ns_outputs"
    for item in lists:
        process_yield_csv(item["csv_path"], title=item["title"], out_dir=out_dir, lambd0=1.0)

if __name__ == "__main__":
    main()