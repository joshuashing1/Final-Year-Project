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


def yield_curves_plot(maturities_years, fitted_curves, rmse_values, title, save_path, lambd_lo, lambd_upp, n_grid):
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
        r"• Nelson-Siegel Fit"
        "\n"
        f"• Grid Search; $\lambda_i \in [{lambd_lo:.2f},{lambd_upp:.2f}],\ i={n_grid}$"
        "\n"
        f"• Avg. RMSE = {avg_rmse:.4f}"
    )
    ax.text(
        0.61, 0.75, info,
        transform=ax.transAxes,
        fontsize=14,
        bbox=dict(boxstyle="square", facecolor="white", edgecolor="red", linewidth=1.5)
    )

    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved figure to {save_path}")
    

def rmse_lambd_grid(t: np.ndarray, y: np.ndarray, lambds: np.ndarray) -> np.ndarray:
    """RMSE(λ-grid) for a single curve (t, y), refitting betas via OLS at each λ."""
    out = np.empty(lambds.size, dtype=float)
    for j, lambd in enumerate(lambds):
        curve, _ = betas_ns_ols(lambd, t, y)
        yhat = curve(t)
        out[j] = np.sqrt(np.mean((yhat - y) ** 2))
    return out


def plot_rmse_heatmap(lambds: np.ndarray, err_mat: np.ndarray, title: str, save_path: str):
    """Heatmap plot for RMSE(λ-grid) across entire yield dataset.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(
        err_mat,
        aspect="auto",
        origin="lower",
        extent=[lambds[0], lambds[-1], 1, err_mat.shape[0]]
    )
    ax.set_xlabel("λ (Decay)")
    ax.set_ylabel("Yield Curve Index")
    ax.set_title(f"{title}: Root Mean Squared Error over λ-grid")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("RMSE")
    ax.grid(True, color="white", alpha=0.6, linewidth=0.7)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved heatmap to {save_path}")


def process_yield_csv(csv_path: str, title: str, lambd_lo: float, lambd_upp: float, n_grid: int):
    """Load a single yield CSV dataset, calibrate per row, save parameters & plot.
    """
    df = pd.read_csv(csv_path, header=0)
    maturities_years = parse_maturities(df.columns.tolist())

    results = []
    fitted_curves = []
    heatmap_lambds = np.linspace(lambd_lo, lambd_upp, 200)
    rmse_lambda_grid = []

    for i, row in df.iterrows():
        y = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(y)
        t = maturities_years[mask]
        y_valid = y[mask]

        curve, _, _ = calibrate_ns_grid(t, y_valid, lambd_lo, lambd_upp, n_grid)

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
        
        rmse_lambda_grid.append(rmse_lambd_grid(t, y_valid, heatmap_lambds))

    out = pd.DataFrame(results)
    params_path = f"{title}_ns_grid_search_betas.csv"
    out.to_csv(params_path, index=False)
    print(f"Saved {len(out)} rows to {params_path}")

    fig_path = f"{title}_ns_grid_search_curve.png"
    yield_curves_plot(maturities_years, fitted_curves, out["rmse"], title=title, save_path=fig_path, 
        lambd_lo=lambd_lo, lambd_upp=lambd_upp, n_grid=n_grid)
    
    err_heatmap = np.vstack(rmse_lambda_grid)  
    heatmap_path = f"{title}_ns_lambda_rmse_heatmap.png"
    plot_rmse_heatmap(heatmap_lambds, err_heatmap, title=title, save_path=heatmap_path)


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

    for item in lists:
        process_yield_csv(item["csv_path"], title=item["title"], lambd_lo = 0.05, lambd_upp=5.0, n_grid=1000) # λ-grid inputs

if __name__ == "__main__":
    main()