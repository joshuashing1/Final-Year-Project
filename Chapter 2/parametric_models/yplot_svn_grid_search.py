import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from calibration_CuPy import calibrate_svn_grid_CuPy

def parse_maturities(labels):
    """Convert labels with 'M' or 'Y' to years as floats.
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


def yield_curves_plot(maturities_years, fitted_curves, rmse_values, title, save_path, lambd1_lo, lambd1_upp, lambd2_lo, lambd2_upp, n_grid1, n_grid2):
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
        r"• Svensson Fit"
        "\n"
        f"• Grid Search; $\lambda_1 \in [{lambd1_lo:.2f},{lambd1_upp:.2f}],\ n_1={int(n_grid1)}$"
        "\n"
        f"                 $\lambda_2 \in [{lambd2_lo:.2f},{lambd2_upp:.2f}],\ n_2={int(n_grid2)}$"
        "\n"
        f"• Avg. RMSE = {avg_rmse:.4f}"
    )
    ax.text(
        0.58, 0.72, info,
        transform=ax.transAxes,
        fontsize=14,
        bbox=dict(boxstyle="square", facecolor="white", edgecolor="red", linewidth=1.5)
    )

    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved figure to {save_path}")


def process_yield_csv(csv_path: str, title: str, lambd1_lo: float, lambd1_upp: float, lambd2_lo: float, lambd2_upp: float, n_grid1: int, n_grid2: int):
    """Load a single yield CSV dataset, calibrate per row, save parameters & plot.
    """
    df = pd.read_csv(csv_path, header=0)
    maturities_years = parse_maturities(df.columns.tolist())

    results = []
    fitted_curves = []

    for i, row in df.iterrows():
        y = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(y)
        t = maturities_years[mask]
        y_valid = y[mask]
        if t.size == 0:
            continue

        curve, _, _ = calibrate_svn_grid_CuPy(t, y_valid, lambd1_lo, lambd1_upp, lambd2_lo, lambd2_upp, n_grid1, n_grid2)

        fitted_curves.append(curve)
        yhat = curve(t)
        rmse = float(np.sqrt(np.mean((yhat - y_valid) ** 2)))

        results.append({
            "row_index": i,
            "beta1": float(curve.beta1),
            "beta2": float(curve.beta2),
            "beta3": float(curve.beta3),
            "beta4": float(curve.beta4),
            "lambd1": float(curve.lambd1),
            "lambd2": float(curve.lambd2),
            "rmse": rmse
        })

    out = pd.DataFrame(results)

    params_path = f"{title}_svn_grid_search_betas.csv"
    out.to_csv(params_path, index=False)
    print(f"Saved {len(out)} rows to {params_path}")

    fig_path = f"{title}_svn_grid_search_curve.png"
    yield_curves_plot(maturities_years, fitted_curves, out["rmse"], title=title, save_path=fig_path,
        lambd1_lo=lambd1_lo, lambd1_upp=lambd1_upp, lambd2_lo=lambd2_lo, lambd2_upp=lambd2_upp, n_grid1=n_grid1, n_grid2=n_grid2)


def main():
    """Configure the currency datasets to process (grid search on Svensson)."""
    datasets = [
        {"csv_path": r"Chapter 2\Data\USTreasury_Yield_Final.csv", "title": "USD"},
        {"csv_path": r"Chapter 2\Data\CGB_Yield_Final.csv", "title": "CNY"},
        {"csv_path": r"Chapter 2\Data\GLC_Yield_Final.csv", "title": "GBP"},
        {"csv_path": r"Chapter 2\Data\SGS_Yield_Final.csv", "title": "SGD"},
        {"csv_path": r"Chapter 2\Data\ECB_Yield_Final.csv", "title": "EUR"}
    ]

    for item in datasets:
        process_yield_csv(item["csv_path"], title=item["title"], lambd1_lo=0.05, lambd1_upp=5.00, lambd2_lo=0.05, lambd2_upp=5.00, n_grid1=1000, n_grid2=1000) # λ1 x λ2-grid inputs

if __name__ == "__main__":
    main()
