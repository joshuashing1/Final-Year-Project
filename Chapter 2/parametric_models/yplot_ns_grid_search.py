import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from calibration import calibrate_ns_grid, ns_grid_error_surface

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

    ax.set_xlabel("Maturity (years) -- Grid Search", fontsize=14)
    ax.set_ylabel("Interest Rate (%)", fontsize=14)
    ax.set_title(title, fontsize=24, fontweight="bold", pad=12)
    ax.set_ylim(-2, 10)
    ax.set_xlim(left=0, right=x_max)

    avg_rmse = float(np.nanmean(rmse_values))
    ax.text(
        0.70, 0.92, f"Avg. RMSE = {avg_rmse:.4f}",
        transform=ax.transAxes,
        fontsize=14,
        bbox=dict(boxstyle="square", facecolor="white", edgecolor="red", linewidth=1.5)
    )

    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved figure to {save_path}")
    
    
def plot_ns_grid_surface(lambdas, maturities, err_matrix, title, save_path):
    """3D surface: x=λ, y=maturity (years), z=per-maturity squared error."""
    # Build matching grids so shapes are (n_lambda, n_maturity)
    L, M = np.meshgrid(lambdas, maturities, indexing='ij')

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(L, M, err_matrix, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_xlabel("λ (decay)")
    ax.set_ylabel("Maturity (years)")
    ax.set_zlabel("Squared error")
    ax.set_title(title, pad=12)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved 3D surface to {save_path}")


def process_yield_csv(csv_path: str, title: str, out_dir: str, lambd0: float = 1.0):
    """Load a single CSV, calibrate per row, save params & plot."""
    df = pd.read_csv(csv_path, header=0)
    maturities_years = parse_maturities(df.columns.tolist())

    results = []
    fitted_curves = []

    for i, row in df.iterrows():
        y = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(y)
        t = maturities_years[mask]
        y = y[mask]

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

    out = pd.DataFrame(results)

    params_path = f"{title}_ns_grid_search_betas.csv"
    out.to_csv(params_path, index=False)
    print(f"Saved {len(out)} rows to {params_path}")

    fig_path = f"{title}_ns_grid_search_curve.png"
    yield_curves_plot(maturities_years, fitted_curves, out["rmse"], title=title, save_path=str(fig_path))


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
    for list in lists:
        process_yield_csv(list["csv_path"], title=list["title"], out_dir=out_dir, lambd0=1.0)

if __name__ == "__main__":
    main()