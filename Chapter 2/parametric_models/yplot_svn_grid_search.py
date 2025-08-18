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

    avg_rmse = float(np.nanmean(rmse_values))
    info = (
        r"• Svensson Fit"
        "\n"
        f"• Grid Search; $\lambda_{{i}}^{1} \in [{lambd1_lo:.2f},{lambd1_upp:.2f}]$"
        "\n"
        f"                        $\lambda_{{j}}^{2} \in [{lambd2_lo:.2f},{lambd2_upp:.2f}]$"
        "\n"
        f"• Avg. RMSE = {avg_rmse:.4f}"
    )
    ax.text(
        0.58, 0.68, info,
        transform=ax.transAxes,
        fontsize=25,
        bbox=dict(boxstyle="square", facecolor="white", edgecolor="red", linewidth=1.5)
    )

    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved figure to {save_path}")


def rmse_radius_profile_from_surface(sse_grid: np.ndarray,
                                     l1s: np.ndarray,
                                     l2s: np.ndarray,
                                     n_obs: int,
                                     r_bins: np.ndarray,
                                     agg: str = "min") -> np.ndarray:
    """Collapse 2D (λ1,λ2) SSE grid to 1D RMSE vs radius bins (min or mean within each bin)."""
    L1, L2 = np.meshgrid(l1s, l2s, indexing="ij")
    r = np.sqrt(L1**2 + L2**2)
    rmse = np.sqrt(sse_grid / float(n_obs))

    r_idx = np.digitize(r.ravel(), r_bins) - 1  # 0..len(r_bins)-2
    rmse_flat = rmse.ravel()
    prof = np.full(r_bins.size - 1, np.nan, dtype=float)
    for k in range(prof.size):
        mask = (r_idx == k)
        if not np.any(mask):
            continue
        vals = rmse_flat[mask]
        prof[k] = np.nanmin(vals) if agg == "min" else np.nanmean(vals)
    return prof


def plot_rmse_radius_heatmap(r_centers: np.ndarray, err_mat: np.ndarray, title: str, save_path: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(
        err_mat,
        aspect="auto",
        origin="lower",
        extent=[r_centers[0], r_centers[-1], 1, err_mat.shape[0]]
    )
    ax.set_xlabel(r"$\|\boldsymbol{\lambda}\|_2$  (Euclidean radius)", fontsize=22)
    ax.set_ylabel("Yield Curve Index", fontsize=22)
    ax.set_title(f"{title}: RMSE over λ-radius for Svensson", fontsize=25)
    ax.tick_params(axis="both", which="both", labelsize=20)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("RMSE", fontsize=22)
    cbar.ax.tick_params(labelsize=20)
    
    ax.grid(True, color="white", alpha=0.6, linewidth=0.7)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved heatmap to {save_path}")



def process_yield_csv(csv_path: str, title: str, lambd1_lo: float, lambd1_upp: float, lambd2_lo: float, lambd2_upp: float, n_grid1: int, n_grid2: int):
    """Load a single yield CSV dataset, calibrate per row, save parameters & plot.
    """
    df = pd.read_csv(csv_path, header=0)
    maturities_years = parse_maturities(df.columns.tolist())

    results = []
    fitted_curves = []
    
    r_min = 0.0
    r_max = np.sqrt(lambd1_upp**2 + lambd2_upp**2)
    n_radius = 200
    r_bins = np.linspace(r_min, r_max, n_radius + 1)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    radius_profiles = []

    for i, row in df.iterrows():
        y = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(y)
        t = maturities_years[mask]
        y_valid = y[mask]
        if t.size == 0:
            continue

        curve, _, _, (sse_grid, l1s, l2s) = calibrate_svn_grid_CuPy(t, y_valid, lambd1_lo, lambd1_upp, lambd2_lo, lambd2_upp, n_grid1, n_grid2, return_surface=True)

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
        
        prof = rmse_radius_profile_from_surface(
            sse_grid, l1s, l2s, n_obs=len(t), r_bins=r_bins, agg="min"  # or "mean"
        )
        radius_profiles.append(prof)

    out = pd.DataFrame(results)

    params_path = f"{title}_svn_grid_search_betas.csv"
    out.to_csv(params_path, index=False)
    print(f"Saved {len(out)} rows to {params_path}")

    fig_path = f"{title}_svn_grid_search_curve.png"
    yield_curves_plot(maturities_years, fitted_curves, out["rmse"], title=title, save_path=fig_path,
        lambd1_lo=lambd1_lo, lambd1_upp=lambd1_upp, lambd2_lo=lambd2_lo, lambd2_upp=lambd2_upp, n_grid1=n_grid1, n_grid2=n_grid2)
    
    err_heatmap = np.vstack(radius_profiles)   
    heatmap_path = f"{title}_svn_lambda_radius_rmse_heatmap.png"
    plot_rmse_radius_heatmap(r_centers, err_heatmap, title=title, save_path=heatmap_path)


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
