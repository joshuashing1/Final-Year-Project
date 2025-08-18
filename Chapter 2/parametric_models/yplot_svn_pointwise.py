import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from calibration import calibrate_svn_ptwise


def parse_maturities(labels):
    """Convert labels with 'M' or 'Y' suffixes into years as floats."""
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


def yield_curves_plot(maturities_years, fitted_curves, rmse_values, title, lambd0, save_path):
    """Plot fitted Svensson yield curves across a grid of maturities."""
    x_min = float(np.nanmin(maturities_years))
    x_max = float(np.nanmax(maturities_years))
    x_grid = np.linspace(x_min, x_max, 300)

    fig, ax = plt.subplots(figsize=(12, 5))
    for curve in fitted_curves:
        ax.plot(x_grid, curve(x_grid), linewidth=0.8)

    ax.set_xlabel("Maturity (Years)", fontsize=14)
    ax.set_ylabel("Interest Rate (%)", fontsize=14)
    ax.set_title(title, fontsize=24, fontweight="bold", pad=12)
    ax.set_xlim(left=0, right=x_max)
    ax.set_ylim(-2, 10)

    avg_rmse = float(np.nanmean(rmse_values))
    info = (
        r"• Svensson Fit"
        "\n"
        rf"• Pointwise OLS; $\lambda_{{init}}^{(1)}={lambd0[0]:.1f}$, $\lambda_{{init}}^{(2)}={lambd0[1]:.1f}$"
        "\n"
        f"• Avg. RMSE = {avg_rmse:.4f}"
    )
    ax.text(
        0.64, 0.75, info,
        transform=ax.transAxes,
        fontsize=14,
        bbox=dict(boxstyle="square", facecolor="white", edgecolor="red", linewidth=1.5)
    )

    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved figure to {save_path}")


def process_yield_csv(csv_path: str, title: str, lambd0: tuple[float, float]):
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
        y = y[mask]
        if t.size == 0:
            continue  

        curve, _ = calibrate_svn_ptwise(t, y, lambd0=lambd0)

        fitted_curves.append(curve)
        yhat = curve(t)
        rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))

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

    params_path = f"{title}_svn_pointwise_betas.csv"
    out.to_csv(params_path, index=False)
    print(f"Saved {len(out)} rows to {params_path}")

    fig_path = f"{title}_svn_pointwise_curve.png"
    yield_curves_plot(maturities_years, fitted_curves, out["rmse"], title=title, lambd0=lambd0, save_path=str(fig_path))


def main():
    """Configure the currency datasets to process."""
    list = [
        {"csv_path": r"Chapter 2\Data\USTreasury_Yield_Final.csv", "title": "USD"},
        {"csv_path": r"Chapter 2\Data\CGB_Yield_Final.csv",        "title": "CNY"},
        {"csv_path": r"Chapter 2\Data\GLC_Yield_Final.csv",        "title": "GBP"},
        {"csv_path": r"Chapter 2\Data\SGS_Yield_Final.csv",        "title": "SGD"},
        {"csv_path": r"Chapter 2\Data\ECB_Yield_Final.csv",        "title": "EUR"},
    ]

    for item in list:
        process_yield_csv(item["csv_path"], title=item["title"], lambd0=(2.0, 5.0)) # initial values of lambds

if __name__ == "__main__":
    main()
