import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from calibration import calibrate_ns_ols
# from calibration import calibrate_svn_ols

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


def main():
    """Calibrate Nelson-Siegel interest rate model using OLS calibrator with root mean squared error statistics.
    Comment this function to calibrate Svensson interest rate model.
    """
    csv_path = r"Chapter 2\Data\USTreasury-Yield-Final.csv"
    df = pd.read_csv(csv_path, header=0)
    maturities_years = parse_maturities(df.columns.tolist())
    results = []
    fitted_curves = []
    for i, row in df.iterrows():
        y = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(y)
        t = maturities_years[mask]
        y = y[mask]
        curve, opt_res = calibrate_ns_ols(t, y, lambd0=1.0)
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
    out = pd.DataFrame(results) # save parameters to CSV
    out_path = "ns_fitted_params.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to {out_path}")\
    
    """Plot in accordance to tenor structure."""
    x_min = float(np.nanmin(maturities_years))
    x_max = float(np.nanmax(maturities_years))
    x_grid = np.linspace(x_min, x_max, 300)

    fig, ax = plt.subplots(figsize=(12, 5))
    for curve in fitted_curves:
        ax.plot(x_grid, curve(x_grid), linewidth=0.8)

    ax.set_xlabel("Maturity (years)", fontsize=14)
    ax.set_ylabel("Interest Rate (%)", fontsize=14)
    ax.set_title("USD", fontsize=24, fontweight="bold", pad=12) # change currency convention
    ax.set_ylim(-2, 10)

    avg_rmse = float(np.nanmean(out["rmse"]))
    ax.text(
        0.70, 0.92, f"Avg. RMSE = {avg_rmse:.4f}",
        transform=ax.transAxes,
        fontsize=14,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="red", linewidth=1.5)
    )

    fig.tight_layout()
    fig_path = "nelson-USD-yield-curve.png"
    plt.savefig(fig_path, dpi=200)
    plt.show()
    print(f"Saved figure to {fig_path}")

if __name__ == "__main__":
    main()
