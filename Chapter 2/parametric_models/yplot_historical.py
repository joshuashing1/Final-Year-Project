import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from calibration import _assert_same_shape

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


class LinearInterpolant:
    """Piecewise-linear interpolant over (t, y) with flat extrapolation."""
    def __init__(self, t, y):
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        _assert_same_shape(t, y)
        order = np.argsort(t)
        self.t = t[order]
        self.y = y[order]

    def __call__(self, x):
        return np.interp(x, self.t, self.y, left=self.y[0], right=self.y[-1])


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

    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved figure to {save_path}")


def process_yield_csv(csv_path: str, title: str):
    """Load a single yield CSV dataset, build linear interpolants per row, plot.
    """
    df = pd.read_csv(csv_path, header=0)
    maturities_years = parse_maturities(df.columns.tolist())

    fitted_curves = []

    for _, row in df.iterrows():
        y = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(y)
        t = maturities_years[mask]
        y_obs = y[mask]

        curve = LinearInterpolant(t, y_obs)
        fitted_curves.append(curve)

    fig_path = f"{title}_historical_curve.png"
    yield_curves_plot(maturities_years, fitted_curves, title=title, save_path=fig_path)


def main():
    """Configure the currency datasets to process.
    """
    datasets = [
        {"csv_path": r"Chapter 2\Data\USTreasury_Yield_Final.csv", "title": "USD"},
        {"csv_path": r"Chapter 2\Data\CGB_Yield_Final.csv",        "title": "CNY"},
        {"csv_path": r"Chapter 2\Data\GLC_Yield_Final.csv",        "title": "GBP"},
        {"csv_path": r"Chapter 2\Data\SGS_Yield_Final.csv",        "title": "SGD"},
        {"csv_path": r"Chapter 2\Data\ECB_Yield_Final.csv",        "title": "EUR"},
    ]

    for item in datasets:
        process_yield_csv(item["csv_path"], title=item["title"])


if __name__ == "__main__":
    main()
