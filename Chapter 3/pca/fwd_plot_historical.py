import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _assert_same_shape(t: np.ndarray, y: np.ndarray) -> None:
    assert t.shape == y.shape, "Mismatching shapes of time and values"


def parse_tenor(s: str) -> float:
    """Return tenor in years from labels like '1M','6M','1.0Y','10.0Y'."""
    s = s.strip().upper()
    if s.endswith("M"):
        return float(s[:-1]) / 12.0
    if s.endswith("Y"):
        return float(s[:-1])
    return float(s)


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


def fwd_curves_plot(maturities_years, fitted_curves, title, save_path):
    """2D plot of many forward/yield curves using a consistent publication style."""
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

    # Reasonable defaults — adapt as needed
    ax.set_xlim(left=0, right=x_max)
    ax.set_ylim(-2, 10)

    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved figure to {save_path}")

def main():
    csv_path = r"Chapter 3\data\GLC_fwd_curve_raw.csv"

    df = pd.read_csv(csv_path)
    df = df.set_index("time")

    tenor_years = pd.Series({c: parse_tenor(c) for c in df.columns})
    tenor_years = tenor_years.sort_values()
    df = df.loc[:, tenor_years.index]

    Tau = tenor_years.to_numpy()
    fitted_curves = []

    df_iter = df

    for i in range(len(df_iter)):
        y_row = df_iter.iloc[i].to_numpy(dtype=float)
        fitted_curves.append(LinearInterpolant(Tau, y_row))

    title = "GBP historical forward curves"
    save_path = "historical_fwd_curves.png"
    fwd_curves_plot(Tau, fitted_curves, title, save_path)


if __name__ == "__main__":
    main()
