"""
This Python script plots the historical forward curves of the GBP.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utility_functions.utils import parse_tenor  

def _assert_same_shape(t: np.ndarray, y: np.ndarray) -> None:
    """
    Checking mismatch of shapes of time and values.
    """
    assert t.shape == y.shape

class LinearInterpolant:
    """
    Piecewise-linear interpolant over (t, y) with flat extrapolation.
    """
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
    """
    2D plot of forward curves.
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

    ax.set_xlim(left=0, right=x_max)
    ax.set_ylim(-2, 10)

    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved figure to {save_path}")

def fwd_curves_evolution(maturities_years, X, title, save_path, t_index=None):
    """
    3D surface of forward rate history.
    """
    X = np.asarray(X, dtype=float)
    n_times, n_tenors = X.shape
    if t_index is None:
        t_index = np.arange(n_times, dtype=float)
    else:
        t_index = np.asarray(t_index, dtype=float)

    T = np.asarray(maturities_years, dtype=float)
    TT, TTIME = np.meshgrid(T, t_index)

    DPI = 100
    W_IN = 1573 / DPI
    H_IN = 750  / DPI
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        TT, TTIME, X,
        cmap="inferno",
        linewidth=0.15,
        antialiased=True,
        shade=True,
        rcount=min(300, n_times),
        ccount=min(200, n_tenors),
        alpha=0.97,
    )

    ax.set_xlabel("Tenor T (years)", fontsize=22, labelpad=12)
    ax.set_ylabel("Time t (index)", fontsize=22, labelpad=12)
    ax.set_zlabel("Forward rate f(t, T)", fontsize=22, labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.tick_params(axis="both", which="minor", labelsize=16)
    ax.view_init(elev=28, azim=-55) 
    fig.suptitle(title, fontsize=48, fontweight="bold", y=0.95)
    cbar = fig.colorbar(surf, shrink=0.8, aspect=24, pad=0.08)
    cbar.set_label("f(t, T)", fontsize=18)

    zmin, zmax = np.nanmin(X), np.nanmax(X)
    ax.set_zlim(zmin, zmax)

    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved 3D surface to {save_path}")

def main():
    csv_path = r"Chapter 3\data\GLC_fwd_curve_raw.csv"
    df = pd.read_csv(csv_path)

    if "t" in df.columns:
        t_index = df["t"].to_numpy()
        df = df.set_index("t")
    else:
        t_index = np.arange(len(df), dtype=int)

    tenor_years = pd.Series({c: parse_tenor(c) for c in df.columns})
    tenor_years = tenor_years.sort_values()
    df = df.loc[:, tenor_years.index]

    Tau = tenor_years.to_numpy()

    fitted_curves = [LinearInterpolant(Tau, df.iloc[i].to_numpy(dtype=float))
                     for i in range(len(df))]

    fwd_curves_plot(
        Tau, fitted_curves,
        title="GBP reconstructed forward curves",
        save_path="historical_fwd_curves.png",
    )

    fwd_curves_evolution(
        maturities_years=Tau,
        X=df.to_numpy(dtype=float),
        title="Historical forward rate",
        save_path="historical_fwd_surface.png",
        t_index=t_index,
    )

if __name__ == "__main__":
    main()
