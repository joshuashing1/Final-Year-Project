import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def parse_tenor(s: str) -> float:
    s = str(s).strip().upper()
    if s.endswith("M"):
        return float(s[:-1]) / 12.0
    if s.endswith("Y"):
        return float(s[:-1])
    return float(s)


def standardize_params(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (X - mu) / sd


def standardize_inverse(Z: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return Z * sd + mu

def hist_fwd_curves_plot(maturities_years, fitted_curves, title, save_path):
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

def scalar_sigma(Sigma: np.ndarray) -> np.ndarray:
    """
    Sigma: (T, N, K) factor loadings used in the simulation.
    Returns: sig (T, N) where sig[t, n] = sqrt(sum_k Sigma[t, n, k]^2).
    """
    return np.linalg.norm(Sigma, axis=2)

def export_volatility(Sigma: np.ndarray, taus: np.ndarray, labels: list[str], dt: float,
                     save_path: Path = "volatility_evolution.csv") -> Path:
    """
    Build a CSV with rows=time and columns=selected tenor labels.
    Time 't' is in years on a uniform grid: t = 0, dt, 2dt, ...
    """
    # scalarize multi-factor vol to one σ per tenor
    sig = scalar_sigma(Sigma)                  # (T, N)
    T, N = sig.shape

    # map desired labels -> τ grid indices (must be exact matches in taus)
    want = np.array([parse_tenor(x) for x in labels], float)
    idx = [int(np.where(np.isclose(taus, w))[0][0]) for w in want]  # raises if not found

    # slice the nine columns; build time grid in years
    sig_sel = sig[:, idx]                      # (T, 9)
    tgrid = np.arange(1, T + 1, dtype=int)
    
    df = pd.DataFrame(sig_sel, columns=labels)
    df.insert(0, "t", tgrid)                   # y-axis as time (rows)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved σ(t,τ) CSV to {save_path}")
    return save_path

def export_simul_fwd(sim_matrix: np.ndarray, taus: np.ndarray, labels: list[str], dt: float,
                     save_path: Path = Path("simulated_fwd_rates.csv")) -> Path:
    """
    Build a CSV with rows=time and columns=selected tenor labels.
    Time 't' is a step index: 1, 2, ..., T  (decimals; no ×100).
    """
    T, N = sim_matrix.shape
    want = np.array([parse_tenor(x) for x in labels], float)
    idx  = [int(np.where(np.isclose(taus, w))[0][0]) for w in want]  # raises if not found
    sim_sel = sim_matrix[:, idx]                 # (T, len(labels))
    tgrid   = np.arange(1, T + 1, dtype=int)

    df = pd.DataFrame(sim_sel, columns=labels)
    df.insert(0, "t", tgrid)

    save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved simulated f(t,τ) CSV to {save_path}")
    return save_path