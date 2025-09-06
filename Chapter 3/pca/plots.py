# hjm_pca_drift.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------
# Utilities
# --------------------------
def parse_tenor(s: str) -> float:
    """
    Parse tenor labels like '1M', '6M', '1.0Y', '2.5Y' into years (float).
    If the label is already numeric (e.g. '0.5'), it's interpreted as years.
    """
    s = str(s).strip()
    if s.endswith(("M", "m")):
        return float(s[:-1]) / 12.0
    if s.endswith(("Y", "y")):
        return float(s[:-1])
    # Fallback: try plain float
    return float(s)


class PolynomialInterpolator:
    def __init__(self, params: np.ndarray):
        # params is highest-power first (np.polyfit's convention)
        if not isinstance(params, np.ndarray):
            params = np.asarray(params, dtype=float)
        self.params = params.astype(float)

    def calc(self, x: float) -> float:
        # Evaluate polynomial with coefficients in descending powers
        # Equivalent to np.polyval(self.params, x)
        return float(np.polyval(self.params, x))


def integrate(f, x0: float, x1: float, dx: float) -> float:
    """Simple trapezoidal rule with uniform spacing."""
    n = int((x1 - x0) / dx) + 1
    xs = np.linspace(x0, x1, n)
    ys = np.array([f(x) for x in xs], dtype=float)
    return np.trapz(ys, xs)


# --------------------------
# Main
# --------------------------
def main():
    np.random.seed(0)

    # --- Load data ---
    df = pd.read_csv("data\GLC_fwd_curve_raw.csv").set_index("time")
    # The original notebook divided by 100 to convert % to decimals
    df = df / 100.0

    # Align columns by increasing tenor
    tenor_map = {c: parse_tenor(c) for c in df.columns}
    cols_sorted = sorted(df.columns, key=lambda c: tenor_map[c])
    df = df[cols_sorted]
    tenors = np.array([tenor_map[c] for c in cols_sorted], dtype=float)

    # Matrices
    hist_rates = df.to_numpy(dtype=float)          # shape [n_time, n_tenor]
    T_idx = df.index.to_numpy()                    # just for reference/plotting if needed

    # --- Quick sanity prints ---
    print("Data shape:", hist_rates.shape)
    print("Tenors (years):", tenors)

    # --- Plots: Historical curves ---
    plt.figure(figsize=(12, 4))
    plt.plot(hist_rates)
    plt.xlabel(r"Time index")
    plt.title(r"Historical $f(t,\tau)$ by $t$")
    plt.tight_layout()

    plt.figure(figsize=(12, 4))
    plt.plot(tenors, hist_rates.T)
    plt.xlabel(r"Tenor $\tau$ (years)")
    plt.title(r"Historical $f(t,\tau)$ by $\tau$")
    plt.tight_layout()

    # --- Differences along time: df(t, τ) ---
    diff_rates = np.diff(hist_rates, axis=0)
    assert hist_rates.shape[1] == diff_rates.shape[1]

    plt.figure(figsize=(12, 4))
    plt.plot(diff_rates)
    plt.xlabel(r"Time index")
    plt.title(r"$df(t,\tau)$ by $t$")
    plt.tight_layout()

    # --- PCA: covariance across tenors (annualized) ---
    sigma = np.cov(diff_rates.T) * 252.0
    print("Sigma shape:", sigma.shape)

    # Eigen-decomposition (symmetric matrix -> eigh)
    eigval_all, eigvec_all = np.linalg.eigh(sigma)
    # Sort descending
    order_desc = np.argsort(eigval_all)[::-1]
    eigval_all = eigval_all[order_desc]
    eigvec_all = eigvec_all[:, order_desc]

    print("All eigenvalues (desc):")
    print(eigval_all)

    # Choose number of factors
    factors = 3
    princ_eigval = eigval_all[:factors]
    princ_comp = eigvec_all[:, :factors]  # columns are principal components

    print("\nPrincipal eigenvalues (top-k):")
    print(princ_eigval)
    print("\nPrincipal components (columns = PCs):")
    print(princ_comp)

    # Plot principal components vs tenor (typical interpretation)
    plt.figure(figsize=(10, 4))
    for i in range(princ_comp.shape[1]):
        plt.plot(tenors, princ_comp[:, i], marker='.', label=f"PC{i+1}")
    plt.title("Principal components (eigenvectors over tenor)")
    plt.xlabel(r"Tenor $\tau$ (years)")
    plt.ylabel("Loading")
    plt.legend()
    plt.tight_layout()

    # --- Discretized volatility functions from PCs ---
    # vols[:, i] = sqrt(lambda_i) * PC_i(tenor)
    sqrt_eigval = np.sqrt(princ_eigval)               # shape (k,)
    vols = princ_comp * sqrt_eigval[np.newaxis, :]    # broadcast to [n_tenor, k]
    print("vols shape:", vols.shape)

    plt.figure(figsize=(10, 4))
    for i in range(vols.shape[1]):
        plt.plot(tenors, vols[:, i], marker='.', label=f"Vol {i+1}")
    plt.xlabel(r"Tenor $\tau$ (years)")
    plt.ylabel(r"Volatility $\sigma$")
    plt.title("Discretized volatilities")
    plt.legend()
    plt.tight_layout()

    # --- Volatility fitting (polynomials) ---
    fitted_vols = []

    def fit_volatility(i: int, degree: int, title: str):
        vol = vols[:, i]
        params = np.polyfit(tenors, vol, degree)      # highest-power first
        fitted = PolynomialInterpolator(params)
        xs = tenors
        ys_fit = [fitted.calc(x) for x in xs]
        plt.plot(xs, vol, marker='.', label='Discretized volatility')
        plt.plot(xs, ys_fit, label='Fitted volatility')
        plt.title(title)
        plt.xlabel(r"Tenor $\tau$ (years)")
        plt.legend()
        fitted_vols.append(fitted)

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    fit_volatility(0, degree=0, title="1st component")
    plt.subplot(1, 3, 2)
    fit_volatility(1, degree=3, title="2nd component")
    plt.subplot(1, 3, 3)
    fit_volatility(2, degree=3, title="3rd component")
    plt.tight_layout()

    # --- Discretize fitted vol functions for MC/drift calc ---
    mc_tenors = np.linspace(0.0, 25.0, 51)   # 0, 0.5, ..., 25
    mc_vols = np.column_stack([
        [fv.calc(tau) for tau in mc_tenors] for fv in fitted_vols
    ])  # shape [len(mc_tenors), k]

    plt.figure(figsize=(10, 4))
    plt.plot(mc_tenors, mc_vols, marker='.')
    plt.xlabel(r"Tenor $\tau$ (years)")
    plt.title("Volatilities (fitted, discretized)")
    plt.tight_layout()

    # --- Risk-neutral drift m(τ) = sum_i [ (∫_0^τ σ_i(u) du) * σ_i(τ) ] ---
    def m(tau: float, fitted_list) -> float:
        out = 0.0
        for fitted in fitted_list:
            out += integrate(fitted.calc, 0.0, tau, 0.01) * fitted.calc(tau)
        return out

    mc_drift = np.array([m(tau, fitted_vols) for tau in mc_tenors])

    plt.figure(figsize=(10, 4))
    plt.plot(mc_tenors, mc_drift, marker='.')
    plt.xlabel(r"Tenor $\tau$ (years)")
    plt.title("Risk-neutral drift")
    plt.tight_layout()

    # Show all figures at the end
    plt.show()


if __name__ == "__main__":
    main()
