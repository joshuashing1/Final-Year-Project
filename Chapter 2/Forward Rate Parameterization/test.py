import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numbers import Real
from typing import Union, Tuple

# ---- Nelson-Siegel Model Implementation ----

EPS = np.finfo(float).eps

@dataclass
class NelsonSiegelCurve:
    beta0: float
    beta1: float
    beta2: float
    tau: float

    def factors(
        self, T: Union[float, np.ndarray]
    ) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
        tau = self.tau
        if isinstance(T, Real) and T <= 0:
            return 1, 0
        elif isinstance(T, np.ndarray):
            zero_idx = T <= 0
            T = T.copy()
            T[zero_idx] = EPS  # avoid warnings in calculations
        exp_tt0 = np.exp(-T / tau)
        factor1 = (1 - exp_tt0) / (T / tau)
        factor2 = factor1 - exp_tt0
        if isinstance(T, np.ndarray):
            factor1[zero_idx] = 1
            factor2[zero_idx] = 0
        return factor1, factor2

    def zero(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        factor1, factor2 = self.factors(T)
        return self.beta0 + self.beta1 * factor1 + self.beta2 * factor2

    def __call__(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.zero(T)

# ---- Helper: Maturity Parsing ----

def parse_maturities(maturities):
    # Parse header strings like '3 Mo', '1 Yr', '6M', '10 Yr' etc. to years (float)
    parsed = []
    for m in maturities:
        m = m.strip().replace(' ', '')
        if 'M' in m or 'Mo' in m:
            number = float(''.join([ch for ch in m if ch.isdigit() or ch == '.']))
            parsed.append(number / 12)
        elif 'Y' in m or 'Yr' in m:
            number = float(''.join([ch for ch in m if ch.isdigit() or ch == '.']))
            parsed.append(number)
        else:
            # fallback: try to interpret as years
            parsed.append(float(m))
    return np.array(parsed)

# ---- Fitting Function ----

def nelson_siegel_func(T, beta0, beta1, beta2, tau):
    # Used for curve_fit
    T = np.array(T)
    tau = max(tau, EPS)
    exp_tt0 = np.exp(-T / tau)
    factor1 = (1 - exp_tt0) / (T / tau)
    factor2 = factor1 - exp_tt0
    return beta0 + beta1 * factor1 + beta2 * factor2

# ---- Fit NS Curve to All Curves in a CSV ----

def fit_ns_to_csv(csv_path, country='Country', plot_curves=True):
    print(f"\n=== Fitting Nelson-Siegel: {country} ({csv_path}) ===")
    # Read CSV
    df = pd.read_csv(csv_path)
    # Handle comma header or possible whitespace
    maturities = [str(c).strip() for c in df.columns]
    T_years = parse_maturities(maturities)

    for idx, row in df.iterrows():
        yields = row.values.astype(float)
        # Initial guess: [long rate, slope, curvature, tau]
        b0_guess = yields[-1]
        b1_guess = yields[0] - yields[-1]
        b2_guess = 0.0
        tau_guess = 2.0
        p0 = [b0_guess, b1_guess, b2_guess, tau_guess]
        # Bound tau to be >0
        try:
            params, _ = curve_fit(
                nelson_siegel_func,
                T_years, yields, 
                p0=p0,
                bounds=([-10, -10, -10, 0.01], [15, 15, 15, 20]),
                maxfev=10000
            )
        except RuntimeError:
            print(f"Curve {idx}: Optimization failed, skipping.")
            continue
        beta0, beta1, beta2, tau = params
        print(f"Row {idx+1} params: beta0={beta0:.4f}, beta1={beta1:.4f}, beta2={beta2:.4f}, tau={tau:.4f}")

        if plot_curves:
            T_fine = np.linspace(min(T_years), max(T_years), 200)
            ns_curve = NelsonSiegelCurve(*params)
            plt.figure(figsize=(7,4))
            plt.plot(T_years, yields, 'o', label='Observed')
            plt.plot(T_fine, ns_curve(T_fine), '-', label='Nelson-Siegel fit')
            plt.title(f"{country} Yield Curve Fit (row {idx+1})")
            plt.xlabel("Maturity (years)")
            plt.ylabel("Yield (%)")
            plt.legend()
            plt.tight_layout()
            plt.show()

# ---- Main Script ----

if __name__ == "__main__":
    fit_ns_to_csv('Chapter 2\Data\GBP-Yield-Curve.csv', country='GBP')
    fit_ns_to_csv('Chapter 2\Data\SG-Yield-Curve.csv', country='SG')
