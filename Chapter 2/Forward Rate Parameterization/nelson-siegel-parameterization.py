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

    def factors(self, T: Union[float, np.ndarray]) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
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
            parsed.append(float(m))
    return np.array(parsed)

# ---- Fitting Function ----

def nelson_siegel_func(T, beta0, beta1, beta2, tau):
    T = np.array(T)
    tau = max(tau, EPS)
    exp_tt0 = np.exp(-T / tau)
    factor1 = (1 - exp_tt0) / (T / tau)
    factor2 = factor1 - exp_tt0
    return beta0 + beta1 * factor1 + beta2 * factor2

# ---- Fit NS Curve to All Curves in a CSV AND SAVE BETAS ----

def fit_ns_to_csv(csv_path, country='Country', plot_curves=True, save_betas=True):
    print(f"\n=== Fitting Nelson-Siegel: {country} ({csv_path}) ===")
    df = pd.read_csv(csv_path)
    maturities = [str(c).strip() for c in df.columns]
    T_years = parse_maturities(maturities)

    betas_list = []

    for idx, row in df.iterrows():
        yields = row.values.astype(float)
        b0_guess = yields[-1]
        b1_guess = yields[0] - yields[-1]
        b2_guess = 0.0
        tau_guess = 2.0
        p0 = [b0_guess, b1_guess, b2_guess, tau_guess]
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
        betas_list.append([beta0, beta1, beta2, tau])  # <-- Save params
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

    if save_betas and betas_list:
        betas_df = pd.DataFrame(
            betas_list,
            columns=['beta0', 'beta1', 'beta2', 'tau']
        )
        betas_csv = f"nelson-siegel-{country}-betas.csv"
        betas_df.to_csv(betas_csv, index=False)
        print(f"\nSaved all fitted betas to: {betas_csv}")

# ---- New: Plot All Curves in One Figure (like screenshot) ----

def plot_all_curves(csv_path, country='Country', figsize=(3,3), ylim=(-2, 10)):
    df = pd.read_csv(csv_path)
    maturities = [str(c).strip() for c in df.columns]
    T_years = parse_maturities(maturities)

    plt.figure(figsize=figsize)
    for idx, row in df.iterrows():
        yields = row.values.astype(float)
        plt.plot(T_years, yields, lw=1)

    plt.title(country, fontsize=35, weight='bold', color='#183057', pad=10)
    plt.xlabel("Maturity (years)", fontsize=25)
    plt.ylabel("Swap Rate (%)", fontsize=25)
    plt.xlim([min(T_years), max(T_years)])
    plt.ylim(ylim)
    plt.grid(False)
    plt.tight_layout()

    ax = plt.gca()
    ax.spines['left'].set_color('#183057')
    ax.spines['bottom'].set_color('#183057')
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', colors='#183057', labelsize=25)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(25)
        label.set_color('#183057')
    plt.show()

# ---- Main Script ----
if __name__ == "__main__":
    # List of (CSV path, country code)
    datasets = [
        (r'Chapter 2/Data/GBP-Yield-Curve.csv', 'GBP'),
        (r'Chapter 2/Data/SG-Yield-Curve.csv', 'SGD'),
        (r'Chapter 2/Data/USFed-Yield-Curve.csv', 'USD'),
        (r'Chapter 2/Data/CGB-Yield-Curve.csv', 'RMB'),
        (r'Chapter 2/Data/ECB-Yield-Curve.csv', 'EUR'),
    ]
    for csv_path, country in datasets:
        try:
            plot_all_curves(csv_path, country=country)
        except Exception as e:
            print(f"Error plotting {country}: {e}")

    for csv_path, country in datasets:
        try:
            fit_ns_to_csv(csv_path, country=country, plot_curves=False, save_betas=True)
        except Exception as e:
            print(f"Error fitting {country}: {e}")