import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numbers import Real
from typing import Union, Tuple
import os

EPS = np.finfo(float).eps

@dataclass
class NelsonSiegelSvenssonCurve:
    beta0: float
    beta1: float
    beta2: float
    beta3: float
    lambd1: float
    lambd2: float

    def factors(self, T):
        lambd1 = self.lambd1
        lambd2 = self.lambd2
        if isinstance(T, Real) and T <= 0:
            return 1, 0, 0
        elif isinstance(T, np.ndarray):
            zero_idx = T <= 0
            T = T.copy()
            T[zero_idx] = EPS
        exp_tt1 = np.exp(-T / lambd1)
        exp_tt2 = np.exp(-T / lambd2)
        factor1 = (1 - exp_tt1) / (T / lambd1)
        factor2 = factor1 - exp_tt1
        factor3 = (1 - exp_tt2) / (T / lambd2) - exp_tt2
        if isinstance(T, np.ndarray):
            factor1[zero_idx] = 1
            factor2[zero_idx] = 0
            factor3[zero_idx] = 0
        return factor1, factor2, factor3

    def zero(self, T):
        factor1, factor2, factor3 = self.factors(T)
        return self.beta0 + self.beta1 * factor1 + self.beta2 * factor2 + self.beta3 * factor3

    def __call__(self, T):
        return self.zero(T)

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

def svensson_func(T, beta0, beta1, beta2, beta3, lambd1, lambd2):
    T = np.array(T)
    lambd1 = max(lambd1, EPS)
    lambd2 = max(lambd2, EPS)
    exp_tt1 = np.exp(-T / lambd1)
    exp_tt2 = np.exp(-T / lambd2)
    factor1 = (1 - exp_tt1) / (T / lambd1)
    factor2 = factor1 - exp_tt1
    factor3 = (1 - exp_tt2) / (T / lambd2) - exp_tt2
    return beta0 + beta1 * factor1 + beta2 * factor2 + beta3 * factor3

def fit_svensson_to_csv(csv_path, country='Country', plot_curves=False, save_betas=True):
    print(f"\n=== Fitting Svensson: {country} ({csv_path}) ===")
    df = pd.read_csv(csv_path)
    maturities = [str(c).strip() for c in df.columns]
    T_years = parse_maturities(maturities)

    betas_list = []
    rmse_list = []

    for idx, row in df.iterrows():
        yields = row.values.astype(float)
        b0_guess = yields[-1]
        b1_guess = yields[0] - yields[-1]
        b2_guess = 0.0
        b3_guess = 0.0
        lambd1_guess = 2.0
        lambd2_guess = 0.5
        p0 = [b0_guess, b1_guess, b2_guess, b3_guess, lambd1_guess, lambd2_guess]
        try:
            params, _ = curve_fit(
                svensson_func,
                T_years, yields,
                p0=p0,
                bounds=([-10, -10, -10, -10, 0.01, 0.01], [15, 15, 15, 15, 20, 20]),
                maxfev=20000
            )
        except RuntimeError:
            print(f"Curve {idx}: Optimization failed, skipping.")
            continue
        beta0, beta1, beta2, beta3, lambd1, lambd2 = params
        betas_list.append([beta0, beta1, beta2, beta3, lambd1, lambd2])
        yfit = svensson_func(T_years, *params)
        rmse = np.sqrt(np.mean((yields - yfit) ** 2))
        rmse_list.append(rmse)

        print(f"Row {idx+1} params: beta0={beta0:.4f}, beta1={beta1:.4f}, beta2={beta2:.4f}, beta3={beta3:.4f}, lambd1={lambd1:.4f}, lambd2={lambd2:.4f} | RMSE={rmse:.5f}")

        if plot_curves:
            T_fine = np.linspace(min(T_years), max(T_years), 200)
            sv_curve = NelsonSiegelSvenssonCurve(*params)
            plt.figure(figsize=(12,6))
            plt.plot(T_years, yields, 'o', label='Observed')
            plt.plot(T_fine, sv_curve(T_fine), '-', label='Svensson fit')
            plt.title(f"{country} Yield Curve Fit (row {idx+1})", fontsize=35, weight='bold', color='#183057', pad=10)
            plt.xlabel("Maturity (years)", fontsize=25)
            plt.ylabel("Yield (%)", fontsize=25)
            plt.legend()
            plt.tight_layout()
            ax = plt.gca()
            textstr = f"RMSE = {rmse:.5f}"
            ax.text(
                0.97, 0.97, textstr,
                transform=ax.transAxes,
                fontsize=25,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', edgecolor='#183057', boxstyle='square,pad=0.3', alpha=0.85)
            )
            plt.show()

    if save_betas and betas_list:
        betas_df = pd.DataFrame(
            betas_list,
            columns=['beta0', 'beta1', 'beta2', 'beta3', 'lambd1', 'lambd2']
        )
        if len(betas_list) == len(rmse_list):
            betas_df['rmse'] = rmse_list
        betas_csv = f"svensson-{country}-betas.csv"
        betas_df.to_csv(betas_csv, index=False)
        print(f"\nSaved all fitted betas (+RMSE) to: {betas_csv}")

def plot_all_curves(csv_path, country='Country', figsize=(12,6), ylim=(-2, 10)):
    df = pd.read_csv(csv_path)
    maturities = [str(c).strip() for c in df.columns]
    T_years = parse_maturities(maturities)

    avg_rmse = None
    betas_csv = f"svensson-{country}-betas.csv"
    try:
        if os.path.exists(betas_csv):
            betas_df = pd.read_csv(betas_csv)
            if "rmse" in betas_df.columns:
                avg_rmse = betas_df["rmse"].mean()
    except Exception:
        avg_rmse = None

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

    if avg_rmse is not None:
        textstr = f"Avg. RMSE = {avg_rmse:.4f}"
        ax.text(
            0.97, 0.97, textstr,
            transform=ax.transAxes,
            fontsize=25,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='white', edgecolor='red', boxstyle='square,pad=0.3', alpha=0.85)
        )
    plt.show()

if __name__ == "__main__":
    datasets = [
        (r'Chapter 2/Data/GBP-Yield-Curve.csv', 'GBP'),
        (r'Chapter 2/Data/SG-Yield-Curve.csv', 'SGD'),
        (r'Chapter 2/Data/USFed-Yield-Curve.csv', 'USD'),
        (r'Chapter 2/Data/CGB-Yield-Curve.csv', 'RMB'),
        (r'Chapter 2/Data/ECB-Yield-Curve.csv', 'EUR'),
    ]
    for csv_path, country in datasets:
        try:
            fit_svensson_to_csv(csv_path, country=country, plot_curves=False, save_betas=True)
        except Exception as e:
            print(f"Error fitting {country}: {e}")

    for csv_path, country in datasets:
        try:
            plot_all_curves(csv_path, country=country)
        except Exception as e:
            print(f"Error plotting {country}: {e}")
