"""
This Python script aims to parse the historical GBP forward data by using the VAE network to generate the
time-dependent latent factors and to reconstruct the forward curves. These latent factors will then 
be used to compute the local volatility used for the HJM simulation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent   
PRJ_ROOT = THIS.parent                   

if str(PRJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PRJ_ROOT))

from utility_functions.utils import parse_tenor, standardize_params, standardize, standardize_inverse, hist_fwd_curves_plot
from machine_functions.autoencoder_variational import VariationalNN

class PolynomialInterpolator:
    def __init__(self, coeffs: np.ndarray):
        self.coeffs = np.asarray(coeffs, dtype=float)

    @classmethod
    def fit(cls, x, y, degree: int = 3):
        coeffs = np.polyfit(np.asarray(x, float), np.asarray(y, float), degree)
        return cls(coeffs)

    def __call__(self, x):
        return np.polyval(self.coeffs, np.asarray(x, float))


def generate_tenor_matrix(df: pd.DataFrame):
    """
    Returns a matrix representation of the historical GBP forward rates across the tenors.
    """
    if "t" in df.columns:
        T = df["t"].to_numpy(dtype=np.int32)
        df = df.drop(columns=["t"])
    else:
        T = np.arange(len(df), dtype=np.int32)

    tenor_labels = list(df.columns)
    maturities_years = np.array([parse_tenor(c) for c in tenor_labels], dtype=float)
    order = np.argsort(maturities_years)
    tenor_labels = [tenor_labels[i] for i in order]
    maturities_years = maturities_years[order]

    X = df[tenor_labels].to_numpy(dtype=np.float32)
    return X, tenor_labels, maturities_years, T


def process_fwd_csv(csv_path: str, title: str, epochs: int, batch_size: int, lr: float, activation: str, noise_std: float, latent_dim: int,
    num_latent_samples: int, save_latent: bool = True, kld_beta: float = 1.0, seed: int = 0):
    """
    Data processing of historical GBP forward dataset and the plotting of forward curves
    from the pre-trained network.
    """
    rng = np.random.default_rng(seed)

    df = pd.read_csv(csv_path, header=0)
    X, tenor_labels, maturities_years, T = generate_tenor_matrix(df)
    n_obs, n_tenors = X.shape
    print(f"[{title}] Loaded {n_obs} rows with {n_tenors} tenors from '{csv_path}'.")

    # call network
    vae = VariationalNN(param_in=n_tenors, activation=activation, latent_dim=latent_dim, rng=rng)

    mu_real, sd_real = standardize_params(X)
    Xz_real = standardize(X, mu_real, sd_real)
    X_train = (Xz_real + rng.normal(0.0, noise_std, size=Xz_real.shape).astype(np.float32)
               if noise_std > 0 else Xz_real)

    # train network using dataset of mini batch size of 64
    vae.train(X=X_train, epochs=epochs, batch_size=batch_size, lr=lr, shuffle=True, verbose=True, 
              num_latent_samples=num_latent_samples, beta_kld=kld_beta)
    Zhat = vae.reconstruct_mc_mean(Xz_real, num_latent_samples=num_latent_samples)
    X_smooth = standardize_inverse(Zhat.astype(np.float32), mu_real, sd_real)
    avg_rmse = float(np.sqrt(np.mean((X_smooth - X) ** 2)))
    print(f"[{title}] VAE fit average RMSE on full grid: {avg_rmse:.6f}")

    # out reconstructed forward
    out_df = pd.DataFrame(X_smooth, columns=tenor_labels)
    out_df.insert(0, "t", T)  
    out_df.to_csv("vae_reconstructed_fwd_rates.csv", index=False)

    # save VAE latent variables
    if save_latent:
        lat_mu = vae.get_latent(Xz_real).astype(np.float32)
        lat_df = pd.DataFrame(lat_mu, columns=[f"z{i+1}" for i in range(lat_mu.shape[1])])
        lat_df.insert(0, "t", T)
        lat_df.to_csv(f"vae_fwd_latent_space.csv", index=False)

    # historical plot
    fitted_curves = [PolynomialInterpolator.fit(maturities_years, row, degree=3) for row in X_smooth]
    fig_path = f"vae_reconstructed_fwd_curves.png"
    hist_fwd_curves_plot(maturities_years, fitted_curves, title=title, save_path=fig_path)

    return avg_rmse


def main():
    datasets = [{"csv_path": r"Chapter 3\data\GLC_fwd_curve_raw.csv",
                 "title": "VAE generated forward curves"}]
    K = 5
    for item in datasets:
        process_fwd_csv(
            csv_path=item["csv_path"],
            title=item["title"],
            epochs=100,
            batch_size=64,
            lr=1e-3,
            activation="relu",
            noise_std=0.00,
            latent_dim=3,
            num_latent_samples=K,
            save_latent=True,
            kld_beta=0.01,
            seed=0
        )

if __name__ == "__main__":
    main()
