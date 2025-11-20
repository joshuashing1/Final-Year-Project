"""
This Python script deals with the usage of the pre-trained VAE network to model the yield curves 
of a single currency dataset. The pre-training of the network is done using the Svensson parameters 
of its respective currency.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent      
PRJ_ROOT = THIS.parent                      

if str(PRJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PRJ_ROOT))

from utility_functions.utils import parse_tenor, standardize_fit, standardize_apply, standardize_inverse, yield_curves_plot
from machine_functions.autoencoder_variational import VariationalNN
from synthetic_svn_training_vae import pretrain_on_synthetic
from parametric_models.yplot_historical import LinearInterpolant


def process_yield_csv_vae(csv_path: str, title: str, epochs: int, batch_size: int, lr: float, activation: str, noise_std: float,
    latent_dim: int, num_latent_samples: int, save_latent: bool = True, pretrain: dict | None = None, kld_beta: float = 1.0, seed: int = 0,
):
    """
    Data processing of single currency dataset and the plotting of yield curves
    from the pre-trained network
    """
    rng = np.random.default_rng(seed)

    df = pd.read_csv(csv_path, header=0) # load csv
    tenor_labels = [str(c) for c in df.columns]                   
    maturities_years = np.array([parse_tenor(c) for c in tenor_labels], dtype=float)

    X = df.to_numpy(dtype=np.float32) # create dense matrix
    n_obs, n_tenors = X.shape
    T = np.arange(n_obs, dtype=np.int32)
    print(f"[{title}] Loaded {n_obs} rows with {n_tenors} tenors from '{csv_path}'.")

    # call network
    vae = VariationalNN(param_in=n_tenors, activation=activation, latent_dim=latent_dim, rng=rng)

    if pretrain is not None:
        pretrain_on_synthetic(vae, maturities_years, pretrain, verbose=True)
    mu_real, sd_real = standardize_fit(X)
    Xz_real = standardize_apply(X, mu_real, sd_real)
    if noise_std > 0:
        X_train = Xz_real + rng.normal(0.0, noise_std, size=Xz_real.shape).astype(np.float32) 
    else: 
        X_train = Xz_real

    # train network on synthetic Svensson curves and reconstrucy yield of input currency data
    vae.train(X=X_train, epochs=epochs, batch_size=batch_size, lr=lr, shuffle=True, verbose=True, num_latent_samples=num_latent_samples, beta_kld=kld_beta)
    Zhat = vae.reconstruct_mc_mean(Xz_real, num_latent_samples=num_latent_samples)
    X_smooth = standardize_inverse(Zhat.astype(np.float32), mu_real, sd_real)
    avg_rmse = float(np.sqrt(np.mean((X_smooth - X) ** 2)))
    print(f"[{title}] VAE fit average RMSE on full grid: {avg_rmse:.6f}")

    # out reconstructed yield
    out_df = pd.DataFrame(X_smooth, columns=tenor_labels)
    out_csv = f"{title}_vae_yield_reconstructed.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[{title}] Saved smoothed CSV to {out_csv}")

    # save VAE latent variables
    if save_latent:
        lat_mu = vae.get_latent(Xz_real).astype(np.float32)       # mean of q(z|x)
        lat_df = pd.DataFrame(lat_mu, columns=[f"z{i+1}" for i in range(lat_mu.shape[1])])
        lat_df.insert(0, "t", T)
        lat_csv = f"{title}_vae_latent_space.csv"
        lat_df.to_csv(lat_csv, index=False)
        print(f"[{title}] Saved latent mean factors to {lat_csv}")

    # historical plot
    fitted_curves = [LinearInterpolant(maturities_years, row) for row in X_smooth]
    fig_path = f"{title}_vae_curve.png"
    yield_curves_plot(maturities_years, fitted_curves, title=f"{title}", save_path=fig_path)

    return avg_rmse


def main():
    # quartiles of Svensson parameters wrt. its currency
    svn_ranges = {
        "beta1":  (2.9778, 3.3357),
        "beta2":  (-3.1356, -2.6116),
        "beta3":  (-671.6345, 919.9867),
        "beta4":  (-925.3099, 665.6271),
        "lambd1": (1.3813, 2.2522),
        "lambd2": (1.4414, 2.0658),
    }

    datasets = [{"csv_path": r"Chapter 2\data\SGS_Yield_Final.csv", "title": "SGD"}] # edit file path accordingly

    K = 5 # monte carlo samples for vae samples generation

    pretrain_cfg = {
        "n_samples": 20000,
        "ranges": svn_ranges,
        "epochs": 300,
        "batch_size": 256,
        "lr": 1e-3,
        "noise_std": 0.00,        
        "noise_std_train": 0.01,  
        "seed": 0,
        "num_latent_samples": K,
    }

    for item in datasets:
        process_yield_csv_vae(
            csv_path=item["csv_path"],
            title=item["title"],
            epochs=100,
            batch_size=64,
            lr=1e-3,
            activation="relu",
            noise_std=0.00,        
            latent_dim=2,
            num_latent_samples=K,
            save_latent=True,
            pretrain=pretrain_cfg, 
            kld_beta=0.01,         # KLD loss multiplier
            seed=0,
        )

if __name__ == "__main__":
    main()
