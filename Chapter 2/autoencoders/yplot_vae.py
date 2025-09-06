# yplot_vae.py
import os, sys
import numpy as np
import pandas as pd

THIS = os.path.abspath(os.path.dirname(__file__))
PRJ_ROOT = os.path.abspath(os.path.join(THIS, ".."))
if PRJ_ROOT not in sys.path:
    sys.path.insert(0, PRJ_ROOT)

from utils import (
    parse_tenor,
    standardize_fit, standardize_apply, standardize_inverse,
    yield_curves_plot,
)
from autoencoder_variational import VariationalNN
from synthetic_svn_training_vae import pretrain_on_synthetic
from parametric_models.yplot_historical import LinearInterpolant


def process_yield_csv_vae(csv_path: str, title: str, epochs: int, batch_size: int, lr: float, activation: str, noise_std: float,
    latent_dim: int, num_latent_samples: int, save_latent: bool = True, pretrain: dict | None = None, kld_beta: float = 1.0, seed: int = 0,
):
    rng = np.random.default_rng(seed)

    # 1) Load perfectly rectangular CSV: header = tenor labels, rows = curves; no time column
    df = pd.read_csv(csv_path, header=0)
    tenor_labels = [str(c) for c in df.columns]                   # keep given order (already increasing)
    maturities_years = np.array([parse_tenor(c) for c in tenor_labels], dtype=float)

    # 2) Dense matrix (complete grid)
    X = df.to_numpy(dtype=np.float32)                              # shape [n_obs, n_tenors]
    n_obs, n_tenors = X.shape
    print(f"[{title}] Loaded {n_obs} rows with {n_tenors} tenors from '{csv_path}'.")

    # Integer time index for convenience
    T = np.arange(n_obs, dtype=np.int32)

    # 3) VAE
    vae = VariationalNN(param_in=n_tenors, activation=activation, latent_dim=latent_dim, rng=rng)

    # 4) Optional pretrain on synthetic Svensson curves
    if pretrain is not None:
        pretrain_on_synthetic(vae, maturities_years, pretrain, verbose=True)

    # 5) Standardize + optional denoising noise for fine-tuning
    mu_real, sd_real = standardize_fit(X)
    Xz_real = standardize_apply(X, mu_real, sd_real)
    X_train = Xz_real + rng.normal(0.0, noise_std, size=Xz_real.shape).astype(np.float32) if noise_std > 0 else Xz_real

    # Train VAE
    vae.train(X=X_train, epochs=epochs, batch_size=batch_size, lr=lr, shuffle=True, verbose=True, num_latent_samples=num_latent_samples, beta_kld=kld_beta)

    # 6) Always MC-mean decode at evaluation time
    Zhat = vae.reconstruct_mc_mean(Xz_real, num_latent_samples=num_latent_samples)
    X_smooth = standardize_inverse(Zhat.astype(np.float32), mu_real, sd_real)

    # 7) RMSE on full grid (matrix is complete)
    avg_rmse = float(np.sqrt(np.mean((X_smooth - X) ** 2)))
    print(f"[{title}] VAE fit average RMSE on full grid: {avg_rmse:.6f}")

    # 8) Save smoothed CSV (same header)
    out_df = pd.DataFrame(X_smooth, columns=tenor_labels)
    out_csv = f"{title}_vae_yield_reconstructed.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[{title}] Saved smoothed CSV to {out_csv}")

    # 9) Save latent means (posterior μ)
    if save_latent:
        lat_mu = vae.get_latent(Xz_real).astype(np.float32)       # mean of q(z|x)
        lat_df = pd.DataFrame(lat_mu, columns=[f"z{i+1}" for i in range(lat_mu.shape[1])])
        lat_df.insert(0, "t", T)
        lat_csv = f"{title}_vae_latent_space.csv"
        lat_df.to_csv(lat_csv, index=False)
        print(f"[{title}] Saved latent mean factors to {lat_csv}")

    # 10) Plot smoothed curves (historical panel)
    fitted_curves = [LinearInterpolant(maturities_years, row) for row in X_smooth]
    fig_path = f"{title}_vae_curve.png"
    yield_curves_plot(maturities_years, fitted_curves, title=f"{title}", save_path=fig_path)

    return avg_rmse


def main():
    # Example Svensson ranges (adjust to your universe)
    sv_ranges = {
        "beta1":  (2.9778, 3.3357),
        "beta2":  (-3.1356, -2.6116),
        "beta3":  (-671.6345, 919.9867),
        "beta4":  (-925.3099, 665.6271),
        "lambd1": (1.3813, 2.2522),
        "lambd2": (1.4414, 2.0658),
    }

    # Your rectangular dataset (header=tenors, rows=curves)
    datasets = [{"csv_path": r"Chapter 2\data\SGS_Yield_Final.csv", "title": "SGD"}]

    # Monte Carlo samples for both training and evaluation
    K = 5

    pretrain_cfg = {
        "n_samples": 20000,
        "ranges": sv_ranges,
        "epochs": 300,
        "batch_size": 256,
        "lr": 1e-3,
        "noise_std": 0.00,        # noise added to generated curves (raw units)
        "noise_std_train": 0.01,  # noise in standardized space during pretraining
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
            noise_std=0.00,        # fine-tune noise in standardized space
            latent_dim=2,
            num_latent_samples=K,
            save_latent=True,
            pretrain=pretrain_cfg, # set to None to skip pretraining
            kld_beta=0.01,         # KLD loss multiplier
            seed=0,
        )

if __name__ == "__main__":
    main()
