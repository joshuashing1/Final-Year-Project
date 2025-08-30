# yplot_vae.py  (fixed)
import os, sys
import numpy as np
import pandas as pd

THIS = os.path.abspath(os.path.dirname(__file__))
PRJ_ROOT = os.path.abspath(os.path.join(THIS, ".."))
if PRJ_ROOT not in sys.path:
    sys.path.insert(0, PRJ_ROOT)

from utils import (
    detect_and_split_dates, build_dense_matrix,
    standardize_fit, standardize_apply, standardize_inverse,
    yield_curves_plot
)
from autoencoder_variational import VariationalNN
from synthetic_svn_training import pretrain_on_synthetic
from parametric_models.yplot_historical import LinearInterpolant


# ---------------- End-to-end ----------------
def process_yield_csv_vae(csv_path: str, title: str, epochs: int, batch_size: int, lr: float, activation: str,
                          noise_std: float, latent_dim: int = 13, save_latent=True, pretrain: dict | None = None,
                          weight_decay: float = 0.0, kld_weight: float = 1.0,
                          beta: float = 1.0, decode_mode: str = "mean",
                          num_latent_samples: int = 1):
    """
    num_latent_samples (K) is used for:
      - training Monte-Carlo gradient estimates, and
      - MC decoding if decode_mode == "mc_mean".
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    rng = np.random.default_rng(0)

    # 1) Load CSV
    df_raw = pd.read_csv(csv_path, header=0)
    dates, values_df = detect_and_split_dates(df_raw)

    # 2) Dense matrix
    X, maturities_years, tenor_labels = build_dense_matrix(values_df)
    n_obs, n_tenors = X.shape
    print(f"[{title}] Loaded {n_obs} rows with {n_tenors} tenors.")

    # 3) VAE (Higgins only; mean-field)
    # FIX: use param_in (not in_dim)
    vae = VariationalNN(
        param_in=n_tenors, activation=activation, latent_dim=latent_dim, rng=rng, beta=beta
    )

    # 4) Pretrain (optional)
    if pretrain is not None:
        pretrain_on_synthetic(vae, maturities_years, pretrain, verbose=True)

    # 5) Fine-tune (standardize on real stats)
    mu_real, sd_real = standardize_fit(X)
    Xz_real = standardize_apply(X, mu_real, sd_real)
    X_train = Xz_real + rng.normal(0.0, noise_std, size=Xz_real.shape).astype(np.float32) if noise_std > 0 else Xz_real

    # FIX: call train(...) (the class has no train_epoch)
    epoch_totals, epoch_recs, epoch_klds = vae.train(
        X=X_train,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        shuffle=True,
        verbose=True,
        kld_weight=kld_weight,
        num_latent_samples=num_latent_samples,
    )

    # 6) Reconstruct & invert standardization
    if decode_mode == "mean":
        Zhat = vae.reconstruct_mean(Xz_real)
        subtitle = "VAE (posterior mean decode)"
        tag = "vae_mean"
    elif decode_mode == "mc_mean":
        Zhat = vae.reconstruct_mc_mean(Xz_real, num_latent_samples=num_latent_samples)
        subtitle = f"VAE (MC mean, K={num_latent_samples})"
        tag = f"vae_mcmeanK{num_latent_samples}"
    else:  # "sampled"
        Zhat = vae.reconstruct(Xz_real)
        subtitle = "VAE (single sampled decode)"
        tag = "vae_sampled"

    X_smooth = standardize_inverse(Zhat.astype(np.float32), mu_real, sd_real)

    # 7) RMSE on observed quotes only
    rmses = []
    for i, row in values_df.iterrows():
        y = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() == 0:
            rmses.append(np.nan); continue
        y_obs = y[mask]; yhat_obs = X_smooth[i, mask]
        rmse = float(np.sqrt(np.mean((yhat_obs - y_obs) ** 2)))
        rmses.append(rmse)
    avg_rmse = float(np.nanmean(rmses))
    print(f"[{title}] VAE fit average RMSE on observed quotes: {avg_rmse:.6f}")

    # 8) Save smoothed CSV
    out_df = pd.DataFrame(X_smooth, columns=tenor_labels)
    if dates is not None:
        out_df.insert(0, "Date", dates)
    out_csv = f"{title}_yield_reconstructed_{tag}.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[{title}] Saved smoothed CSV to {out_csv}")

    # 9) Save latent means (posterior μ)
    # FIX: VariationalNN exposes get_latent(), not encode_mean()
    if save_latent:
        lat_mu = vae.get_latent(Xz_real).astype(np.float32)
        lat_df = pd.DataFrame(lat_mu, columns=[f"z{i+1}" for i in range(lat_mu.shape[1])])
        if dates is not None:
            lat_df.insert(0, "Date", dates)
        lat_csv = f"{title}_latent_mean_vae.csv"
        lat_df.to_csv(lat_csv, index=False)
        print(f"[{title}] Saved latent mean factors to {lat_csv}")

    # 10) Plot smoothed curves
    fitted_curves = [LinearInterpolant(maturities_years, row) for row in X_smooth]
    fig_path = f"{title}_{tag}_curve.png"
    # FIX: utils.yield_curves_plot has no 'subtitle' kwarg; fold subtitle into title
    plot_title = f"{title} — {subtitle}" if subtitle else title
    yield_curves_plot(maturities_years, fitted_curves, title=plot_title, save_path=fig_path)

    return avg_rmse


# ---------------- Main ----------------
def main():
    sv_ranges = {
        "beta1":  (3.7304, 4.4242),
        "beta2":  (-3.6421, 1.4914),
        "beta3":  (-0.8597, 2.9231),
        "beta4":  (-2.6874, 2.7582),
        "lambd1": (0.8896, 5.0500),
        "lambd2": (1.1007, 5.0500)
    }

    datasets = [{"csv_path": r"Chapter 2\Data\GLC_Yield_Final.csv", "title": "GBP"}]

    # Unified K for training and MC decoding (if decode_mode='mc_mean')
    K = 10

    pretrain_cfg = {
        "n_samples": 20000,
        "ranges": sv_ranges,
        "epochs": 300,
        "batch_size": 256,
        "lr": 1e-3,
        "noise_std": 0.00,
        "noise_std_train": 0.005,
        "seed": 0,
        "num_latent_samples": K,   # K during pretraining
    }

    for item in datasets:
        process_yield_csv_vae(
            csv_path=item["csv_path"],
            title=item["title"],
            epochs=120,
            batch_size=64,
            lr=1e-3,
            activation="relu",
            noise_std=0.00,
            latent_dim=13,
            save_latent=True,
            pretrain=pretrain_cfg,
            weight_decay=0.0,
            kld_weight=1.0,
            beta=1.0,
            decode_mode="mc_mean",
            num_latent_samples=K
        )

if __name__ == "__main__":
    main()
