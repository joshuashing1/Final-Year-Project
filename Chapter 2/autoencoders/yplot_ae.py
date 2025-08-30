import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS = os.path.abspath(os.path.dirname(__file__))            
PRJ_ROOT = os.path.abspath(os.path.join(THIS, ".."))
if PRJ_ROOT not in sys.path:
    sys.path.insert(0, PRJ_ROOT)

from utils import detect_and_split_dates, build_dense_matrix, standardize_fit, standardize_apply, standardize_inverse, yield_curves_plot
from autoencoder import AutoencoderNN
from synthetic_svn_training import pretrain_on_synthetic
from parametric_models.yplot_historical import LinearInterpolant

def plot_overlay(maturities_years, raw_row, smooth_row, title, save_path):
    plt.figure(figsize=(12, 6), dpi=120)
    plt.plot(maturities_years, raw_row, marker="o", linewidth=1.0, label="Original (interp)")
    plt.plot(maturities_years, smooth_row, marker="o", linewidth=1.5, label="Smoothed (AE)")
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Interest Rate (%)")
    plt.title(f"{title}: Original vs Smoothed (example row)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved overlay to {save_path}")


def process_yield_csv(csv_path: str, title: str, epochs: int, batch_size: int, lr: float, activation: str, noise_std: float, seed=0, example_index=0, save_latent=True, pretrain: dict | None = None):
    """
    If 'pretrain' is provided, it will pre-train the AE on synthetic Svensson curves first, then fine-tune on real data.
    pretrain = {
      "n_samples": 20000,
      "ranges": {...},               # see default example in main()
      "epochs": 50, "batch_size": 256, "lr": 1e-3,
      "noise_std": 0.00,             # noise added to generated curves
      "noise_std_train": 0.01,       # noise in standardized space during pretraining
      "seed": 0
    }
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rng = np.random.default_rng(seed)

    # 1) Load CSV
    df_raw = pd.read_csv(csv_path, header=0)
    dates, values_df = detect_and_split_dates(df_raw)

    # 2) Dense matrix from real data
    X, maturities_years, tenor_labels = build_dense_matrix(values_df)
    n_obs, n_tenors = X.shape
    print(f"[{title}] Loaded {n_obs} rows with {n_tenors} tenors.")

    # 3) Create AE
    ae = AutoencoderNN(param_in=n_tenors, activation=activation, rng=rng)

    # 4) (Optional) pre-train on synthetic Svensson
    if pretrain is not None:
        # ensure maturities known before pretrain
        pretrain_on_synthetic(ae, maturities_years, pretrain, verbose=True)

    # 5) Fine-tune on REAL data (standardize on real stats)
    mu_real, sd_real = standardize_fit(X)
    Xz_real = standardize_apply(X, mu_real, sd_real)
    if noise_std > 0:
        X_real_train = Xz_real + rng.normal(0.0, noise_std, size=Xz_real.shape).astype(np.float32)
    else:
        X_real_train = Xz_real

    ae.train(X=X_real_train, epochs=epochs, batch_size=batch_size, lr=lr, shuffle=True, verbose=True)

    # 6) Reconstruct & invert standardization (REAL data)
    Zhat = ae.reconstruct(Xz_real).astype(np.float32)
    X_smooth = standardize_inverse(Zhat, mu_real, sd_real)
    
    rmses = []
    for i, row in values_df.iterrows():
        y = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() == 0:
            rmses.append(np.nan)
            continue
        # compare AE reconstruction only at observed tenors of that row
        y_obs = y[mask]
        yhat_obs = X_smooth[i, mask]
        rmse = float(np.sqrt(np.mean((yhat_obs - y_obs) ** 2)))
        rmses.append(rmse)

    avg_rmse = float(np.nanmean(rmses))
    print(f"[{title}] AE fit average RMSE on observed quotes: {avg_rmse:.6f}")

    # 7) Save smoothed CSV
    out_df = pd.DataFrame(X_smooth, columns=tenor_labels)
    if dates is not None:
        out_df.insert(0, "Date", dates)
    out_csv = f"{title}_ae_yield_reconstructed.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[{title}] Saved smoothed CSV to {out_csv}")

    # 8) Save latent if requested (from REAL standardized inputs)
    if save_latent:
        lat = ae.get_latent(Xz_real).astype(np.float32)
        lat_df = pd.DataFrame(lat, columns=[f"z{i+1}" for i in range(lat.shape[1])])
        if dates is not None:
            lat_df.insert(0, "Date", dates)
        lat_csv = f"{title}_ae_latent_factors.csv"
        lat_df.to_csv(lat_csv, index=False)
        print(f"[{title}] Saved latent factors to {lat_csv}")

    # 9) Overlay
    # example_idx = max(0, min(example_index, len(X) - 1))
    # overlay_path = f"{title}_smoothed_overlay.png"
    # plot_overlay(maturities_years, X[example_idx], X_smooth[example_idx], title, overlay_path)

    # 10) Historical plot
    fitted_curves = [LinearInterpolant(maturities_years, row) for row in X_smooth]
    fig_path = f"{title}_ae_curve.png"
    yield_curves_plot(maturities_years, fitted_curves, title=f"{title}", save_path=fig_path)

    return avg_rmse

def main():
    # === Configure synthetic Svensson parameter ranges (reasonable defaults) ===
    # Units: same as your data (e.g., percent). Tune ranges to your universe.
    sv_ranges = {
        "beta1":  (4.2596, 5.0103),
        "beta2":  (-2.0138, -0.9452),
        "beta3":  (-1.9614, 0.9350),
        "beta4":  (-3.4036, 1.1247),
        "lambd1": (0.5693, 3.5197),
        "lambd2": (2.0220, 3.1856)
    }

    # === Datasets you want to smooth ===
    datasets = [{"csv_path": r"Chapter 2\data\ECB_Yield_Final.csv", "title": "EUR"}]

    # === Pretrain configuration (set to None to disable pretraining) ===
    pretrain_cfg = {
        "n_samples": 20000,
        "ranges": sv_ranges,
        "epochs": 300,
        "batch_size": 256,
        "lr": 1e-3,
        "noise_std": 0.00,       # noise added to the generated curves (in raw units)
        "noise_std_train": 0.01, # noise in standardized space during pretraining
        "seed": 0,
    }

    for item in datasets:
        process_yield_csv(
            csv_path=item["csv_path"],
            title=item["title"],
            epochs=100,            # fine-tune epochs on REAL data
            batch_size=64,
            lr=1e-3,
            activation="relu",
            noise_std=0.01,        # denoising during fine-tuning on REAL data
            save_latent=True,
            pretrain=pretrain_cfg  # <-- set to None to skip pretraining
        )


if __name__ == "__main__":
    main()
