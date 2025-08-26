import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS = os.path.abspath(os.path.dirname(__file__))            
PRJ_ROOT = os.path.abspath(os.path.join(THIS, ".."))
if PRJ_ROOT not in sys.path:
    sys.path.insert(0, PRJ_ROOT)

from utils import detect_and_split_dates, build_dense_matrix, standardize_fit, standardize_apply, standardize_inverse
from autoencoder import AutoencoderNN
from synthetic_svn_training import pretrain_on_synthetic
from parametric_models.yplot_historical import LinearInterpolant, yield_curves_plot

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


def process_yield_csv(csv_path: str, title: str, epochs=200, batch_size=128, lr=1e-3, activation="relu", noise_std=0.01, seed=0, example_index=0, save_latent=True, pretrain: dict | None = None):
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

    # 7) Save smoothed CSV
    out_df = pd.DataFrame(X_smooth, columns=tenor_labels)
    if dates is not None:
        out_df.insert(0, "Date", dates)
    out_csv = f"{title}_smoothed.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[{title}] Saved smoothed CSV to {out_csv}")

    # 8) Save latent if requested (from REAL standardized inputs)
    if save_latent:
        lat = ae.get_latent(Xz_real).astype(np.float32)
        lat_df = pd.DataFrame(lat, columns=[f"z{i+1}" for i in range(lat.shape[1])])
        if dates is not None:
            lat_df.insert(0, "Date", dates)
        lat_csv = f"{title}_latent.csv"
        lat_df.to_csv(lat_csv, index=False)
        print(f"[{title}] Saved latent factors to {lat_csv}")

    # 9) Overlay
    example_idx = max(0, min(example_index, len(X) - 1))
    overlay_path = f"{title}_smoothed_overlay.png"
    plot_overlay(maturities_years, X[example_idx], X_smooth[example_idx], title, overlay_path)

    # 10) Historical plot
    fitted_curves = [LinearInterpolant(maturities_years, row) for row in X_smooth]
    fig_path = f"{title}_historical_curve_smoothed.png"
    yield_curves_plot(maturities_years, fitted_curves, title=f"{title} (AE-smoothed)", save_path=fig_path)


def main():
    # === Configure synthetic Svensson parameter ranges (reasonable defaults) ===
    # Units: same as your data (e.g., percent). Tune ranges to your universe.
    sv_ranges = {
        "beta1":  (3.3661, 4.9353),
        "beta2":  (-3.3400, -1.6373),
        "beta3":  (-4.5277, -0.2641),
        "beta4":  (-8.0692, -0.9213),
        "lambd1": (0.4154, 5.0450),
        "lambd2": (0.0850, 2.3469)
    }

    # === Datasets you want to smooth ===
    datasets = [{"csv_path": r"Chapter 2\Data\USTreasury_Yield_Final.csv", "title": "USD"}]

    # === Pretrain configuration (set to None to disable pretraining) ===
    pretrain_cfg = {
        "n_samples": 20000,
        "ranges": sv_ranges,
        "epochs": 60,
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
            epochs=300,            # fine-tune epochs on REAL data
            batch_size=64,
            lr=1e-3,
            activation="relu",
            noise_std=0.01,        # denoising during fine-tuning on REAL data
            save_latent=True,
            pretrain=pretrain_cfg  # <-- set to None to skip pretraining
        )


if __name__ == "__main__":
    main()
