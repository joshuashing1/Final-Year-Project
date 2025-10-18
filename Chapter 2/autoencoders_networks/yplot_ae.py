import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS = os.path.abspath(os.path.dirname(__file__))            
PRJ_ROOT = os.path.abspath(os.path.join(THIS, ".."))
if PRJ_ROOT not in sys.path:
    sys.path.insert(0, PRJ_ROOT)

from utility_functions.utils import parse_tenor, standardize_fit, standardize_apply, standardize_inverse, yield_curves_plot
from machine_functions.autoencoder import AutoencoderNN
from synthetic_svn_training_ae import pretrain_on_synthetic
from parametric_models.yplot_historical import LinearInterpolant


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
    rng = np.random.default_rng(seed)

    # 1) Load CSV
    df = pd.read_csv(csv_path, header=0)
    tenor_labels = [str(c) for c in df.columns]                  # preserve given order (already increasing)
    maturities_years = np.array([parse_tenor(c) for c in tenor_labels], dtype=float)

    # 2) Dense matrix from real data
    X = df.to_numpy(dtype=np.float32)                             # shape [n_obs, n_tenors]
    n_obs, n_tenors = X.shape
    print(f"[{title}] Loaded {n_obs} rows with {n_tenors} tenors from '{csv_path}'.")
    
    # a simple integer time index (0..n_obs-1) since there is no 'time' column
    T = np.arange(n_obs, dtype=np.int32)

    # 3) Create AE
    ae = AutoencoderNN(param_in=n_tenors, activation=activation, rng=rng)

    # 4) (Optional) pre-train on synthetic Svensson
    if pretrain is not None:
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
    
    avg_rmse = float(np.sqrt(np.mean((X_smooth - X) ** 2)))
    print(f"[{title}] AE fit average RMSE on observed quotes: {avg_rmse:.6f}")

    # 7) Save smoothed CSV
    out_df = pd.DataFrame(X_smooth, columns=tenor_labels)
    out_csv = f"{title}_ae_yield_reconstructed.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[{title}] Saved smoothed CSV to {out_csv}")

    # 8) Save latent if requested (from REAL standardized inputs)
    if save_latent:
        lat = ae.get_latent(Xz_real).astype(np.float32)
        lat_df = pd.DataFrame(lat, columns=[f"z{i+1}" for i in range(lat.shape[1])])
        lat_df.insert(0, "t", T)
        lat_csv = f"{title}_ae_latent_factors.csv"
        lat_df.to_csv(lat_csv, index=False)
        print(f"[{title}] Saved latent factors to {lat_csv}")

    # 10) Historical plot
    fitted_curves = [LinearInterpolant(maturities_years, row) for row in X_smooth]
    fig_path = f"{title}_ae_curve.png"
    yield_curves_plot(maturities_years, fitted_curves, title=f"{title}", save_path=fig_path)

    return avg_rmse

def main():
    sv_ranges = {
        "beta1":  (3.3662, 4.9353),
        "beta2":  (-3.3400, -1.6373),
        "beta3":  (-4.5276, -0.2641),
        "beta4":  (-8.0692, -0.9213),
        "lambd1": (0.4154, 5.0450),
        "lambd2": (0.0850, 2.3469)
    }

    datasets = [{"csv_path": r"Chapter 2\data\USTreasury_Yield_Final.csv", "title": "USD"}] # copy file path here

    pretrain_cfg = {
        "n_samples": 20000,
        "ranges": sv_ranges,
        "epochs": 300,
        "batch_size": 256,
        "lr": 1e-3,
        "noise_std": 0.00,       
        "noise_std_train": 0.01, 
        "seed": 0,
    }

    for item in datasets:
        process_yield_csv(
            csv_path=item["csv_path"],
            title=item["title"],
            epochs=100,            
            batch_size=64,
            lr=1e-3,
            activation="relu",
            noise_std=0.01,        
            save_latent=True,
            pretrain=pretrain_cfg  
        )


if __name__ == "__main__":
    main()
