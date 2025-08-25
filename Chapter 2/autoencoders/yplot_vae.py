# smooth_with_autoencoder.py
# Pre-train Autoencoder on synthetic Svensson curves, then fine-tune on real data.
# Outputs:
#   - <title>_smoothed.csv
#   - <title>_smoothed_overlay.png
#   - <title>_historical_curve_smoothed.png
#   - <title>_latent.csv (if save_latent=True)

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS = os.path.abspath(os.path.dirname(__file__))            
PRJ_ROOT = os.path.abspath(os.path.join(THIS, ".."))
if PRJ_ROOT not in sys.path:
    sys.path.insert(0, PRJ_ROOT)

# Your modules
from autoencoder import AutoencoderNN
from parametric_models.yplot_historical import parse_maturities, LinearInterpolant, yield_curves_plot
from parametric_models.svensson import SvenssonCurve  # <-- used for synthetic pretraining


# -------------------------
# Utilities
# -------------------------
def detect_and_split_dates(df: pd.DataFrame):
    cols = df.columns.tolist()

    def is_tenor(c):
        s = str(c).strip().upper()
        if s.endswith("M") or s.endswith("Y"):
            return True
        try:
            float(s)
            return True
        except Exception:
            return False

    if len(cols) > 0 and not is_tenor(cols[0]):
        return df.iloc[:, 0].astype(str).tolist(), df.iloc[:, 1:].copy()
    return None, df.copy()


def build_dense_matrix(values_df: pd.DataFrame):
    tenor_labels = [str(c) for c in values_df.columns]
    maturities_years = parse_maturities(tenor_labels)
    X_list = []
    for _, row in values_df.iterrows():
        y = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(y)

        if mask.sum() >= 2:
            interp = LinearInterpolant(maturities_years[mask], y[mask])
            y_filled = interp(maturities_years)
        elif mask.sum() == 1:
            y_filled = np.full_like(maturities_years, y[mask][0], dtype=float)
        else:
            y_filled = np.zeros_like(maturities_years, dtype=float)

        X_list.append(y_filled)
    X = np.vstack(X_list).astype(np.float32)
    return X, maturities_years, tenor_labels


def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (X - mu) / sd


def standardize_inverse(Z: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return Z * sd + mu


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


# -------------------------
# Synthetic Svensson generation
# -------------------------
def generate_synthetic_svensson(n_samples: int,
                                maturities_years: np.ndarray,
                                ranges: dict,
                                noise_std: float = 0.0,
                                seed: int = 0) -> np.ndarray:
    """
    ranges keys: 'beta1','beta2','beta3','beta4','lambd1','lambd2' -> (low, high) tuples
    Returns: X_syn with shape [n_samples, len(maturities_years)] in same units as real data (e.g., %).
    """
    rng = np.random.default_rng(seed)
    m = len(maturities_years)
    X = np.empty((n_samples, m), dtype=np.float32)
    for i in range(n_samples):
        b1  = rng.uniform(*ranges["beta1"])
        b2  = rng.uniform(*ranges["beta2"])
        b3  = rng.uniform(*ranges["beta3"])
        b4  = rng.uniform(*ranges["beta4"])
        l1  = rng.uniform(*ranges["lambd1"])
        l2  = rng.uniform(*ranges["lambd2"])

        curve = SvenssonCurve(beta1=b1, beta2=b2, beta3=b3, beta4=b4, lambd1=l1, lambd2=l2)
        y = curve(maturities_years)  # evaluate at desired maturities
        if noise_std > 0:
            y = y + rng.normal(0.0, noise_std, size=y.shape)
        X[i] = y.astype(np.float32)
    return X


def pretrain_on_synthetic(ae: AutoencoderNN,
                          maturities_years: np.ndarray,
                          syn_cfg: dict,
                          verbose: bool = True):
    """
    Pre-train AE on synthetic Svensson curves (standardized on synthetic stats).
    syn_cfg has:
      - n_samples, ranges (dict), epochs, batch_size, lr, noise_std, seed
    """
    X_syn = generate_synthetic_svensson(
        n_samples=syn_cfg["n_samples"],
        maturities_years=maturities_years,
        ranges=syn_cfg["ranges"],
        noise_std=float(syn_cfg.get("noise_std", 0.0)),
        seed=int(syn_cfg.get("seed", 0))
    )
    mu_syn, sd_syn = standardize_fit(X_syn)
    Xz_syn = standardize_apply(X_syn, mu_syn, sd_syn)

    # small denoising during pretrain (optional)
    rng = np.random.default_rng(int(syn_cfg.get("seed", 0)))
    if syn_cfg.get("noise_std_train", 0.0) and syn_cfg["noise_std_train"] > 0:
        Xz_train = Xz_syn + rng.normal(0.0, syn_cfg["noise_std_train"], size=Xz_syn.shape).astype(np.float32)
    else:
        Xz_train = Xz_syn

    ae.train(
        X=Xz_train,
        epochs=int(syn_cfg["epochs"]),
        batch_size=int(syn_cfg["batch_size"]),
        lr=float(syn_cfg["lr"]),
        shuffle=True,
        verbose=verbose
    )
    if verbose:
        print("[pretrain] Finished synthetic pretraining.")


# -------------------------
# Processing function (pretrain + finetune)
# -------------------------
def process_yield_csv(csv_path: str, title: str,
                      epochs=200, batch_size=128, lr=1e-3,
                      activation="relu", noise_std=0.01,
                      seed=0, example_index=0, save_latent=True,
                      pretrain: dict | None = None):
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
    datasets = [
        {"csv_path": r"Chapter 2\Data\USTreasury_Yield_Final.csv", "title": "USD"},
        # Add more if you like:
        # {"csv_path": r"Chapter 2\Data\CGB_Yield_Final.csv", "title": "CNY"},
        # {"csv_path": r"Chapter 2\Data\GLC_Yield_Final.csv", "title": "GBP"},
        # {"csv_path": r"Chapter 2\Data\SGS_Yield_Final.csv", "title": "SGD"},
        # {"csv_path": r"Chapter 2\Data\ECB_Yield_Final.csv", "title": "EUR"},
    ]

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
