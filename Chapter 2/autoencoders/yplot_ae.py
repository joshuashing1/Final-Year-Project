# smooth_with_autoencoder.py
# Produces:
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
# Processing function
# -------------------------
def process_yield_csv(csv_path: str, title: str,
                      epochs=200, batch_size=128, lr=1e-3,
                      activation="relu", noise_std=0.01,
                      seed=0, example_index=0, save_latent=True):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rng = np.random.default_rng(seed)

    # 1) Load CSV
    df_raw = pd.read_csv(csv_path, header=0)
    dates, values_df = detect_and_split_dates(df_raw)

    # 2) Dense matrix
    X, maturities_years, tenor_labels = build_dense_matrix(values_df)
    n_obs, n_tenors = X.shape
    print(f"[{title}] Loaded {n_obs} rows with {n_tenors} tenors.")

    # 3) Standardize + optional noise
    mu, sd = standardize_fit(X)
    Xz = standardize_apply(X, mu, sd)
    if noise_std > 0:
        X_train = Xz + rng.normal(0.0, noise_std, size=Xz.shape).astype(np.float32)
    else:
        X_train = Xz

    # 4) Train AE
    ae = AutoencoderNN(param_in=n_tenors, activation=activation, rng=rng)
    ae.train(X=X_train, epochs=epochs, batch_size=batch_size, lr=lr, shuffle=True, verbose=True)

    # 5) Reconstruct
    Zhat = ae.reconstruct(Xz).astype(np.float32)
    X_smooth = standardize_inverse(Zhat, mu, sd)

    # 6) Save smoothed CSV
    out_df = pd.DataFrame(X_smooth, columns=tenor_labels)
    if dates is not None:
        out_df.insert(0, "Date", dates)
    out_csv = f"{title}_smoothed.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[{title}] Saved smoothed CSV to {out_csv}")

    # 7) Save latent if requested
    if save_latent:
        lat = ae.get_latent(Xz).astype(np.float32)
        lat_df = pd.DataFrame(lat, columns=[f"z{i+1}" for i in range(lat.shape[1])])
        if dates is not None:
            lat_df.insert(0, "Date", dates)
        lat_csv = f"{title}_latent.csv"
        lat_df.to_csv(lat_csv, index=False)
        print(f"[{title}] Saved latent factors to {lat_csv}")

    # 8) Overlay plot
    example_idx = max(0, min(example_index, len(X) - 1))
    overlay_path = f"{title}_smoothed_overlay.png"
    plot_overlay(maturities_years, X[example_idx], X_smooth[example_idx], title, overlay_path)

    # 9) Historical plot
    fitted_curves = [LinearInterpolant(maturities_years, row) for row in X_smooth]
    fig_path = f"{title}_historical_curve_smoothed.png"
    yield_curves_plot(maturities_years, fitted_curves, title=f"{title} (AE-smoothed)", save_path=fig_path)


# -------------------------
# Main driver
# -------------------------
def main():
    datasets = [
        {"csv_path": r"Chapter 2\Data\USTreasury_Yield_Final.csv", "title": "USD"},
        # you can add more datasets here (CNY, GBP, etc.)
    ]

    for item in datasets:
        process_yield_csv(
            csv_path=item["csv_path"],
            title=item["title"],
            epochs=300,
            batch_size=64,
            lr=1e-3,
            activation="relu",
            noise_std=0.01,
            save_latent=True
        )


if __name__ == "__main__":
    main()
