# run.py
# Usage: python run.py
# Assumes this file sits in the same folder as "variational-autoencoder.py"
# and the CSVs are at the given relative paths.

import os
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# ---------------------------
# Load BetaVAE from file path
# ---------------------------
def load_beta_vae(module_path: str = r"Chapter 2/Yield Curve Autoencoders/variational-autoencoder.py"):
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Cannot find {module_path}")
    spec = importlib.util.spec_from_file_location("vae_module", module_path)
    vae_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vae_module)  # type: ignore
    return vae_module.BetaVAE

# --------------------------------------------
# Helpers: maturity parsing & (de)normalization
# --------------------------------------------
def parse_maturities(maturities):
    parsed = []
    for m in maturities:
        m = str(m).strip().replace(' ', '')
        if 'M' in m or 'Mo' in m:
            number = float(''.join([ch for ch in m if ch.isdigit() or ch == '.']))
            parsed.append(number / 12)
        elif 'Y' in m or 'Yr' in m:
            number = float(''.join([ch for ch in m if ch.isdigit() or ch == '.']))
            parsed.append(number)
        else:
            parsed.append(float(m))
    return np.array(parsed)

def scale_to_tanh_range(X):
    """
    Column-wise min-max scale to [-1, 1].
    Returns X_scaled, mins, maxs
    """
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    ranges = np.where((maxs - mins) == 0.0, 1.0, (maxs - mins))
    X01 = (X - mins) / ranges
    Xscaled = 2.0 * X01 - 1.0
    return Xscaled, mins, maxs

def inverse_from_tanh_range(Xscaled, mins, maxs):
    ranges = np.where((maxs - mins) == 0.0, 1.0, (maxs - mins))
    X01 = 0.5 * (Xscaled + 1.0)
    return X01 * ranges + mins

# --------------------------------
# Core training/eval for one CSV
# --------------------------------
def fit_bvae_to_csv(
    BetaVAE,
    csv_path,
    country='Country',
    latent_dim=13,
    beta=4.0,
    gamma=1000.0,
    max_capacity=25.0,
    capacity_max_iter=int(1e5),
    loss_type='B',
    use_tanh_out=False,
    epochs=200,
    batch_size=8,
    plot_curves=True,
    save_latent=True,
    save_recon=True
):
    print(f"\n=== Training β-VAE: {country} ({csv_path}) ===")
    df = pd.read_csv(csv_path)
    maturities = [str(c).strip() for c in df.columns]
    T_years = parse_maturities(maturities)
    X = df.values.astype(float)  # (num_samples, num_maturities)

    # Optional scaling if using tanh output
    if use_tanh_out:
        X_in, mins, maxs = scale_to_tanh_range(X)
    else:
        X_in = X.copy()
        mins, maxs = None, None

    in_dim = X_in.shape[1]

    # Build model
    model = BetaVAE(
        in_dim=in_dim,
        latent_dim=latent_dim,
        beta=beta,
        gamma=gamma,
        max_capacity=max_capacity,
        capacity_max_iter=capacity_max_iter,
        loss_type=loss_type,
        use_tanh_out=use_tanh_out,
    )
    # Set M/N scaling if desired (batch_size / dataset_size)
    dataset_size = X_in.shape[0]
    model.M_N.assign(float(batch_size) / float(max(dataset_size, 1)))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

    # Train
    history = model.fit(
        X_in,
        X_in,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
    )
    print(f"Final training loss: {history.history['loss'][-1]:.6f}")

    # Reconstructions and latent (use μ as the latent code)
    # The model exposes encode(...) -> (mu, logvar)
    mu, logvar = model.encode(X_in, training=False)
    mu_np = mu.numpy()
    X_hat_scaled = model.generate(X_in).numpy()

    # If we scaled input for tanh, invert back to original scale for saving/plotting
    if use_tanh_out:
        X_hat = inverse_from_tanh_range(X_hat_scaled, mins, maxs)
    else:
        X_hat = X_hat_scaled

    # Save artifacts
    base = f"bvae-{country}"
    if save_latent:
        out_latent = f"{base}-latent.csv"
        pd.DataFrame(mu_np, columns=[f"z{i+1}" for i in range(mu_np.shape[1])]).to_csv(out_latent, index=False)
        print(f"Saved latent means to: {out_latent}")

    if save_recon:
        out_recon = f"{base}-recon.csv"
        pd.DataFrame(X_hat, columns=maturities).to_csv(out_recon, index=False)
        print(f"Saved reconstructed curves to: {out_recon}")

    # Plot reconstructions
    if plot_curves:
        plt.figure(figsize=(12, 6))
        for i in range(X_hat.shape[0]):
            plt.plot(T_years, X_hat[i], lw=1)
        plt.title(country, fontsize=40, weight='bold', color='#183057', pad=10)
        plt.xlabel("Maturity (years)", fontsize=25, color='#183057', weight='bold')
        plt.ylabel("Swap Rate (%)", fontsize=25, color='#183057', weight='bold')
        plt.xlim([min(T_years), max(T_years)])
        plt.ylim(-2, 11)

        ax = plt.gca()
        ax.tick_params(axis='both', colors='#183057', labelsize=18)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(22)
            label.set_color('#183057')
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('#183057')
            ax.spines[spine].set_linewidth(2)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        plt.tight_layout()
        plt.show()

    return {
        "T_years": T_years,
        "X": X,
        "X_in": X_in,
        "X_hat": X_hat,
        "mu": mu_np,
        "model": model,
    }

# ----------------
# Main entrypoint
# ----------------
if __name__ == "__main__":
    # Reproducibility (optional)
    tf.random.set_seed(42)
    np.random.seed(42)

    BetaVAE = load_beta_vae(r"Chapter 2/Yield Curve Autoencoders/variational-autoencoder.py")

    # Same dataset list/paths as your earlier scripts
    datasets = [
        (r'Chapter 2/Data/GBP-Yield-Curve.csv', 'GBP'),
        (r'Chapter 2/Data/SG-Yield-Curve.csv',  'SGD'),
        (r'Chapter 2/Data/USFed-Yield-Curve.csv','USD'),
        (r'Chapter 2/Data/CGB-Yield-Curve.csv',  'RMB'),
        (r'Chapter 2/Data/ECB-Yield-Curve.csv',  'EUR'),
    ]

    # Global training settings for all runs
    latent_dim = 13
    epochs = 200
    batch_size = 8
    use_tanh_out = False  # set True if you want outputs constrained to [-1, 1] with scaling

    for csv_path, country in datasets:
        try:
            fit_bvae_to_csv(
                BetaVAE,
                csv_path,
                country=country,
                latent_dim=latent_dim,
                loss_type='B',          # 'H' (Higgins) or 'B' (capacity)
                beta=4.0,
                gamma=1000.0,
                max_capacity=25.0,
                capacity_max_iter=int(1e5),
                use_tanh_out=use_tanh_out,
                epochs=epochs,
                batch_size=batch_size,
                plot_curves=True,       # set False to skip plotting
                save_latent=True,
                save_recon=True
            )
        except Exception as e:
            print(f"Error training β-VAE on {country}: {e}")
