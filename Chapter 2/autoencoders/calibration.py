import numpy as np
# import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoencoder import AutoencoderNN              
from parametric_models.svensson import SvenssonCurve

# ---------- Svensson synthetic generator ----------
def generate_synthetic_svensson(n_samples, taus, ranges, noise_std=0.0005, seed=0):
    """
    ranges keys: beta1,beta2,beta3,beta4,lambd1,lambd2 -> (low, high) tuples
    Returns X with shape [n_samples, len(taus)] in rate units (e.g., 0.025 = 2.5%).
    """
    rng = np.random.default_rng(seed)
    m = len(taus)
    X = np.empty((n_samples, m), dtype=np.float32)
    for i in range(n_samples):
        b1  = rng.uniform(*ranges["beta1"])
        b2  = rng.uniform(*ranges["beta2"])
        b3  = rng.uniform(*ranges["beta3"])
        b4  = rng.uniform(*ranges["beta4"])
        l1  = rng.uniform(*ranges["lambd1"])
        l2  = rng.uniform(*ranges["lambd2"])
        # Optionally enforce l2 >= l1 for better separation:
        # l1, l2 = sorted((l1, l2))

        curve = SvenssonCurve(b1, b2, b3, b4, l1, l2).zero(taus).astype(np.float32)
        if noise_std and noise_std > 0:
            curve = curve + rng.normal(0.0, noise_std, size=m).astype(np.float32)
        X[i] = curve
    return X

# ---------- Standardization ----------
def fit_standardizer(X):
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd < 1e-8] = 1.0
    return mu, sd

def transform(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)

def inverse_transform(Z, mu, sd):
    return (Z * sd + mu).astype(np.float32)

# ---------- Save / Load ----------
def save_ae(ae: AutoencoderNN, path: str, include_optimizer: bool = False):
    payload = {"param_in": np.array([ae.param_in], dtype=np.int32)}
    names = ["e1", "e2", "e3", "d1", "d2", "out"]
    layers = [ae.e1, ae.e2, ae.e3, ae.d1, ae.d2, ae.out]
    for name, layer in zip(names, layers):
        payload[f"{name}_W"] = layer.W
        payload[f"{name}_b"] = layer.b
        if include_optimizer:
            payload[f"{name}_mW"] = layer.mW
            payload[f"{name}_vW"] = layer.vW
            payload[f"{name}_mb"] = layer.mb
            payload[f"{name}_vb"] = layer.vb
    np.savez_compressed(path, **payload)

def load_ae(ae: AutoencoderNN, path: str, include_optimizer: bool = False):
    data = np.load(path)
    if "param_in" in data and int(data["param_in"][0]) != ae.param_in:
        raise ValueError("param_in mismatch between checkpoint and model")
    names = ["e1", "e2", "e3", "d1", "d2", "out"]
    layers = [ae.e1, ae.e2, ae.e3, ae.d1, ae.d2, ae.out]
    for name, layer in zip(names, layers):
        layer.W[...] = data[f"{name}_W"]
        layer.b[...] = data[f"{name}_b"]
        if include_optimizer:
            for k in ("mW","vW","mb","vb"):
                key = f"{name}_{k}"
                if key in data:
                    getattr(layer, k)[...] = data[key]
                else:
                    getattr(layer, k)[...] = 0.0

# ---------- Main pipeline ----------
if __name__ == "__main__":
    # 1) Maturity grid (must match your real data order)
    taus = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], dtype=np.float32)
    m = len(taus)

    # 2) Generate Svensson synthetic dataset
    ranges = {
        "beta1":  (-0.01, 0.06),  # long-run level
        "beta2":  (-0.05, 0.05),  # slope
        "beta3":  (-0.05, 0.05),  # curvature 1
        "beta4":  (-0.05, 0.05),  # curvature 2
        "lambd1": ( 0.05, 5.00),  # time constants > 0
        "lambd2": ( 0.05, 5.00),
    }
    X_syn = generate_synthetic_svensson(
        n_samples=40000, taus=taus, ranges=ranges, noise_std=0.0005, seed=42
    )

    # 3) Standardize (fit on synthetic)
    mu, sd = fit_standardizer(X_syn)
    X_syn_z = transform(X_syn, mu, sd)

    # 4) Train autoencoder
    ae = AutoencoderNN(param_in=m)
    ae.train(X_syn_z, epochs=80, batch_size=256, lr=1e-3, shuffle=True, verbose=True)

    # 5) Save model + scaler
    save_ae(ae, "ae_weights.npz", include_optimizer=False)
    np.savez("scaler_syn.npz", mu=mu, sd=sd, taus=taus)

    # -------- Later (or in another script) --------
    # Load for inference on real data (must use same maturity grid + order)
    ae2 = AutoencoderNN(param_in=m)
    load_ae(ae2, "ae_weights.npz", include_optimizer=False)
    scal = np.load("scaler_syn.npz")
    mu, sd, taus_loaded = scal["mu"], scal["sd"], scal["taus"]

    # Example real curve (replace with your actual quotes; shape [m])
    x_real = np.array([0.021, 0.022, 0.023, 0.024, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029],
                      dtype=np.float32)

    # 6) Predict (reconstruct) on real data
    x_real_z    = transform(x_real[None, :], mu, sd)   # [1, m]
    xhat_real_z = ae2.reconstruct(x_real_z)           # [1, m]
    xhat_real   = inverse_transform(xhat_real_z, mu, sd)[0]

    print("Input real:", x_real)
    print("Reconstructed:", xhat_real)
