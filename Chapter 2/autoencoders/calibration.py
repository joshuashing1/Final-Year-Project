""" 
Append project root 'Chapter 2/' into Python's import search path.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from autoencoder import AutoencoderNN
from parametric_models.svensson import SvenssonCurve

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

        curve = SvenssonCurve(b1, b2, b3, b4, l1, l2).zero(taus).astype(np.float32)
        if noise_std and noise_std > 0:
            curve = curve + rng.normal(0.0, noise_std, size=m).astype(np.float32)
        X[i] = curve
    return X

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

if __name__ == "__main__":
    # 1) Maturity grid (must match your real data order)
    taus = np.array([1/12, 1/3, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], dtype=np.float32)
    m = len(taus)

    # 2) Generate Svensson synthetic dataset (raw rates, no standardization)
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

    # 3) Train autoencoder directly on raw rates
    ae = AutoencoderNN(param_in=m, activation= "relu") # "relu", "sigmoid", "tanh"
    ae.train(X_syn, epochs=80, batch_size=256, lr=1e-3, shuffle=True, verbose=True)

    # 4) Save model (no scaler file anymore)
    save_ae(ae, "ae_weights.npz", include_optimizer=False)

    # -------- Later (or in another script) --------
    # Load for inference on real data (must use same maturity grid + order)
    ae2 = AutoencoderNN(param_in=m, activation=ACT)  # keep activation consistent with training
    load_ae(ae2, "ae_weights.npz", include_optimizer=False)

    # Example real curve (replace with your actual quotes; shape [m])
    x_real = np.array([0.021, 0.022, 0.023, 0.024, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029],
                      dtype=np.float32)

    # 5) Predict (reconstruct) on real data directly (no z-scoring)
    xhat_real = ae2.reconstruct(x_real[None, :])[0]

    print("Input real:", x_real)
    print("Reconstructed:", xhat_real)