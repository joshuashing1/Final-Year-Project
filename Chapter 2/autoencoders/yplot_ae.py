"""Main script for running Autoencoder training.
"""
import numpy as np
from calibration import svn_gen, save_ae, load_ae
from autoencoder import AutoencoderNN

tenors = {
    "SGD": [3/12, 6/12, 1, 2, 5, 10, 15, 20, 30],
    "EUR": [3/12, 6/12] + list(range(1, 31)),
    "USD": [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30],
    "CNY": [3/12, 6/12, 1, 3, 5, 7, 10, 30],
    "GBP": [m/12 for m in range(1, 12)] + [1 + 0.5*k for k in range(0, 2*26 - 2 + 1)],
}

# currency-specific sampling ranges 
svn_param = {
    "beta1":  (-0.01, 0.06),
    "beta2":  (-0.05, 0.05),
    "beta3":  (-0.05, 0.05),
    "beta4":  (-0.05, 0.05),
    "lambd1": ( 0.05, 5.00),
    "lambd2": ( 0.05, 5.00),
}

if __name__ == "__main__":
    # 1) Maturity grid (must match your real data order)
    taus = np.array([1/12, 1/3, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], dtype=np.float32) # modify term structure according to currency
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
    X_syn = svn_gen(
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