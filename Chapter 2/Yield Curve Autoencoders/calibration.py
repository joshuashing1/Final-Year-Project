import numpy as np

from autoencoder import AutoencoderNN

# ---------- Example ----------
param_in = 30
X = np.random.randn(5000, param_in).astype(np.float32)
ae = AutoencoderNN(param_in)  # ReLU-only
ae.train(X, epochs=100, batch_size=128, lr=1e-3)
Z = ae.get_latent(X[:5])         # 5 x 13
Xhat = ae.reconstruct(X[:5])     # 5 x param_in