""" 
Append project root 'Chapter 2/' into Python's import search path.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from autoencoder import AutoencoderNN
from autoencoder_variational import Variational_NN
from parametric_models.svensson import SvenssonCurve

def svn_gen(n_samples, taus, ranges, noise_std=0.0005, seed=0):
    """
    ranges keys: beta1,beta2,beta3,beta4,lambd1,lambd2 -> (low, high) tuples
    Returns X with shape [n_samples, len(taus)] in rate units (e.g., 0.025 = 2.5%).
    """
    rng = np.random.default_rng(seed)
    m = len(taus)
    X = np.empty((n_samples, m), dtype=np.float32)
    for i in range(n_samples):
        beta1  = rng.uniform(*ranges["beta1"])
        beta2  = rng.uniform(*ranges["beta2"])
        beta3  = rng.uniform(*ranges["beta3"])
        beta4  = rng.uniform(*ranges["beta4"])
        lambd1  = rng.uniform(*ranges["lambd1"])
        lambd2  = rng.uniform(*ranges["lambd2"])

        curve = SvenssonCurve(beta1, beta2, beta3, beta4, lambd1, lambd2).zero(taus).astype(np.float32)
        if noise_std and noise_std > 0:
            curve = curve + rng.normal(0.0, noise_std, size=m).astype(np.float32)
        X[i] = curve
    return X

def save_ae(ae: AutoencoderNN, path: str, include_optimizer: bool = False):
    payload = {"param_in": np.array([ae.param_in], dtype=np.int32)}
    names = ["e1", "e2", "e3", "d1", "d2", "out"]
    layers = [ae.encoder1, ae.encoder2, ae.encoder3, ae.decoder1, ae.decoder2, ae.out]
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