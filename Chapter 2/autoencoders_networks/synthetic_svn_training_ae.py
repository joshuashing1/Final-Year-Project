import sys
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve().parent
PRJ_ROOT = THIS.parent

if str(PRJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PRJ_ROOT))

from machine_functions.autoencoder import AutoencoderNN
from parametric_models.machine_functions.svensson import SvenssonCurve
from utility_functions.utils import standardize_fit, standardize_apply

def generate_synthetic_svensson(n_samples: int, maturities_years: np.ndarray, ranges: dict, noise_std: float, seed: int
    ) -> np.ndarray:
    """
    generate Svensson parameters from continuous uniform distribution on interval; 1st and 3rd quartiles 
    of fitted Svensson parameters.
    """
    rng = np.random.default_rng(seed)
    m = len(maturities_years)
    X = np.empty((n_samples, m), dtype=np.float32)
    for i in range(n_samples):
        beta1  = rng.uniform(*ranges["beta1"])
        beta2  = rng.uniform(*ranges["beta2"])
        beta3  = rng.uniform(*ranges["beta3"])
        beta4  = rng.uniform(*ranges["beta4"])
        lambd1  = rng.uniform(*ranges["lambd1"])
        lambd2  = rng.uniform(*ranges["lambd2"])

        curve = SvenssonCurve(beta1=beta1, beta2=beta2, beta3=beta3, beta4=beta4, lambd1=lambd1, lambd2=lambd2)
        y = curve(maturities_years)  
        if noise_std > 0:
            y = y + rng.normal(0.0, noise_std, size=y.shape)
        X[i] = y.astype(np.float32)
    return X


def pretrain_on_synthetic(ae: AutoencoderNN, maturities_years: np.ndarray, syn_cfg: dict, verbose: bool = True):
    """
    pre-train AE network on synthetic Svensson curves with standardization applied.
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

    ae.train(X=Xz_train, epochs=int(syn_cfg["epochs"]), batch_size=int(syn_cfg["batch_size"]), lr=float(syn_cfg["lr"]), shuffle=True, verbose=verbose)
    if verbose:
        print("[pretrain] Finished synthetic pretraining.")