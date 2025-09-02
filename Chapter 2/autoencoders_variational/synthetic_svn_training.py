# synthetic_svn_training.py
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

THIS = os.path.abspath(os.path.dirname(__file__))
PRJ_ROOT = os.path.abspath(os.path.join(THIS, ".."))
if PRJ_ROOT not in sys.path:
    sys.path.insert(0, PRJ_ROOT)

# VAE + loss function (PyTorch BCE + KL)
from vae import VAE  # imported for type clarity; not strictly required
from nn_fn import loss_criterion

# Svensson remains NumPy-callable
from parametric_models.svensson import SvenssonCurve

# Torch-based standardization utils
from utils import standardize_fit, standardize_apply


@torch.no_grad()
def generate_synthetic_svensson(
    n_samples: int,
    maturities_years: np.ndarray | torch.Tensor,
    ranges: dict,
    noise_std: float,
    seed: int
) -> torch.Tensor:
    """
    ranges keys: 'beta1','beta2','beta3','beta4','lambd1','lambd2' -> (low, high) tuples
    Returns: X_syn as torch.FloatTensor with shape [n_samples, len(maturities_years)]
             in the same units as real data (e.g., %).
    """
    # Ensure maturities as numpy for SvenssonCurve evaluation
    if isinstance(maturities_years, torch.Tensor):
        maturities_np = maturities_years.detach().cpu().numpy().astype(np.float64)
    else:
        maturities_np = np.asarray(maturities_years, dtype=np.float64)

    rng = np.random.default_rng(seed)
    m = len(maturities_np)
    X_np = np.empty((n_samples, m), dtype=np.float32)

    for i in range(n_samples):
        beta1  = rng.uniform(*ranges["beta1"])
        beta2  = rng.uniform(*ranges["beta2"])
        beta3  = rng.uniform(*ranges["beta3"])
        beta4  = rng.uniform(*ranges["beta4"])
        lambd1 = rng.uniform(*ranges["lambd1"])
        lambd2 = rng.uniform(*ranges["lambd2"])

        curve = SvenssonCurve(
            beta1=beta1, beta2=beta2, beta3=beta3, beta4=beta4,
            lambd1=lambd1, lambd2=lambd2
        )

        y = curve(maturities_np)  # NumPy array
        if noise_std and noise_std > 0:
            y = y + rng.normal(0.0, noise_std, size=y.shape)

        X_np[i] = y.astype(np.float32)

    # Return torch tensor (CPU by default; move to device later as needed)
    return torch.from_numpy(X_np)  # [n_samples, m]


@torch.no_grad()
def _maybe_add_train_noise(Xz_syn: torch.Tensor, syn_cfg: dict) -> torch.Tensor:
    """Optional denoising noise during pretrain (torch-native)."""
    noise_std_train = float(syn_cfg.get("noise_std_train", 0.0))
    if noise_std_train > 0.0:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(syn_cfg.get("seed", 0)))
        noise = torch.randn_like(Xz_syn, generator=g) * noise_std_train
        return (Xz_syn + noise).float()
    return Xz_syn.float()


def pretrain_on_synthetic(
    ae: VAE,  # name kept for backward-compat; this should be your VAE instance
    maturities_years: np.ndarray | torch.Tensor,
    syn_cfg: dict,
    verbose: bool = True
):
    """
    Pre-train VAE on synthetic Svensson curves (standardized on synthetic stats).
    syn_cfg has:
      - n_samples (int)
      - ranges (dict): keys 'beta1','beta2','beta3','beta4','lambd1','lambd2' -> (low, high)
      - epochs (int)
      - batch_size (int)
      - lr (float)
      - noise_std (float)
      - seed (int)
      - [noise_std_train] (float, optional)
      - [device] ("cpu" | "cuda", optional)
    """
    device = syn_cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # === 1) Generate synthetic curves ===
    X_syn = generate_synthetic_svensson(
        n_samples=int(syn_cfg["n_samples"]),
        maturities_years=maturities_years,
        ranges=syn_cfg["ranges"],
        noise_std=float(syn_cfg.get("noise_std", 0.0)),
        seed=int(syn_cfg.get("seed", 0))
    ).float()  # [N, M] torch.FloatTensor (CPU)

    # === 2) Standardize (fit on synthetic stats) ===
    mu_syn, sd_syn = standardize_fit(X_syn)           # [1, M], [1, M]
    Xz_syn = standardize_apply(X_syn, mu_syn, sd_syn) # [N, M]

    # Optional denoising noise during pretrain
    Xz_train = _maybe_add_train_noise(Xz_syn, syn_cfg)  # [N, M]

    # === 3) Prep dataloader ===
    # For a VAE, inputs == targets (reconstruction). If you prefer binarization, do it here.
    # Example (commented): y = (Xz_train > 0.5).float()
    dataset = TensorDataset(Xz_train, Xz_train)
    loader = DataLoader(dataset, batch_size=int(syn_cfg["batch_size"]), shuffle=True, drop_last=False)

    # === 4) Model/optimizer to device ===
    ae.to(device)
    ae.train()  # nn.Module.train() — enable training mode
    optimizer = Adam(ae.parameters(), lr=float(syn_cfg["lr"]))

    # === 5) Train loop (BCE + KL) ===
    epochs = int(syn_cfg["epochs"])
    for epoch in range(1, epochs + 1):
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            y_hat, logvar, mu, _ = ae(xb)
            loss = loss_criterion(y_hat, yb, logvar, mu)
            loss.backward()
            optimizer.step()

            running += loss.item()

        if verbose:
            avg = running / len(loader)
            print(f"[pretrain][epoch {epoch}/{epochs}] loss = {avg:.4f}")

    if verbose:
        print("[pretrain] Finished synthetic pretraining.")
