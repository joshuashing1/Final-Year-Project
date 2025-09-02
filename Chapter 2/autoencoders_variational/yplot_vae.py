# yplot_vae.py
import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn.functional as F

THIS = os.path.abspath(os.path.dirname(__file__))
PRJ_ROOT = os.path.abspath(os.path.join(THIS, ".."))
if PRJ_ROOT not in sys.path:
    sys.path.insert(0, PRJ_ROOT)

from utils import (
    detect_and_split_dates, build_dense_matrix,
    standardize_fit, standardize_apply, standardize_inverse,
    yield_curves_plot
)
from vae import VAE
from synthetic_svn_training import pretrain_on_synthetic
from parametric_models.yplot_historical import LinearInterpolant


# ---------- helpers (PyTorch) ----------

def _mse_kld_loss(y_hat: torch.Tensor,
                  y: torch.Tensor,
                  mu: torch.Tensor,
                  logvar: torch.Tensor,
                  beta_kld: float = 1.0) -> torch.Tensor:
    """
    MSE reconstruction + beta * KL(q(z|x) || N(0,I))
    Matches the logic you used in the NumPy VariationalNN.
    """
    # MSE summed over features, then mean over batch
    recon = F.mse_loss(y_hat, y, reduction="sum")
    # KL term summed over batch/latent dims (standard closed-form)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta_kld * kld


@torch.no_grad()
def _reconstruct_mc_mean(vae: VAE,
                         Xz: torch.Tensor,
                         num_latent_samples: int = 3,
                         device: str = "cpu") -> torch.Tensor:
    """
    Monte-Carlo mean reconstruction: average decoder(x) over K latent samples.
    """
    vae.eval()
    Xz = Xz.to(device)
    outs = []
    for _ in range(num_latent_samples):
        y_hat, _, _, _ = vae(Xz)  # sampling happens inside
        outs.append(y_hat)
    return torch.stack(outs, dim=0).mean(dim=0).detach().cpu()


@torch.no_grad()
def _latent_mu(vae: VAE, Xz: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """
    Posterior mean μ(x): run encoder + latent heads without sampling, return μ.
    """
    vae.eval()
    Xz = Xz.to(device)
    with torch.no_grad():
        p_x = vae.encoder(Xz)
        # compute μ and logvar explicitly without sampling
        mu = vae.latent_z.mu(p_x)
    return mu.detach().cpu()


def _finetune_vae_on_real(vae: VAE,
                          Xz_train: torch.Tensor,
                          epochs: int,
                          batch_size: int,
                          lr: float,
                          beta_kld: float = 1.0,
                          device: str = "cpu",
                          verbose: bool = True):
    """
    Simple PyTorch train loop on standardized real data with MSE + β·KL.
    """
    dataset = TensorDataset(Xz_train, Xz_train)  # AE setup: inputs == targets
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    vae.to(device)
    vae.train()
    opt = Adam(vae.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            y_hat, logvar, mu, _ = vae(xb)
            loss = _mse_kld_loss(y_hat, yb, mu, logvar, beta_kld=beta_kld)
            loss.backward()
            opt.step()

            running += loss.item()

        if verbose:
            avg = running / len(loader)
            print(f"[finetune][epoch {ep}/{epochs}] loss = {avg:.4f}")


# ---------- main pipeline (PyTorch) ----------

def process_yield_csv_vae(
    csv_path: str, title: str, epochs: int, batch_size: int, lr: float,
    activation: str,  # kept for signature compatibility; VAE uses ReLU internally
    noise_std: float, latent_dim: int,
    num_latent_samples: int,
    save_latent: bool = True, pretrain: dict | None = None,
    kld_beta: float = 1.0
):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    rng = np.random.default_rng(0)
    device = pretrain.get("device", "cpu") if pretrain else ("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load CSV
    df_raw = pd.read_csv(csv_path, header=0)
    dates, values_df = detect_and_split_dates(df_raw)

    # 2) Dense matrix (torch tensors)
    X, maturities_years, tenor_labels = build_dense_matrix(values_df)
    n_obs, n_tenors = X.shape
    print(f"[{title}] Loaded {n_obs} rows with {n_tenors} tenors.")

    # 3) VAE (choose a hidden size matching your older 30-unit hidden layers)
    hidden_size = 30
    vae = VAE(input_size=n_tenors, hidden_size=hidden_size, latent_size=latent_dim)

    # 4) Pre-train on synthetic Svensson curves (optional)
    if pretrain is not None:
        pretrain_on_synthetic(vae, maturities_years, pretrain, verbose=True)

    # 5) Fine-tune on real data (standardize on REAL stats)
    mu_real, sd_real = standardize_fit(X)                  # torch [1, M]
    Xz_real = standardize_apply(X, mu_real, sd_real)       # torch [N, M]

    if noise_std > 0:
        # small Gaussian noise injection (on standardized space)
        g = torch.Generator(device="cpu")
        g.manual_seed(0)
        noise_t = torch.randn_like(Xz_real, generator=g) * float(noise_std)
        X_train = (Xz_real + noise_t).float()
    else:
        X_train = Xz_real.float()

    _finetune_vae_on_real(
        vae=vae,
        Xz_train=X_train,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        beta_kld=kld_beta,
        device=device,
        verbose=True
    )

    # 6) Always MC-mean decode in standardized space
    Zhat_t = _reconstruct_mc_mean(vae, Xz_real, num_latent_samples=num_latent_samples, device=device)  # torch [N, M]

    # Inverse-standardize back to yield units
    X_smooth_t = standardize_inverse(Zhat_t, mu_real, sd_real)  # torch [N, M]
    X_smooth = X_smooth_t.detach().cpu().numpy().astype(np.float32)

    # 7) RMSE on observed quotes (use only available maturities per row)
    rmses = []
    for i, row in values_df.iterrows():
        y = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() == 0:
            rmses.append(np.nan)
            continue
        rmse = float(np.sqrt(np.mean((X_smooth[i, mask] - y[mask]) ** 2)))
        rmses.append(rmse)
    avg_rmse = float(np.nanmean(rmses))
    print(f"[{title}] VAE fit average RMSE on observed quotes: {avg_rmse:.6f}")

    # 8) Save smoothed CSV
    out_df = pd.DataFrame(X_smooth, columns=tenor_labels)
    if dates is not None:
        out_df.insert(0, "Date", dates)
    out_csv = f"{title}_vae_yield_reconstructed.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[{title}] Saved smoothed CSV to {out_csv}")

    # 9) Save latent means (posterior μ)
    if save_latent:
        lat_mu_t = _latent_mu(vae, Xz_real, device=device)             # torch [N, latent_dim]
        lat_mu = lat_mu_t.numpy().astype(np.float32)
        lat_df = pd.DataFrame(lat_mu, columns=[f"z{i+1}" for i in range(lat_mu.shape[1])])
        if dates is not None:
            lat_df.insert(0, "Date", dates)
        lat_csv = f"{title}_vae_latent_mean.csv"
        lat_df.to_csv(lat_csv, index=False)
        print(f"[{title}] Saved latent mean factors to {lat_csv}")

    # 10) Plot smoothed curves
    fitted_curves = [LinearInterpolant(maturities_years, row) for row in X_smooth]
    fig_path = f"{title}_vae_curve.png"
    plot_title = f"{title}"
    yield_curves_plot(maturities_years, fitted_curves, title=plot_title, save_path=fig_path)

    return avg_rmse


def main():
    sv_ranges = {
        "beta1":  (3.3662, 4.9353),
        "beta2":  (-3.3400, -1.6373),
        "beta3":  (-4.5276, -0.2641),
        "beta4":  (-8.0692, -0.9213),
        "lambd1": (0.4154, 5.0450),
        "lambd2": (0.0850, 2.3469)
    }

    datasets = [{"csv_path": r"Chapter 2\data\USTreasury_Yield_Final.csv", "title": "USD"}]

    K = 3

    pretrain_cfg = {
        "n_samples": 20000,
        "ranges": sv_ranges,
        "epochs": 300,
        "batch_size": 256,
        "lr": 1e-3,
        "noise_std": 0.00,
        "noise_std_train": 0.01,
        "seed": 0,
        "num_latent_samples": K,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # pretrain helper uses BCE+KL in synthetic_svn_training; that's fine for warm start
    }

    for item in datasets:
        process_yield_csv_vae(
            csv_path=item["csv_path"],
            title=item["title"],
            epochs=100,
            batch_size=64,
            lr=1e-3,
            activation="relu",   # kept for signature compatibility
            noise_std=0.00,
            latent_dim=13,
            save_latent=True,
            pretrain=pretrain_cfg,
            num_latent_samples=K,
            kld_beta=1.0
        )

if __name__ == "__main__":
    main()
