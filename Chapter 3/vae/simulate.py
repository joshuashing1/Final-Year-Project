"""
This Python script computes the dynamic volatility derived from 
VAE latent factors and implements the HJM simulation using it.
"""

import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from utility_functions.utils import export_volatility, export_simul_fwd

BASE = Path("Chapter 3")
VAE_DIR = BASE / "vae"
DATA_DIR = BASE / "data"
FWD, LAT  = VAE_DIR/"vae_reconstructed_fwd_rates.csv", VAE_DIR/"vae_fwd_latent_space.csv"
HIST_FWD  = DATA_DIR/"GLC_fwd_curve_raw.csv"

def parse_tenor(s):
    """
    Converts all tenors into years.
    """
    s=str(s).strip().upper()
    return float(s[:-1])/12 if s.endswith("M") else (float(s[:-1]) if s.endswith("Y") else float(s))

def vol_compute(fwd_csv: Path | str, lat_csv: Path | str, eps: float = 1e-8):
    """
    This function uses log-linear ordinary least square (OLS) method to find 
    the eigenvalues across time. We then compute our volatility surface at 
    every time stamp t.
    """
    F = pd.read_csv(fwd_csv).sort_values("t")
    Z = pd.read_csv(lat_csv).sort_values("t")
    M = pd.merge(F, Z, on="t", how="inner")
    ten = [c for c in F.columns if c!="t"]
    zcols = [c for c in Z.columns if c!="t"]

    Xf = M[ten].to_numpy(float);  Xf = Xf/100 if Xf.max()>1 else Xf
    Xf = np.maximum(Xf, eps)
    logf = np.log(Xf)

    f0_t = Xf[:,0]
    Y = logf - np.log(f0_t)[:,None]       
    Zm = M[zcols].to_numpy(float) # design matrix of the VAE latent factors          

    Omega = np.linalg.lstsq(Zm, Y, rcond=None)[0].T
    Y_hat = Zm @ Omega.T                     
    F_hat = f0_t[:,None] * np.exp(Y_hat)      
    Sigma = F_hat[:,:,None] * Omega[None,:,:] # compute volatility for each time t, tenor τ, factor p
    print(Sigma.shape)

    taus = np.array([parse_tenor(x) for x in ten], dtype = float)
    return F_hat, Xf, taus, Omega, Sigma, M["t"].to_numpy(), ten

def align_hist_fwd_rates(hist_csv: Path | str, times_ref: np.ndarray, ten_ref: list[str], eps: float = 1e-8) -> np.ndarray:
    """Load historical forward curves CSV and align time and tenor 
    to match grid used in VAE calibration."""
    H = pd.read_csv(hist_csv).sort_values("t")
    H = pd.merge(pd.DataFrame({"t": times_ref}), H, on="t", how="inner")
    ten_hist = [c for c in H.columns if c!="t"]
    ten_use  = [c for c in ten_ref if c in ten_hist]
    Xh = H[ten_use].to_numpy(float)
    Xh = Xh/100 if Xh.max()>1 else Xh
    Xh = np.maximum(Xh, eps)

    if len(ten_use) != len(ten_ref):
        J = len(ten_ref)
        Xpad = np.full((len(H), J), np.nan, float)
        col_map = {c:i for i,c in enumerate(ten_use)}
        for j,c in enumerate(ten_ref):
            if c in col_map: Xpad[:,j] = Xh[:, col_map[c]]
        Xh = Xpad
    return Xh

def drift_computation(Sigma_t: np.ndarray, taus: np.ndarray, deg: int = 3) -> np.ndarray:
    """
    Compute the drift of HJM SDE at every time stamp, t.
    """
    N,P = Sigma_t.shape
    alpha = np.zeros(N)
    for p in range(P):
        cp = np.polyfit(taus, Sigma_t[:,p], deg)
        sigma_tau  = np.polyval(cp, taus)
        integ_tau = np.array([np.trapz(sigma_tau[:j+1], taus[:j+1]) for j in range(N)])
        alpha += sigma_tau * integ_tau
    return alpha

def simulate_path(F_hat: np.ndarray, Sigma: np.ndarray, taus: np.ndarray, tgrid: np.ndarray, seed: int = 123):
    """
    Euler–Maruyama scheme with Musiela shift. We implement a dynamic 
    VAE volatility point estimator for various tenors across all time.
    """
    rng = np.random.default_rng(seed)
    T, N, P = Sigma.shape
    path = np.empty((T, N), float)
    path[0] = F_hat[0]                     
    Alpha = np.empty((T - 1, N), float)         

    for t in range(1, T):
        dt = tgrid[t] - tgrid[t - 1]
        fprev = path[t - 1]
        Sigma_current = Sigma[t - 1]
        alpha_hjm = drift_computation(Sigma_current, taus, deg=3)
        dfdtau = np.gradient(fprev, taus)
        diffusion = Sigma_current @ (rng.normal(size=P) * np.sqrt(dt))
        Alpha[t - 1] = alpha_hjm
        path[t] = fprev + (alpha_hjm + dfdtau) * dt + diffusion
        
    return path, Alpha

if __name__ == "__main__":
    F_hat, F_hist_calib, taus, Omega, Sigma, times, ten = vol_compute(FWD, LAT) # calculate volatility surface

    DPI, W_IN, H_IN = 100, 1573 / 100, 750 / 100
    fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)
    for row in F_hat:
        ax.plot(taus, row * 100.0, lw=0.8, alpha=0.7)

    TICK_FS = 27
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_FS)
    ax.xaxis.get_offset_text().set_size(TICK_FS)
    ax.yaxis.get_offset_text().set_size(TICK_FS)

    ax.set_xlabel("Maturity (Years)", fontsize=32)
    ax.set_ylabel("Interest Rate (%)", fontsize=32)
    ax.set_title("GBP Reconstructed Forward Curves", fontsize=37, fontweight="bold", pad=12)
    ax.set_xlim(left=0, right=float(np.nanmax(taus)))
    ax.set_ylim(-2, 10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("vae_reconstructed_fwd_curves.png", dpi=200)
    plt.close(fig)
    print("Saved figure to vae_reconstructed_fwd_curves.png")

    X_hist = align_hist_fwd_rates(HIST_FWD, times, ten) # load historical GBP forward data
    
    annualization = 252.0
    tgrid = np.arange(Sigma.shape[0]) / annualization
    sim_full, Alpha = simulate_path(F_hat, Sigma, taus, tgrid, seed=123) # simulate forward rates with dynamic volatility using HJM

    labels = ['1M', '6M', '1.0Y', '2.0Y', '3.0Y', '5.0Y', '10.0Y', '20.0Y', '25.0Y']
    dt = tgrid[1] - tgrid[0] if len(tgrid) > 1 else 1.0 / annualization

    sigma_csv = export_volatility(Sigma, taus, labels, dt, VAE_DIR / "vae_dynamic_volatility.csv")
    simulated_fwd_csv = export_simul_fwd(sim_full, taus, labels, dt, VAE_DIR / "vae_simulated_fwd_rates.csv")

    selected_tenors = np.array([parse_tenor(x) for x in labels]) # select tenors 1M, 6M,.., 20Y, 25Y for plotting
    idx = [int(np.where(np.isclose(taus, t))[0][0]) for t in selected_tenors]

    hist_simul = X_hist[:, idx]    
    sim_simul  = sim_full[:, idx] 
    
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    axes = axes.ravel()

    for j, ax in enumerate(axes):
        ax.plot(tgrid, hist_simul[:, j], lw=1.6, label="Historical")

        if j == 0:
            ax.plot(tgrid, sim_simul[:, j], lw=1.2, ls="--", color="red", label="no volatility")
        else:
            ax.plot(tgrid, sim_simul[:, j], lw=1.2, ls="--", label="Simulated")

        ax.set_title(labels[j], fontsize=13, fontweight="bold")
        if j % 3 == 0:
            ax.set_ylabel("f(t, T)")
        if j // 3 == 2:
            ax.set_xlabel("Time (years)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    fig.suptitle("Forward Rates: Simulated vs Historical", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("vae_fwd_rates_simulated_curves.png", dpi=150)
    plt.close(fig)
    print("Saved plot to vae_fwd_rates_simulated_curves.png")