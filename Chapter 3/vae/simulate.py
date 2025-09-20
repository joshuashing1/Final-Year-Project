# dynamic_hjm_simulation.py
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path("Chapter 3")
VAE_DIR = BASE / "vae"
DATA_DIR = BASE / "data"
FWD, LAT  = VAE_DIR/"vae_reconstructed_fwd_rates.csv", VAE_DIR/"vae_fwd_latent_space.csv"
HIST_FWD  = DATA_DIR/"GLC_fwd_curve_raw.csv"

EPS, ANNUALIZATION, DT, POLY_DEG, RNG_SEED = 1e-8, 252.0, 1/252.0, 3, 123
TENOR_LABELS = ['1M','6M','1.0Y','2.0Y','3.0Y','5.0Y','10.0Y','20.0Y','25.0Y']

def parse_tenor(s):
    s=str(s).strip().upper()
    return float(s[:-1])/12 if s.endswith("M") else (float(s[:-1]) if s.endswith("Y") else float(s))

def calibrate_and_vol_surface(fwd_csv, lat_csv, eps=EPS):
    F = pd.read_csv(fwd_csv).sort_values("t")
    Z = pd.read_csv(lat_csv).sort_values("t")
    M = pd.merge(F, Z, on="t", how="inner")
    ten   = [c for c in F.columns if c!="t"]
    zcols = [c for c in Z.columns if c!="t"]

    Xf = M[ten].to_numpy(float);  Xf = Xf/100 if Xf.max()>1 else Xf
    Xf = np.maximum(Xf, eps)
    logf = np.log(Xf)

    # level per time = first tenor
    f0_t = Xf[:,0]
    Y    = logf - np.log(f0_t)[:,None]       # (T,N)
    Zm   = M[zcols].to_numpy(float)          # (T,3)

    # OLS across time for each tenor
    Lambda = np.linalg.lstsq(Zm, Y, rcond=None)[0].T  # (N,3)

    # Reconstruct forward curves & vol surface
    Y_hat = Zm @ Lambda.T                     # (T,N)
    F_hat = f0_t[:,None] * np.exp(Y_hat)      # (T,N)
    Sigma = F_hat[:,:,None] * Lambda[None,:,:]# (T,N,3)

    taus = np.array([parse_tenor(x) for x in ten])
    return F_hat, Xf, taus, Lambda, Sigma, M["t"].to_numpy(), ten

def align_historical_forward_curves(hist_csv, times_ref, ten_ref, eps=EPS):
    """Load historical forward curves CSV and align rows by time and columns by tenor to match calibration."""
    H = pd.read_csv(hist_csv).sort_values("t")
    # keep only reference times
    H = pd.merge(pd.DataFrame({"t": times_ref}), H, on="t", how="inner")
    # intersect tenors in case of mismatch
    ten_hist = [c for c in H.columns if c!="t"]
    ten_use  = [c for c in ten_ref if c in ten_hist]
    Xh = H[ten_use].to_numpy(float)
    Xh = Xh/100 if Xh.max()>1 else Xh
    Xh = np.maximum(Xh, eps)

    # If some requested tenors are missing, pad with NaNs to preserve order
    if len(ten_use) != len(ten_ref):
        J = len(ten_ref)
        Xpad = np.full((len(H), J), np.nan, float)
        col_map = {c:i for i,c in enumerate(ten_use)}
        for j,c in enumerate(ten_ref):
            if c in col_map: Xpad[:,j] = Xh[:, col_map[c]]
        Xh = Xpad
    return Xh

def drift_from_sigma(Sigma_t, taus, deg=3):
    N,K = Sigma_t.shape
    mu = np.zeros(N)
    for k in range(K):
        ck = np.polyfit(taus, Sigma_t[:,k], deg)
        sigma_tau  = np.polyval(ck, taus)
        integ_tau  = np.polyval(np.polyint(ck), taus)
        mu += sigma_tau * integ_tau
    return mu

def simulate_path(F_hat, Sigma, taus, rng_seed=RNG_SEED):
    """
    Euler–Maruyama with explicit Musiela shift:
        f_{t+dt}(τ) = f_t(τ) + [ α_HJM(t,τ) + ∂f/∂τ (t,τ) ] dt + Σ(t,τ) dW_t
    where α_HJM comes from drift_from_sigma(Sigma_t, taus, POLY_DEG)
    and Σ(t,τ) is the tenor-by-factor vol surface.
    """
    rng = np.random.default_rng(rng_seed)
    T, N, K = Sigma.shape
    path = np.empty((T, N), float)
    path[0] = F_hat[0]                     # initial curve
    Mu = np.empty((T-1, N), float)         # store α_HJM (optional)

    for t in range(1, T):
        fprev = path[t-1]                  # f(t-1, τ_•)
        Sigma_tm1 = Sigma[t-1]             # (N, K)
        alpha_hjm = drift_from_sigma(Sigma_tm1, taus, POLY_DEG)  # (N,)
        dfdtau = np.gradient(fprev, taus)  # ∂f/∂τ at current curve, vectorized

        # Brownian shocks for K factors
        dW = rng.normal(size=K) * np.sqrt(DT)   # (K,)
        diffusion = Sigma_tm1 @ dW              # (N,)

        # Advance one step
        Mu[t-1] = alpha_hjm
        path[t] = fprev + (alpha_hjm + dfdtau) * DT + diffusion

    return path, Mu


def plot_all_forward_curves(F, taus, title="Reconstructed forward curves", save_path="vae_reconstructed_fwd_curves.png"):
    DPI, W_IN, H_IN = 100, 1573/100, 750/100
    fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)
    for row in F: ax.plot(taus, row*100, lw=0.8, alpha=0.7)
    TICK_FS = 27
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_FS)
    ax.xaxis.get_offset_text().set_size(TICK_FS); ax.yaxis.get_offset_text().set_size(TICK_FS)
    ax.set_xlabel("Maturity (Years)", fontsize=32); ax.set_ylabel("Interest Rate (%)", fontsize=32)
    ax.set_title(title, fontsize=37, fontweight="bold", pad=12)
    ax.set_xlim(left=0, right=float(np.nanmax(taus))); ax.set_ylim(-2, 10); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)
    print(f"Saved figure to {save_path}"); return save_path

def simulation_plots(tgrid, hist, sim, labels, save_path="vae_fwd_rates_simulated_curves.png"):
    fig, axes = plt.subplots(3,3, figsize=(13,9), sharex=True); axes = axes.ravel()
    for j, ax in enumerate(axes):
        ax.plot(tgrid, hist[:,j], lw=1.6, label="Historical")
        if j == 0: ax.plot(tgrid, sim[:,j], lw=1.2, ls="--", color="red", label="Simulated")
        else:      ax.plot(tgrid, sim[:,j], lw=1.2, ls="--", label="Simulated")
        ax.set_title(labels[j], fontsize=13, fontweight='bold')
        if j%3==0: ax.set_ylabel("f(t,T)")
        if j//3==2: ax.set_xlabel("Time (years)")
        ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.suptitle("Forward Rates: Simulated vs Historical", fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"Saved plot to {save_path}")

if __name__=="__main__":
    # 1) Calibrate on VAE outputs (keeps simulation logic identical)
    F_hat, F_hist_calib, taus, Lambda, Sigma, times, ten = calibrate_and_vol_surface(FWD, LAT)

    # 2) Plot calibrated forward curves (fan)
    plot_all_forward_curves(F_hat, taus, title="GBP Reconstructed Forward Curves")

    # 3) Load REAL historical curves for plotting (aligned to calibration times & tenors)
    X_hist = align_historical_forward_curves(HIST_FWD, times, ten)

    # 4) Simulate with calibrated Sigma (unchanged logic)
    sim_full, Mu = simulate_path(F_hat, Sigma, taus)

    # 5) Select the 9 labeled tenors and plot hist vs sim (hist now from HIST_FWD)
    want = np.array([parse_tenor(x) for x in TENOR_LABELS])
    # map desired maturities into the taus grid
    idx = [int(np.where(np.isclose(taus, t))[0][0]) for t in want]
    hist_9 = X_hist[:, idx]   # <- historical from raw CSV (scaled to decimals)
    sim_9  = sim_full[:, idx]

    tgrid = np.arange(Sigma.shape[0]) * DT
    simulation_plots(tgrid, hist_9, sim_9, TENOR_LABELS)
