# dynamic_hjm_simulation.py
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path("Chapter 3") / "vae"
FWD, LAT = BASE/"vae_reconstructed_fwd_rates.csv", BASE/"vae_fwd_latent_space.csv"
EPS, ANNUALIZATION, DT, POLY_DEG, RNG_SEED = 1e-8, 252.0, 1/252.0, 3, 123
TENOR_LABELS = ['1M','6M','1.0Y','2.0Y','3.0Y','5.0Y','10.0Y','20.0Y','25.0Y']

def parse_tenor(s):  # convert labels to years
    s=str(s).strip().upper()
    return float(s[:-1])/12 if s.endswith("M") else (float(s[:-1]) if s.endswith("Y") else float(s))

def calibrate_and_vol_surface(fwd_csv, lat_csv, eps=EPS):
    F, Z = pd.read_csv(fwd_csv).sort_values("t"), pd.read_csv(lat_csv).sort_values("t")
    M = pd.merge(F, Z, on="t", how="inner")
    ten = [c for c in F if c!="t"]; zcols = [c for c in Z if c!="t"]
    Xf = M[ten].to_numpy(float); Xf = Xf/100 if Xf.max()>1 else Xf
    Xf = np.maximum(Xf, eps); logf = np.log(Xf)
    f0_t = Xf[:,0]; Y = logf - np.log(f0_t)[:,None]   # demeaned log
    Zm = M[zcols].to_numpy(float)                     # (T,3)
    Lambda = np.linalg.lstsq(Zm, Y, rcond=None)[0].T  # (N,3)
    Y_hat = Zm @ Lambda.T; F_hat = f0_t[:,None]*np.exp(Y_hat)  # recon
    Sigma = F_hat[:,:,None] * Lambda[None,:,:]        # (T,N,3)
    taus = np.array([parse_tenor(x) for x in ten])
    return F_hat, Xf, taus, Lambda, Sigma

def drift_from_sigma(Sigma_t, taus, deg=3):
    N,K = Sigma_t.shape; mu = np.zeros(N)
    for k in range(K):
        ck = np.polyfit(taus, Sigma_t[:,k], deg)
        sigma_tau, integ_tau = np.polyval(ck, taus), np.polyval(np.polyint(ck), taus)
        mu += sigma_tau * integ_tau
    return mu

def simulate_path(F_hat, Sigma, taus, rng_seed=RNG_SEED):
    rng = np.random.default_rng(rng_seed)
    T,N,K = Sigma.shape; path = np.empty((T,N)); path[0] = F_hat[0]
    Mu = np.empty((T-1,N))
    for t in range(1,T):
        Sigma_tm1 = Sigma[t-1]; mu_tm1 = drift_from_sigma(Sigma_tm1, taus, POLY_DEG)
        Mu[t-1] = mu_tm1
        dW = rng.normal(size=K)*np.sqrt(DT)
        path[t] = path[t-1] + mu_tm1*DT + Sigma_tm1 @ dW
    return path, Mu

def plot_all_forward_curves(F, taus, title="Reconstructed forward curves", save_path="vae_reconstructed_forward_curves.png"):
    """2D plot of reconstructed forward curves with publication-style formatting."""
    # Match the dimensions from fwd_curves_plot
    DPI = 100
    W_IN = 1573 / DPI
    H_IN = 750  / DPI
    fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)

    # Plot each forward curve
    for row in F:
        ax.plot(taus, row*100, lw=0.8, alpha=0.7)

    # Tick and label formatting
    TICK_FS = 27
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_FS)
    ax.xaxis.get_offset_text().set_size(TICK_FS)
    ax.yaxis.get_offset_text().set_size(TICK_FS)

    # Labels and title
    ax.set_xlabel("Maturity (Years)", fontsize=32)
    ax.set_ylabel("Interest Rate (%)", fontsize=32)
    ax.set_title(title, fontsize=37, fontweight="bold", pad=12)

    # Axes limits
    ax.set_xlim(left=0, right=float(np.nanmax(taus)))
    ax.set_ylim(-2, 10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved figure to {save_path}")
    return save_path


def simulation_plots(tgrid, hist, sim, labels, save_path="vae_fwd_rates_simulated_curves.png"):
    fig, axes = plt.subplots(3,3, figsize=(13,9), sharex=True); axes = axes.ravel()
    for j, ax in enumerate(axes):
        # Historical line (black default)
        ax.plot(tgrid, hist[:,j], lw=1.6, label="Historical")
        # Simulated line (red only for subplot 1)
        if j == 0:
            ax.plot(tgrid, sim[:,j], lw=1.2, ls="--", color="red", label="Simulated")
        else:
            ax.plot(tgrid, sim[:,j], lw=1.2, ls="--", label="Simulated")
        ax.set_title(labels[j], fontsize=13, fontweight='bold')
        if j%3==0: ax.set_ylabel("f(t,T)")
        if j//3==2: ax.set_xlabel("Time (years)")
        ax.legend(fontsize=9)  # show legend in every subplot
        ax.grid(alpha=0.25)
    fig.suptitle("Forward Rates: Simulated vs Historical", fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"Saved plot to {save_path}")

if __name__=="__main__":
    F_hat, F_hist, taus, Lambda, Sigma = calibrate_and_vol_surface(FWD, LAT)
    plot_all_forward_curves(F_hat, taus, title="GBP Reconstructed Forward Curves")
    sim_full, Mu = simulate_path(F_hat, Sigma, taus)
    want = np.array([parse_tenor(x) for x in TENOR_LABELS])
    idx = [np.where(np.isclose(taus,t))[0][0] for t in want]
    hist_9, sim_9 = F_hat[:,idx], sim_full[:,idx]
    tgrid = np.arange(Sigma.shape[0]) * DT
    simulation_plots(tgrid, hist_9, sim_9, TENOR_LABELS)