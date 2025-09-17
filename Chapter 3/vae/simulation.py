# simulate_dynamic_hjm.py
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# ---------------- config ----------------
BASE = Path("Chapter 3") / "vae"
FWD  = BASE / "vae_reconstructed_fwd_rates.csv"
LAT  = BASE / "vae_fwd_latent_space.csv"

ANNUALIZATION = 252.0
DT = 1.0 / ANNUALIZATION
POLY_DEG = 3
RNG_SEED = 123
TENOR_LABELS = ['1M','6M','1.0Y','2.0Y','3.0Y','5.0Y','10.0Y','20.0Y','25.0Y']

def parse_tenor(s):
    s=str(s).strip().upper()
    return float(s[:-1])/12 if s.endswith("M") else (float(s[:-1]) if s.endswith("Y") else float(s))

def maturities_from_cols(cols):
    tenors=[c for c in cols if c!='t']
    taus=np.array([parse_tenor(c) for c in tenors], float)
    order=np.argsort(taus)
    return [tenors[i] for i in order], taus[order]

def build_sigmas(fwd_csv: Path, lat_csv: Path, window=1):
    F = pd.read_csv(fwd_csv).sort_values("t").reset_index(drop=True)
    Z = pd.read_csv(lat_csv).sort_values("t").reset_index(drop=True)
    if "t" not in F or "t" not in Z: raise ValueError("Both CSVs need a 't' column.")
    tenor_cols, taus = maturities_from_cols(F.columns); F = F[["t"]+tenor_cols]
    M = F.merge(Z, on="t", how="inner")
    z_cols=[c for c in Z.columns if c!="t"]

    Fm = M[tenor_cols].to_numpy(float)/100.0         # (T,N)
    Zm = M[z_cols].to_numpy(float)             # (T,K)
    dF = Fm[1:]-Fm[:-1]                        # (T-1,N)
    dZ = Zm[1:]-Zm[:-1]                        # (T-1,K)
    times = M["t"].to_numpy()[1:]              # (T-1,)

    Tm1,N = dF.shape; K = dZ.shape[1]
    Sig = np.zeros((Tm1,N,K))
    reg = LinearRegression(fit_intercept=False)
    for i in range(Tm1):
        s0=max(0, i-(window-1)); s1=i+1
        reg.fit(dZ[s0:s1], dF[s0:s1])         # multi-target
        Sig[i] = reg.coef_                    # (N,K)
    return Sig, times, tenor_cols, taus, Fm   # return Fm to get historical path

def drift_from_sigma_rowwise(Sigma_t: np.ndarray, taus: np.ndarray, deg=3):
    """For a fixed t: fit cubic σ_k(τ) across τ for each factor, compute μ_i = Σ_k σ_k(τ_i)∫_0^{τ_i}σ_k(u)du."""
    N,K = Sigma_t.shape
    mu = np.zeros(N)
    for k in range(K):
        ck = np.polyfit(taus, Sigma_t[:,k], deg)      # descending powers
        pin = np.polyint(ck)                          # primitive
        sigma_tau = np.polyval(ck, taus)              # σ_k(τ_i)
        integ_tau = np.polyval(pin, taus)             # ∫_0^{τ_i} σ_k(u) du  (pin(0)=0)
        mu += sigma_tau * integ_tau
    return mu

def simulate_path(F_hist_sorted: np.ndarray, Sig: np.ndarray, taus: np.ndarray, rng_seed=RNG_SEED):
    """Euler: f_{t} = f_{t-1} + μ_{t-1} dt + Σ_{t-1} dW_t with time-varying μ,Σ."""
    rng = np.random.default_rng(rng_seed)
    T,N,K = Sig.shape[0]+1, Sig.shape[1], Sig.shape[2]
    path = np.empty((T,N)); path[0] = F_hist_sorted[0]
    Mu = np.empty((T-1,N))
    for t in range(1,T):
        Sigma_tm1 = Sig[t-1]                      # (N,K)
        mu_tm1 = drift_from_sigma_rowwise(Sigma_tm1, taus, POLY_DEG)
        Mu[t-1] = mu_tm1
        dW = rng.normal(size=K)*np.sqrt(DT)
        path[t] = path[t-1] + mu_tm1*DT + Sigma_tm1 @ dW
    return path, Mu

def plot_9panels(tgrid, hist, sim, labels, save_path: str):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,3, figsize=(13,9), sharex=True)
    axes = axes.ravel()
    for j, ax in enumerate(axes):
        ax.plot(tgrid, hist[:,j], label="Historical", lw=1.6)
        ax.plot(tgrid, sim[:,j],  label="Simulated",  lw=1.2, ls="--")
        ax.set_title(labels[j], fontsize=14, fontweight='bold')
        if j%3==0: ax.set_ylabel("f(t,T)")
        if j//3==2: ax.set_xlabel("Time t (years)")
        if j==1: ax.legend()
        ax.grid(alpha=0.25)
        ax.legend(fontsize=10)
    fig.suptitle("Forward Rates: Simulated vs Historical", fontsize=18, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {save_path}")
    return save_path
    
# ---------------- main ----------------
if __name__ == "__main__":
    # 1) dynamic Σ_t and sorted objects
    Sig, times, tenor_cols_sorted, taus_sorted, F_sorted = build_sigmas(FWD, LAT, window=1)
    T = Sig.shape[0] + 1

    # 2) simulate full path using dynamic basis
    sim_full, Mu = simulate_path(F_sorted, Sig, taus_sorted, rng_seed=RNG_SEED)

    # 3) pick the 9 requested tenors and make the comparison plot
    want = np.array([parse_tenor(x) for x in TENOR_LABELS])
    # indices of requested maturities in the sorted grid
    idx = [int(np.where(np.isclose(taus_sorted, t))[0][0]) for t in want]
    hist_9 = F_sorted[:, idx]
    sim_9  = sim_full[:, idx]

    tgrid_years = np.arange(T) * (1.0/ANNUALIZATION)
    plot_9panels(tgrid_years, hist_9, sim_9, TENOR_LABELS, save_path="vae_forward_rates_simulated_curves.png")
