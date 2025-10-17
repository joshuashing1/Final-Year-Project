# simulate.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Core helpers ----------------
def parse_tenor(s: str) -> float:
    s = s.strip().upper()
    if s.endswith("M"): return float(s[:-1]) / 12.0
    if s.endswith("Y"): return float(s[:-1])
    return float(s)

def simulate_path4(
    F_hat: np.ndarray,      # (T, N) – used for initial curve & length
    Sigma: np.ndarray,      # (T, N) – σ(t, τ) from vae_dynamic_volatility.csv, aligned to F_hat
    taus: np.ndarray,       # (N,)   – tenor grid in years (ascending)
    r_mean: float,          # scalar  ⟨r_t⟩ over all timesteps
    fQ_mean: np.ndarray,    # (N,)    tenorwise ⟨f^Q(·,·+τ)⟩ over all timesteps
    rng_seed: int = 42,
    tgrid: np.ndarray | None = None,   # (T,) time in years; if None uses DT
    DT: float | None = 1.0/252.0       # uniform dt if tgrid is None
) -> np.ndarray:
    """
    df_t(τ) = [ σ_t(τ) + (⟨r⟩ - ⟨f^Q⟩)/(t+τ) + ∂f/∂τ ] dt  +  σ_t(τ) dB_t(τ)
    Returns path: (T, N)
    """
    rng = np.random.default_rng(rng_seed)
    T, N = F_hat.shape
    assert Sigma.shape == (T, N), f"Sigma {Sigma.shape} must match F_hat {F_hat.shape}"
    assert fQ_mean.shape == (N,), f"fQ_mean must be shape (N,), got {fQ_mean.shape}"

    use_tgrid = tgrid is not None
    if use_tgrid:
        assert len(tgrid) == T, "tgrid length must equal T"
    else:
        assert DT and DT > 0, "Provide DT>0 or a valid tgrid"

    path = np.empty((T, N), float)
    path[0] = F_hat[0].copy()
    eps = 1e-12

    for it in range(1, T):
        if use_tgrid:
            t_prev = float(tgrid[it-1])
            dt     = float(tgrid[it] - tgrid[it-1])
        else:
            t_prev = float((it-1) * DT)
            dt     = float(DT)

        fprev  = path[it-1]                # f(t_prev, ·)
        dfdtau = np.gradient(fprev, taus)  # Musiela shift ∂f/∂τ

        # Risk-premium term: (⟨r⟩ - ⟨f^Q⟩)/(t+τ)
        denom     = np.maximum(t_prev + taus, eps)
        risk_prem = (r_mean - fQ_mean) / denom

        # Drift and diffusion using left-endpoint σ(t_prev, ·)
        sigma_now = Sigma[it-1]            # (N,)
        drift     = sigma_now + risk_prem + dfdtau

        dW        = rng.normal(size=N) * np.sqrt(dt)
        diffusion = sigma_now * dW         # diagonal diffusion

        path[it]  = fprev + drift * dt + diffusion

    return path

# ---------------- Data + run ----------------
LABELS = ['1M','6M','1.0Y','2.0Y','3.0Y','5.0Y','10.0Y','20.0Y','25.0Y']
DAY2YR = 252.0

# Load datasets
fQ_df   = pd.read_csv(r"Chapter 4\data\simulated_fwd_rates_Q_msr.csv").set_index("t").sort_index()
rHist   = pd.read_csv(r"Chapter 4\data\short_rate.csv").set_index("t").sort_index()["r_t"]
df_hist = pd.read_csv(r"Chapter 4\data\GLC_fwd_curve_raw.csv").set_index("t").sort_index() / 100.0
sigma_df = pd.read_csv(r"Chapter 4\data\vae_dynamic_volatility.csv").set_index("t").sort_index()

# Align timeline and enforce T = 1264
t = fQ_df.index.intersection(rHist.index).intersection(df_hist.index).intersection(sigma_df.index)
assert len(t) == 1264, f"Aligned timeline has {len(t)} steps, expected 1264. Check your CSV indices."

# Select common tenor set and arrays
F_hat = fQ_df.loc[t, LABELS].to_numpy(float)              # (T, N)
Sigma = sigma_df.loc[t, LABELS].to_numpy(float)           # (T, N) time-varying σ
taus  = np.array([parse_tenor(x) for x in LABELS], float) # (N,)
hist_path = df_hist.loc[t, LABELS].to_numpy(float)        # (T, N) for plotting only

# Means required by the SDE
r_mean  = float(rHist.loc[t].mean())                      # scalar ⟨r_t⟩
fQ_mean = fQ_df.loc[t, LABELS].to_numpy(float).mean(axis=0)  # (N,) tenorwise ⟨f^Q⟩

# Time grid in years (monotone, starting at 0)
timeline_years = (t.to_numpy(float) - t.min()) / DAY2YR

print("\nQ-measure forward rates (first few rows):")
print(pd.DataFrame(F_hat, index=t, columns=LABELS).head())

# Simulate under P using the requested SDE
sim_path = simulate_path4(
    F_hat=F_hat,
    Sigma=Sigma,
    taus=taus,
    r_mean=r_mean,
    fQ_mean=fQ_mean,
    rng_seed=42,
    tgrid=timeline_years,  # precise dt from your index spacing
    DT=None                # ignored when tgrid is provided
)

# Save results
out = pd.DataFrame(sim_path, index=t, columns=LABELS)
out.index.name = "t"
out.to_csv("simulated_forward_P_measure.csv")

# Side-by-side plots (historical vs simulated) for the 9 selected tenors
fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
for j, ax in enumerate(axes.ravel()):
    ax.plot(timeline_years, hist_path[:, j], label="Historical", lw=1.3)
    ax.plot(timeline_years, sim_path[:, j],  label="Simulated", ls="--", lw=1.0)
    ax.set_title(LABELS[j], fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel(r"$f(t, T)$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
fig.suptitle("Forward Rates: Simulated vs Historical", fontsize=16, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("fwd_rates_simulated_p.png", dpi=150)
plt.show()
