import numpy as np, pandas as pd, matplotlib.pyplot as plt

CSV_FWD = r"Chapter 3\data\GLC_fwd_curve_raw.csv"
CSV_LAT = r"Chapter 3\vae\vae_fwd_latent_space.csv"
ANNUALIZATION, WINDOW, RNG_SEED = 252.0, 10, 123
TENOR_LABELS = ['1M','6M','1.0Y','2.0Y','3.0Y','5.0Y','10.0Y','20.0Y','25.0Y']  # use '30.0Y' if present

def parse_tenor(s): s=s.strip().upper(); return float(s[:-1])/12 if s.endswith('M') else (float(s[:-1]) if s.endswith('Y') else float(s))
def zfit(X): mu=X.mean(0); sd=X.std(0, ddof=0); sd[sd==0]=1; return mu,sd
def zapp(X,mu,sd): return (X-mu)/sd
def zinv(Z,mu,sd): return Z*sd+mu

def select_tenors(df, labels):
    taus = np.array([parse_tenor(c) for c in df.columns], float)
    order = np.argsort(taus); df = df.iloc[:, order]; taus = taus[order]
    pick = np.array([parse_tenor(x) for x in labels], float)
    idx = [int(np.where(np.isclose(taus, t))[0][0]) for t in pick]
    return df.iloc[:, idx].copy(), pick

def sigma_local_ols(Fz, Z, t, w):
    """Σ_t (N×K) via local OLS on ΔF ≈ ΔZ B over a lookback window."""
    N = Fz.shape[1]; K = Z.shape[1]
    if t < 2:
        return np.zeros((N, K))
    s0 = max(1, t - w); s1 = t
    dZ = Z[s0:s1]  - Z[s0-1:s1-1]   # (m,K)
    dF = Fz[s0:s1] - Fz[s0-1:s1-1]  # (m,N)
    B, *_ = np.linalg.lstsq(dZ, dF, rcond=None)  # (K,N)
    return B.T                                     # (N,K)

def hjm_drift(Sig, taus):
    n,K = Sig.shape; mu = np.zeros(n)
    for k in range(K):
        s = Sig[:,k]
        integ = np.zeros(n)
        for i in range(1,n):
            dτ = taus[i]-taus[i-1]
            integ[i] = integ[i-1] + 0.5*(s[i]+s[i-1])*dτ
        mu += s*integ
    return mu

def main():
    rng = np.random.default_rng(RNG_SEED)

    # 1) Load forward curves, divide tenor columns by 100 (keep 'time' untouched)
    df = pd.read_csv(CSV_FWD)
    assert 'time' in df.columns, "Expected a 'time' column."
    tenor_cols = [c for c in df.columns if c != 'time']
    df[tenor_cols] = df[tenor_cols] / 100.0  # <-- scale to decimals, like your other script
    df = df.drop(columns=['time'])

    # 2) Select requested tenors (sorted by maturity)
    df_sel, taus = select_tenors(df, TENOR_LABELS)  # [T, 9]
    F = df_sel.to_numpy(float)

    # 3) Standardize across time per tenor (work in standardized space)
    muF, sdF = zfit(F)
    Fz = zapp(F, muF, sdF)

    # 4) Load latent factors (assumed aligned with the same T)
    lat = pd.read_csv(CSV_LAT)
    Z = lat[['z1','z2','z3']].to_numpy(float)
    assert Z.shape[0] == Fz.shape[0], f"Length mismatch: Z={Z.shape[0]} vs F={Fz.shape[0]}"

    T, N = Fz.shape; K = Z.shape[1]; dt = 1.0/ANNUALIZATION
    tgrid = np.arange(T)*dt

    # 5) Build Σ_t and μ_t; backfill first steps with first valid Σ
    Sig = np.zeros((T,N,K)); Mu = np.zeros((T,N))
    first_valid = None
    for t in range(T):
        Sig[t] = sigma_local_ols(Fz, Z, t, WINDOW)
        if first_valid is None and np.any(Sig[t]): first_valid = Sig[t].copy()
        use_Sig = Sig[t] if np.any(Sig[t]) else (first_valid if first_valid is not None else np.zeros((N,K)))
        Mu[t] = hjm_drift(use_Sig, taus)
    if first_valid is not None:
        Sig[:2] = first_valid

    # 6) Simulate in standardized space
    pathz = np.empty((T,N)); pathz[0] = Fz[0]
    for t in range(1,T):
        dW = rng.normal(size=K)*np.sqrt(dt)
        pathz[t] = pathz[t-1] + Mu[t-1]*dt + Sig[t-1]@dW

    # 7) Back to original (decimal) units for plotting
    path = zinv(pathz, muF, sdF)

    # 8) Plot historical vs simulated
    fig, axes = plt.subplots(3,3, figsize=(14,10), sharex=True)
    for j, ax in enumerate(axes.ravel()):
        ax.plot(tgrid, F[:,j],  lw=1.5, label="Historical")
        ax.plot(tgrid, path[:,j], ls="--", lw=1.0, label="Simulated")
        ax.set_title(TENOR_LABELS[j], fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3); ax.tick_params(labelsize=10)
        if j%3==0: ax.set_ylabel("f(t,T) (decimal)")  # now decimals because of /100
        if j//3==2: ax.set_xlabel("Time (years)")
        if j==0: ax.legend(fontsize=9)
    fig.suptitle("Forward Rates: Simulated vs Historical (VAE-Jac via Local OLS)", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0,0,1,0.96]); plt.show()

if __name__ == "__main__":
    main()
