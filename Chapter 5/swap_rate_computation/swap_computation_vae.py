"""
This Python script derives the LIBOR swap rates using simulated rates derived from VAE methodology.
"""

import numpy as np, pandas as pd

def parse_tenor(s: str) -> float:
    """
    Converts all tenors into years.
    """
    s = s.strip().upper()
    if s.endswith("M"): return float(s[:-1]) / 12.0
    if s.endswith("Y"): return float(s[:-1])
    return float(s)

def bond_price(Tgrid: np.ndarray, fwd_grid: np.ndarray):
    """
    Bond price computation using Musiela convention. We interpolate across various tenors and 
    use integral to compute the bond price.
    """
    Tgrid = np.asarray(Tgrid, float)
    fwd_grid = np.asarray(fwd_grid, float)
    sorted_index = np.argsort(Tgrid); Tgrid, fwd_grid = Tgrid[sorted_index], fwd_grid[sorted_index]

    def P(T):
        T = np.atleast_1d(T).astype(float)

        def one(z):
            tenors_t = np.r_[0.0, Tgrid[Tgrid < z], z]
            fwd_rate_t = np.interp(np.clip(tenors_t, Tgrid[0], Tgrid[-1]), Tgrid, fwd_grid) # linear interpolation across the tenors
            return np.exp(-np.trapz(fwd_rate_t, tenors_t))

        out = np.fromiter((one(z) for z in T), float) # out a numpy array of bond prices for varying maturities
        return out[0] if out.size == 1 else out

    return P

def swap_rate_computation(csv_path: str):
    """
    swap_mats : list of np.ndarray
        Each with shape (19, 7). One swap-rate matrix per timestamp in the CSV.
    annuity_mats : list of np.ndarray
        Each with shape (19, 7). Matching Annuity matrix per timestamp.
    """
    df = pd.read_csv(csv_path)

    tenor_cols = [c for c in df.columns if c.lower() != "t"]
    Tgrid = np.array([parse_tenor(c) for c in tenor_cols], float)
    order = np.argsort(Tgrid)
    Tgrid, tenor_cols = Tgrid[order], [tenor_cols[i] for i in order]

    Expiry = np.array([1/12, 3/12, 6/12, 9/12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30], float)
    Tenor = np.array([1, 2, 3, 5, 10, 20, 25], float)

    swap_mats = []
    annuity_mats = []

    for _, r in df.iterrows():
        fgrid = r[tenor_cols].to_numpy(float)
        P = bond_price(Tgrid, fgrid)

        M = np.empty((Expiry.size, Tenor.size), float)  
        A = np.empty((Expiry.size, Tenor.size), float)  

        for i, Ti in enumerate(Expiry):
            for j, m in enumerate(Tenor):
                Tj = Ti + m
                dt = 0.5  # index tenor
                first = np.ceil(Ti / dt) * dt

                if first > Tj:
                    pay_times = np.array([Tj])
                else:
                    pay_times = np.arange(first, Tj + 1e-12, dt)
                    if abs(pay_times[-1] - Tj) > 1e-12:
                        pay_times = np.r_[pay_times, Tj]

                accr = np.diff(np.r_[Ti, pay_times])

                Annuity = float(np.sum(accr * P(pay_times)))

                if Annuity <= 0:
                    M[i, j] = np.nan
                    A[i, j] = np.nan
                else:
                    A[i, j] = Annuity
                    M[i, j] = (float(P(Ti)) - float(P(Tj))) / Annuity

        swap_mats.append(M)
        annuity_mats.append(A)

    return swap_mats, annuity_mats

if __name__ == "main": 
    path = r"Chapter 5\swap_rate_computation\data\vae_simulated_rates_selected.csv" 
    swap_mats, annuity_mats = swap_rate_computation(path) 
    print(f"Got {len(swap_mats)} swap matrices. Example shape: {swap_mats[0].shape if swap_mats else None}") 
    print(f"Got {len(annuity_mats)} annuity matrices. Example shape: {annuity_mats[0].shape if annuity_mats else None}") 
    for i, (M, A) in enumerate(zip(swap_mats, annuity_mats), start=1): 
        print(f"\n=== Timestamp {i} ===") 
        print("Swap Rate Matrix:") 
        print(M) 
        print("\nAnnuity Matrix:") 
        print(A)