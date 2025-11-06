import os
import numpy as np
import pandas as pd

# ---------- utilities ----------
def parse_tenor(s: str) -> float:
    s = str(s).strip().upper()
    if s.endswith("M"):
        return float(s[:-1]) / 12.0
    if s.endswith("Y"):
        return float(s[:-1])
    return float(s)

def bond_price(Tgrid: np.ndarray, fgrid: np.ndarray):
    """
    Build P(T) from an instantaneous forward-rate grid (Tgrid, fgrid)
    via linear interpolation in f and trapezoidal integration.
    """
    Tgrid = np.asarray(Tgrid, float)
    fgrid = np.asarray(fgrid, float)
    o = np.argsort(Tgrid); Tgrid, fgrid = Tgrid[o], fgrid[o]

    def P(T):
        T = np.atleast_1d(T).astype(float)

        def one(z):
            knots = np.r_[0.0, Tgrid[Tgrid < z], z]
            vals  = np.interp(np.clip(knots, Tgrid[0], Tgrid[-1]), Tgrid, fgrid)
            return np.exp(-np.trapz(vals, knots))

        out = np.fromiter((one(z) for z in T), float)
        return out[0] if out.size == 1 else out

    return P

# ---------- IO helpers ----------
def read_rates_csv(csv_path: str) -> pd.DataFrame:
    """
    Read the RATES timeseries file. Must contain 't' column.
    """
    df = pd.read_csv(csv_path, comment="#", skip_blank_lines=True, engine="python")
    df.columns = [str(c).encode("utf-8").decode("utf-8-sig").strip() for c in df.columns]

    if "t" not in [c.lower() for c in df.columns]:
        raise ValueError(
            f"Expected a 't' column in the rates file, but did not find one. "
            f"You're probably pointing to the initial-vol surface by mistake.\nFile: {csv_path}"
        )

    df = df.sort_values(by=[c for c in df.columns if c.lower()=="t"][0], kind="mergesort").reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No data rows found in rates file: {csv_path}")
    return df

def read_initial_vol_csv(csv_path: str) -> pd.DataFrame:
    """
    Read the INITIAL VOL surface file. Must contain 'expiry' column.
    """
    df = pd.read_csv(csv_path, comment="#", skip_blank_lines=True, engine="python")
    df.columns = [str(c).encode("utf-8").decode("utf-8-sig").strip() for c in df.columns]

    if "expiry" not in [c.lower() for c in df.columns]:
        raise ValueError(
            f"Expected an 'expiry' column in the initial-vol file, but did not find one. "
            f"You're probably pointing to the rates file by mistake.\nFile: {csv_path}"
        )
    return df

# ---------- core computations ----------
def swap_rate_computation_from_df(df: pd.DataFrame):
    """
    Build a list of swap-rate matrices, one per timestamp from a prepared RATES DataFrame.
    Returns: matrices, Expiry, Tenor, t_vals
    """
    t_col = [c for c in df.columns if c.lower()=="t"][0]
    t_vals = df[t_col].to_numpy()

    # tenor columns = everything except 't'
    ten_cols = [c for c in df.columns if c != t_col]
    Tgrid = np.array([parse_tenor(c) for c in ten_cols], float)
    order = np.argsort(Tgrid)
    Tgrid, ten_cols = Tgrid[order], [ten_cols[i] for i in order]

    # matrix grids (expiry rows, tenor cols)
    Expiry = np.array([1/12, 3/12, 6/12, 9/12, 1, 2, 3, 4, 5, 6,
                       7, 8, 9, 10, 12, 15, 20, 25, 30], float)
    Tenor  = np.array([1, 2, 3, 5, 10, 20, 25], float)

    matrices = []

    for _, r in df.iterrows():
        fgrid = pd.to_numeric(r[ten_cols], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(fgrid).all():
            raise ValueError("Non-numeric forward rates detected in rates CSV row. Check data cleanliness.")
        P = bond_price(Tgrid, fgrid)

        M = np.empty((Expiry.size, Tenor.size), float)
        for i, Ti in enumerate(Expiry):
            for j, m in enumerate(Tenor):
                Tj = Ti + m
                dt = 0.5  # semiannual fixed leg
                first = np.ceil(Ti / dt) * dt
                if first > Tj:
                    pay_times = np.array([Tj])
                else:
                    pay_times = np.arange(first, Tj + 1e-12, dt)
                    if abs(pay_times[-1] - Tj) > 1e-12:
                        pay_times = np.r_[pay_times, Tj]
                accr = np.diff(np.r_[Ti, pay_times])
                A = float(np.sum(accr * P(pay_times)))  # annuity
                M[i, j] = np.nan if A <= 0 else (float(P(Ti)) - float(P(Tj))) / A

        matrices.append(M)

    return matrices, Expiry, Tenor, t_vals

def read_initial_vol_matrix(initial_vol_df: pd.DataFrame, Expiry: np.ndarray, Tenor: np.ndarray) -> np.ndarray:
    """
    Align initial vol surface (expiry rows × tenor cols) to (Expiry, Tenor) grid.
    """
    df = initial_vol_df.copy()
    # normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    # ensure first column is 'expiry'
    if "expiry" not in [c.lower() for c in df.columns]:
        df = df.rename(columns={df.columns[0]: "expiry"})
    df["__exp_years__"] = df["expiry"].map(parse_tenor)
    df = df.set_index("__exp_years__").drop(columns=["expiry"])
    df.columns = [parse_tenor(c) for c in df.columns]
    df = df.sort_index().sort_index(axis=1)

    aligned = df.reindex(index=Expiry, columns=Tenor)
    if aligned.isna().any().any():
        missing = np.argwhere(np.isnan(aligned.to_numpy()))
        raise ValueError(f"Initial volatility grid missing values at (expiry, tenor): {missing}")
    return aligned.to_numpy(float)  # vols

def ewma_implied_volatility(mats: list[np.ndarray],
                            initial_vol_matrix: np.ndarray,
                            lam: float = 0.94) -> list[np.ndarray]:
    """
    EWMA per cell: var_{t+1} = lam * var_t + (1 - lam) * S_t^2
    Returns a list of volatility matrices (sqrt(var)) with same length as mats.
    """
    if not mats:
        return []
    init = np.asarray(initial_vol_matrix, float)
    if mats[0].shape != init.shape:
        raise ValueError(f"Shape mismatch: swap matrix {mats[0].shape} vs initial vol {init.shape}")
    prev_var = init ** 2
    vols = []
    for S in mats:
        S = np.asarray(S, float)
        new_var = lam * prev_var
        valid = np.isfinite(S)
        new_var[valid] += (1.0 - lam) * (S[valid] ** 2)
        new_var = np.maximum(new_var, 0.0)
        vols.append(np.sqrt(new_var))
        prev_var = new_var  # carry forward
    return vols

def compute_swap_and_vol(rates_csv: str, initial_vol_csv: str, lam: float = 0.94):
    # Read & verify
    df_rates = read_rates_csv(rates_csv)
    df_initv = read_initial_vol_csv(initial_vol_csv)

    swap_mats, Expiry, Tenor, t_vals = swap_rate_computation_from_df(df_rates)
    init_vol = read_initial_vol_matrix(df_initv, Expiry, Tenor)
    vol_mats = ewma_implied_volatility(swap_mats, init_vol, lam)

    if len(swap_mats) != len(vol_mats):
        raise RuntimeError("EWMA output length mismatch.")
    return swap_mats, vol_mats, Expiry, Tenor, t_vals

# ---------- script entry ----------
if __name__ == "__main__":
    # CHANGE THESE to two **different** files:
    rates_csv    = r"Chapter 5\swap_rate_computation\data\pca_simulated_rates_selected.csv"   # has 't' column
    init_vol_csv = r"Chapter 5\swap_rate_computation\data\initial.csv"                        # has 'expiry' column
    lam = 0.94

    print("Rates file:    ", os.path.abspath(rates_csv))
    print("Initial vol file:", os.path.abspath(init_vol_csv))

    swap_mats, vol_mats, Expiry, Tenor, t_vals = compute_swap_and_vol(rates_csv, init_vol_csv, lam)

    print(f"\n# timestamps: {len(t_vals)} | each matrix shape: {swap_mats[0].shape} "
          f"(rows=len(Expiry)={len(Expiry)}, cols=len(Tenor)={len(Tenor)})")

    # Print ALL timestamps with actual t
    for k, t in enumerate(t_vals):
        print(f"\n=== t = {t} ===")
        print("Swap Rate Matrix:")
        print(swap_mats[k])
        print("\nModel-Implied Volatility (EWMA):")
        print(vol_mats[k])
