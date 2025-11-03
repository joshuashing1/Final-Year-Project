import numpy as np, pandas as pd

def parse_tenor(s: str) -> float:
    s = str(s).strip().upper()
    return float(s[:-1]) / 12 if s.endswith("M") else float(s[:-1]) if s.endswith("Y") else float(s)

def build_discount_function(Tgrid: np.ndarray, fgrid: np.ndarray):
    Tgrid = np.asarray(Tgrid, float)
    fgrid = np.asarray(fgrid, float)
    o = np.argsort(Tgrid); Tgrid, fgrid = Tgrid[o], fgrid[o]

    def P(T):
        T = np.atleast_1d(T).astype(float)

        def one(z):
            if z <= 0:
                return 1.0
            knots = np.r_[0.0, Tgrid[Tgrid < z], z]
            vals  = np.interp(np.clip(knots, Tgrid[0], Tgrid[-1]), Tgrid, fgrid)
            return np.exp(-np.trapz(vals, knots))

        out = np.fromiter((one(z) for z in T), float)
        # 👇 return a Python float if caller passed a scalar
        return out[0] if out.size == 1 else out

    return P

def compute_swap_rate_matrices(csv_path: str) -> list:
    """
    Returns: list of np.ndarray, each with shape (19, 7).
             One matrix per timestamp (row) in the CSV.
    """
    df = pd.read_csv(csv_path)

    # Tenor axis from headers (everything except 't')
    ten_cols = [c for c in df.columns if c.lower() != "t"]
    Tgrid = np.array([parse_tenor(c) for c in ten_cols], float)
    order = np.argsort(Tgrid)
    Tgrid, ten_cols = Tgrid[order], [ten_cols[i] for i in order]

    # Expiry (19) and swap-tenor (7) grids (years)
    Exp = np.array([1/12, 3/12, 6/12, 9/12, 1, 2, 3, 4, 5, 6,
                    7, 8, 9, 10, 12, 15, 20, 25, 30], float)
    Ten = np.array([1, 2, 3, 5, 10, 20, 25], float)

    matrices = []

    for _, r in df.iterrows():
        fgrid = r[ten_cols].to_numpy(float)
        P = build_discount_function(Tgrid, fgrid)

        M = np.empty((Exp.size, Ten.size), float)
        for i, Ti in enumerate(Exp):
            for j, m in enumerate(Ten):
                Tj = Ti + m
                # Semiannual schedule (dt=0.5) with first stub if needed
                dt = 0.5
                first = np.ceil(Ti / dt) * dt
                if first > Tj:
                    pay_times = np.array([Tj])
                else:
                    pay_times = np.arange(first, Tj + 1e-12, dt)
                    if abs(pay_times[-1] - Tj) > 1e-12:
                        pay_times = np.r_[pay_times, Tj]
                accr = np.diff(np.r_[Ti, pay_times])

                A = float(np.sum(accr * P(pay_times)))
                M[i, j] = np.nan if A <= 0 else (float(P(Ti)) - float(P(Tj))) / A

        matrices.append(M)

    return matrices

if __name__ == "__main__":
    path = r"Chapter 5\swap_rate_computation\data\pca_simulated_rates_selected.csv"
    mats = compute_swap_rate_matrices(path)
    print(f"Got {len(mats)} matrices. Example shape: {mats[0].shape if mats else None}")
    # mats[k] is a 19x7 np.ndarray for timestamp k
    print(f"Total matrices: {len(mats)}")
    for i, M in enumerate(mats):
        print(f"\n=== Swap Rate Matrix for timestamp {i+1} ===")
        print(M)
