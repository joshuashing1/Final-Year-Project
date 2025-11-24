import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def parse_tenor(s: str) -> float:
    s = s.strip().upper()
    if s.endswith("M"): return float(s[:-1]) / 12.0
    if s.endswith("Y"): return float(s[:-1])
    return float(s)

def export_simul_fwd(sim_matrix: np.ndarray, taus: np.ndarray, labels: list[str], dt: float,
                     save_path: Path = Path("simulated_fwd_rates.csv")) -> Path:
    """
    Build a CSV with rows=time and columns=selected tenor labels.
    Time 't' is a step index: 1, 2, ..., T  (decimals; no ×100).
    """
    T, N = sim_matrix.shape
    want = np.array([parse_tenor(x) for x in labels], float)
    idx  = [int(np.where(np.isclose(taus, w))[0][0]) for w in want]  # raises if not found
    sim_sel = sim_matrix[:, idx]                 # (T, len(labels))
    tgrid   = np.arange(1, T + 1, dtype=int)

    df = pd.DataFrame(sim_sel, columns=labels)
    df.insert(0, "t", tgrid)

    save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved simulated f(t,τ) CSV to {save_path}")
    return save_path