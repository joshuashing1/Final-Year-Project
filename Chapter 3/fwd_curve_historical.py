# 3D evolution of forward curve f(t, τ)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D

# --- load & prep -------------------------------------------------------------
df = pd.read_csv("Chapter 3\GLC_fwd_curve_raw.csv")        # expects a 'time' column, rest are tenors
df = df.set_index("time")

def parse_tenor(s: str) -> float:
    """Return tenor in years from labels like '1M','6M','1.0Y','10.0Y'."""
    s = s.strip().upper()
    if s.endswith("M"):
        return float(s[:-1]) / 12.0
    if s.endswith("Y"):
        return float(s[:-1])
    # fallback: try raw numeric
    return float(s)

# Map tenor labels -> year-floats and sort columns by tenor ascending
tenor_years = pd.Series({c: parse_tenor(c) for c in df.columns})
tenor_years = tenor_years.sort_values()
df = df.loc[:, tenor_years.index]

T = df.index.to_numpy()                      # time axis
Tau = tenor_years.to_numpy()                 # tenor (years)
X, Y = np.meshgrid(Tau, T)                   # shapes: [n_time, n_tenor]
Z = df.to_numpy()                            # same shape as X/Y

# --- 3D surface plot ---------------------------------------------------------
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(X, Y, Z, cmap="inferno", linewidth=0, antialiased=True, alpha=0.95)
# Optional: wireframe overlay for structure (comment out if not desired)
# ax.plot_wireframe(X, Y, Z, rstride=6, cstride=6, linewidth=0.5)

ax.set_xlabel(r"Tenor $\tau$ (years)")
ax.set_ylabel(r"Time $t$ (index)")
ax.set_zlabel(r"Forward rate $f(t,\tau)$ [%]")
fig.colorbar(surf, shrink=0.6, pad=0.08, label=r"$f(t,\tau)$ [%]")
plt.tight_layout()
plt.show()
