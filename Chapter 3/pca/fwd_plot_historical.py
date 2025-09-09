import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Chapter 3\data\GLC_fwd_curve_raw.csv")
df = df.set_index("time")

def parse_tenor(s: str) -> float:
    """Return tenor in years from labels like '1M','6M','1.0Y','10.0Y'."""
    s = s.strip().upper()
    if s.endswith("M"):
        return float(s[:-1]) / 12.0
    if s.endswith("Y"):
        return float(s[:-1])
    return float(s)

tenor_years = pd.Series({c: parse_tenor(c) for c in df.columns})
tenor_years = tenor_years.sort_values()
df = df.loc[:, tenor_years.index]

T = df.index.to_numpy()                      
Tau = tenor_years.to_numpy()                 
X, Y = np.meshgrid(Tau, T)                   
Z = df.to_numpy()                            

DPI, W_PX, H_PX = 100, 1573, 750
W_IN, H_IN = W_PX / DPI, H_PX / DPI

fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap="inferno", linewidth=0, antialiased=True, alpha=0.95)
ax.set_title("Historical forward rate", fontsize=37, fontweight="bold")
ax.set_xlabel(r"Tenor $T$ (years)", fontsize=20, labelpad=15)
ax.set_ylabel(r"Time $t$ (index)", fontsize=20, labelpad=13)
ax.set_zlabel(r"Forward rate $f(t,T)$", fontsize=20, labelpad=13)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.tick_params(axis="z", labelsize=16)
fig.colorbar(surf, shrink=0.6, pad=0.08, label=r"$f(t,T)$")
save_path = "historical_forward_surface.png"
fig.tight_layout()
fig.savefig(save_path, dpi=DPI)
plt.show()