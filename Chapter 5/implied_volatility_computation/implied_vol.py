import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from py_vollib.black import black as black76
from py_vollib.black.greeks.analytical import vega as vega_black

def implied_vol(F0, K, T, annuity, market_price, flag="c", tol=1e-5, max_iter=100):
    sigma = 0.20  # initial guess
    for _ in range(max_iter):
        core_price = black76(flag, F0, K, T, 0.0, sigma) # r=0 implies discounting via annuity
        swaption_price = annuity * core_price                         
        diff = swaption_price - market_price
        if abs(diff) < tol:
            return sigma if sigma >= 0 else np.nan
        v_core = vega_black(flag, F0, K, T, 0.0, sigma)      
        v_sigma = annuity * v_core * 100.0                   
        if v_sigma == 0:
            break
        sigma -= diff / v_sigma
    return sigma if sigma >= 0 else np.nan

T = 3.0 / 12.0 # option expiry

pca_path   = r"Chapter 5\implied_volatility_computation\data\pca_1Y3M_implied_volatility.csv"
libor_path = r"Chapter 5\bloomberg_data\LIBOR_swaption_1Y3M.csv"

df_model = pd.read_csv(pca_path)
df_model.columns = [c.strip() for c in df_model.columns]

df_model["implied_vol"] = df_model.apply(
    lambda row: implied_vol(
        F0=row["S_t"],
        K=row["strike"],
        T=T,
        annuity=row["annuity"],
        market_price=row["mkt_price"],
        flag="c"
    ),
    axis=1
)

df_mkt = pd.read_csv(libor_path)
df_mkt.columns = [c.strip() for c in df_mkt.columns]       

df = df_model.merge(df_mkt[["t", "mkt_implied_vol"]], on="t", how="inner")
df = df.sort_values("t")

DPI = 100
W_IN, H_IN = 1573 / DPI, 750 / DPI

fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)
ax.plot(df["t"], df["implied_vol"], marker="o", label="Model implied vol")
ax.plot(df["t"], df["mkt_implied_vol"], marker="x", linestyle="--", label="Market implied vol")
ax.set_xlabel("Time t")
ax.set_ylabel("Implied volatility")
ax.set_title("1Y × 3M Swaption: Model vs Market Implied Volatility")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()