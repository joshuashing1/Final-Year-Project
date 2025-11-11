import pandas as pd
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
            return sigma
        v_core = vega_black(flag, F0, K, T, 0.0, sigma)      
        v_sigma = annuity * v_core * 100.0                   
        if v_sigma == 0:
            break
        sigma -= diff / v_sigma
    return sigma

T = 3.0 / 12.0 # option expiry in years

# csv_path = r"Chapter 5\implied_volatility_computation\data\pca_1Y3M_implied_volatility.csv"

# df = pd.read_csv(csv_path)
# df.columns = [c.strip() for c in df.columns]  

# df["implied_vol"] = df.apply(
#     lambda row: implied_vol(
#         F0=row["S_t"],
#         K=row["strike"],
#         T=T,
#         annuity=row["annuity"],
#         market_price=row["mkt price"],  
#         flag="c"
#     ),
#     axis=1
# )

# print(df)

pca_path   = r"Chapter 5\implied_volatility_computation\data\pca_1Y3M_implied_volatility.csv"
libor_path = r"Chapter 5\bloomberg_data\LIBOR_swaption_1Y3M.csv"

# --- PCA-based model implied vol ---
df_model = pd.read_csv(pca_path)
df_model.columns = [c.strip() for c in df_model.columns]

df_model["implied_vol"] = df_model.apply(
    lambda row: implied_vol(
        F0=row["S_t"],
        K=row["strike"],
        T=T,
        annuity=row["annuity"],
        market_price=row["mkt price"],
        flag="c"
    ),
    axis=1
)

# --- Market implied vol ---
df_mkt = pd.read_csv(libor_path)
df_mkt.columns = [c.strip() for c in df_mkt.columns]
df_mkt = df_mkt.rename(columns={"mkt implied vol": "mkt_implied_vol"})

# --- Merge on time t ---
df = df_model.merge(df_mkt[["t", "mkt_implied_vol"]], on="t", how="inner")

# (optional) if you want to drop obviously bad vols like the huge negative:
# df = df[(df["implied_vol"] > 0) & (df["implied_vol"] < 1)]

# --- Plot model vs market implied vol ---
DPI = 100
W_IN, H_IN = 1573 / DPI, 750 / DPI

fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)
ax.plot(df["t"], df["implied_vol"], marker="o", label="Model implied vol")
ax.plot(df["t"], df["mkt_implied_vol"], marker="x", linestyle="--", label="Market implied vol")

ax.set_xlabel("Time t")
ax.set_ylabel("Implied volatility")
ax.set_title("1Y x 3M Swaption: Model vs Market Implied Volatility")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()