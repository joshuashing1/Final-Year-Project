import pandas as pd
from py_vollib.black import black as black76
from py_vollib.black.greeks.analytical import vega as vega_black


def implied_vol_black76_swaption(F0, K, T, annuity, market_price,
                                 flag="c", tol=1e-5, max_iter=100):
    sigma = 0.20  # initial guess
    for _ in range(max_iter):
        core_price = black76(flag, F0, K, T, 0.0, sigma) # r=0, discount via annuity
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
csv_path = r"Chapter 5\implied_vol_computation\data\pca_1Y3M_implied_volatility.csv"

df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]  

df["implied_vol"] = df.apply(
    lambda row: implied_vol_black76_swaption(
        F0=row["S_t"],
        K=row["strike"],
        T=T,
        annuity=row["annuity"],
        market_price=row["mkt price"],  
        flag="c"
    ),
    axis=1
)

print(df)
