import py_vollib
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import vega


def implied_vol(S0, K, T, r, market_price, flag="c", tolerance=1e-5, max_iter=100):
    sigma = 0.20  # initial guess

    for _ in range(max_iter):
        price = bs(flag, S0, K, T, r, sigma)
        diff = price - market_price

        if abs(diff) < tolerance:
            return sigma

        v = vega(flag, S0, K, T, r, sigma)
        # py_vollib vega is per 1% change in vol → derivative w.r.t. sigma:
        v_sigma = v * 100.0

        if v_sigma == 0:
            raise RuntimeError("Vega is zero; Newton step would divide by zero.")

        sigma -= diff / v_sigma

    # If not converged, return last iterate (or raise an error if you prefer)
    return sigma


S0, K, T, r, market_price = 25755, 26000, 25/365, 0.055, 323.05
print(implied_vol(S0, K, T, r, market_price))
