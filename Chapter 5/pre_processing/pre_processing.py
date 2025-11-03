import pandas as pd

df = pd.read_csv(r"Chapter 5\pre_processing\simulated_rates_input_csv\vae_simulated_fwd_rates.csv")
df = df.drop(df.columns[[1, 2]], axis=1) # drop maturities 1M and 6M
df.to_csv("vae_simulated_rates_selected.csv", index=False)
