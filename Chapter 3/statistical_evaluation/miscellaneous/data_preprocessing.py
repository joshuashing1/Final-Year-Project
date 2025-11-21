"""
This is a data preprocessing Python script that selects the specific tenors of the 
simulated rates for easier statistical inference.
"""

import pandas as pd
from pathlib import Path

INPUT  = Path(r"Chapter 3\statistical_evaluation\simulated_fwd_rates\GLC_fwd_curve_raw.csv")
OUTPUT = INPUT.with_name("GLC_fwd_curve_selected.csv")

keep_cols = ["t","1M","6M","1.0Y","2.0Y","3.0Y","5.0Y","10.0Y","20.0Y","25.0Y"]

df = pd.read_csv(INPUT)

cols = [c for c in keep_cols if c in df.columns]
df = df[cols].copy()

num_cols = [c for c in df.columns if c != "t"]
df[num_cols] = df[num_cols] / 100.0

df.to_csv(OUTPUT, index=False)
print(f"Saved selected & scaled dataset to: {OUTPUT}")
