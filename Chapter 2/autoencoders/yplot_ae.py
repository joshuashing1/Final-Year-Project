# fit_real_yields.py
import numpy as np
import pandas as pd
from offline_training_ae import AEPretrainer, TENORS

# ---- EDIT THESE ----
CSV_PATH = r"Chapter 2\Data\USTreasury_Yield_Final.csv"
CKPT_PATH = "ae_weights.npz"     # the weights you saved after pretraining
CURRENCY = "USD"                 # grid selector only
ACT = "relu"                     # must match the activation used when training
# RANGES aren't used during inference, but AEPretrainer validates them; reuse training ones or any valid positives for lambdas:
RANGES = {
    "beta1": (0.0, 1.0), "beta2": (-1.0, 1.0), "beta3": (-1.0, 1.0), "beta4": (-1.0, 1.0),
    "lambd1": (0.1, 5.0), "lambd2": (0.1, 5.0)
}
# Column names in the CSV matching the USD grid order [1m, 3m, 6m, 1y, 2y, 3y, 5y, 7y, 10y, 20y, 30y]
USD_COLS = ["1 Mo","3 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"]
DATE_COL = "Date"
OUT_CSV = "USTreasury_AE_reconstructed.csv"
# --------------------

def _ensure_decimal_units(arr: np.ndarray) -> np.ndarray:
    """If values look like percentages (>1.0), convert to decimals."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    med = np.median(np.abs(finite))
    return arr / 100.0 if med > 1.0 else arr

def fit_csv_with_ae():
    # Set up the pretrainer just to get the grid size and to load the model
    trainer = AEPretrainer(currency=CURRENCY, act=ACT, ranges=RANGES, verbose=True, ckpt_path=CKPT_PATH)
    trainer.load()  # loads the pre-trained weights into trainer.model

    # Load the dataset
    df = pd.read_csv(CSV_PATH)

    # Basic checks
    for col in [DATE_COL] + USD_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing column in CSV: {col}")

    # Build matrix X in the exact grid order
    X_raw = df[USD_COLS].to_numpy(dtype=np.float32)
    X = _ensure_decimal_units(X_raw)

    # Optional: drop rows with NaNs (or you can forward-fill instead)
    mask_ok = np.isfinite(X).all(axis=1)
    if not mask_ok.all():
        dropped = (~mask_ok).sum()
        print(f"Dropping {dropped} rows with NaNs.")
    dates = df.loc[mask_ok, DATE_COL].to_numpy()
    X = X[mask_ok]

    # Reconstruct each curve and compute per-row MSE
    Xhat = np.vstack([trainer.reconstruct_real(x) for x in X])
    mse = ((Xhat - X) ** 2).mean(axis=1)

    # Save results
    out = pd.DataFrame({DATE_COL: dates})
    for i, col in enumerate(USD_COLS):
        out[f"y_{col}"] = X[:, i]
        out[f"yhat_{col}"] = Xhat[:, i]
    out["mse"] = mse
    out.to_csv(OUT_CSV, index=False)

    print(f"Saved reconstructed curves to {OUT_CSV}")
    print(f"MSE summary → mean: {mse.mean():.8f} | median: {np.median(mse):.8f} | max: {mse.max():.8f}")

if __name__ == "__main__":
    fit_csv_with_ae()
