"""OOP pre-training for Autoencoder on synthetic Svensson curves (per-currency grid only)."""
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np
from calibration import svn_gen, save_ae, load_ae
from autoencoder import AutoencoderNN

# Tenor grids (years) per currency — currency is ONLY used to select this grid.
TENORS: Dict[str, list] = {
    "SGD": [3/12, 6/12, 1, 2, 5, 10, 15, 20, 30],
    "EUR": [3/12, 6/12] + list(range(1, 31)),
    "USD": [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30],
    "CNY": [3/12, 6/12, 1, 3, 5, 7, 10, 30],
    "GBP": [m/12 for m in range(1, 12)] + [1 + 0.5*k for k in range(0, 2*26 - 2 + 1)]
}

ReqRanges = Dict[str, Tuple[float, float]]

@dataclass
class AEPretrainer:
    """
    Pre-train an autoencoder on synthetic Svensson curves.
    - `currency` is ONLY used to choose the tenor grid (order/length must match your real data).
    - `ranges` are fully manual (beta1..beta4, lambd1..lambd2) and independent of currency.
    """
    currency: str
    ranges: ReqRanges                         # manual (low, high) for beta1..beta4, lambd1..lambd2
    act: str = "relu"                         # "relu", "sigmoid", or "tanh"
    n_samples: int = 40_000
    noise_std: float = 5e-4
    rng_seed_syn: int = 42
    rng_seed_split: int = 123
    epochs: int = 80
    batch_size: int = 256
    lr: float = 1e-3
    verbose: bool = True
    ckpt_path: Optional[str] = None           # defaulted to a generic name; no currency baked in

    # internals (set after init)
    taus: np.ndarray = field(init=False, repr=False)
    m: int = field(init=False, repr=False)
    model: Optional[AutoencoderNN] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        cur = self.currency.upper()
        if cur not in TENORS:
            raise ValueError(f"Unknown currency '{self.currency}'. Choose from {list(TENORS.keys())}")

        # Currency ONLY affects the tenor grid used for synthetic generation / inference.
        self.taus = np.array(TENORS[cur], dtype=np.float32)
        self.m = len(self.taus)

        # Validate manual ranges
        req = ["beta1","beta2","beta3","beta4","lambd1","lambd2"]
        missing = [k for k in req if k not in self.ranges]
        if missing:
            raise ValueError(f"RANGES missing keys: {missing}")
        for k, (lo, hi) in self.ranges.items():
            if hi < lo:
                raise ValueError(f"RANGES['{k}'] has hi < lo: {(lo, hi)}")
            if k.startswith("lambd") and (lo <= 0 or hi <= 0):
                raise ValueError(f"RANGES['{k}'] must be positive: {(lo, hi)}")

        # Generic checkpoint name (no currency baked in)
        if self.ckpt_path is None:
            self.ckpt_path = "ae_weights.npz"

        if self.verbose:
            print(f"Activation: {self.act} | Grid length m={self.m} (currency only for grid)")
            print("Tenors (years):", np.array2string(self.taus, precision=4, separator=", "))
            print("Sampling ranges (manual):")
            for k in req:
                lo, hi = self.ranges[k]
                print(f"  {k:7s}: [{lo:.6f}, {hi:.6f}]")
            print(f"Checkpoint path: {self.ckpt_path}")

    def fit(self) -> float:
        """Generate synthetic data, split 80:20, train AE, save, and return held-out test MSE."""
        X_syn = svn_gen(
            n_samples=self.n_samples,
            taus=self.taus,
            ranges=self.ranges,      # manual; NOT inferred from currency
            noise_std=self.noise_std,
            seed=self.rng_seed_syn,
        )

        # Train/test split
        rng = np.random.default_rng(self.rng_seed_split)
        n = len(X_syn)
        idx = rng.permutation(n)
        cut = int(0.8 * n)
        X_train, X_test = X_syn[idx[:cut]], X_syn[idx[cut:]]

        if self.verbose:
            print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

        # Train
        self.model = AutoencoderNN(param_in=self.m, activation=self.act)
        self.model.train(X_train, epochs=self.epochs, batch_size=self.batch_size,
                         lr=self.lr, shuffle=True, verbose=self.verbose)

        # Save
        save_ae(self.model, self.ckpt_path, include_optimizer=False)

        # Evaluate on held-out
        xhat_test = self.model.reconstruct(X_test)
        test_loss, _ = self.model.loss_fn(xhat_test, X_test)
        if self.verbose:
            print(f"Test MSE (held-out 20%): {test_loss:.8f}")
        return float(test_loss)

    def load(self) -> AutoencoderNN:
        """Load a saved model (activation + input dim must match the pretraining)."""
        ae = AutoencoderNN(param_in=self.m, activation=self.act)
        load_ae(ae, self.ckpt_path, include_optimizer=False)
        self.model = ae
        return ae

    def reconstruct_real(self, y: np.ndarray) -> np.ndarray:
        """
        Reconstruct (denoise/impute) a real curve vector y (length must equal m, same tenor order).
        """
        if self.model is None:
            self.load()
        y = np.asarray(y, dtype=np.float32)
        if len(y) != self.m:
            raise ValueError(f"Length mismatch: got {len(y)}, expected {self.m} for this grid.")
        return self.model.reconstruct(y[None, :])[0]

if __name__ == "__main__":
    # Currency ONLY determines which tenor grid to use.
    CURRENCY = "USD"     # "USD", "EUR", "SGD", "CNY", "GBP"
    ACT = "relu"         # "relu", "sigmoid", or "tanh"

    # Manual ranges (independent of currency)
    RANGES = {
        "beta1":  (3.3661, 4.9353),
        "beta2":  (-3.3400, -1.6373),
        "beta3":  (-4.5277, -0.2641),
        "beta4":  (-8.0692, -0.9213),
        "lambd1": (0.4154, 5.0450),
        "lambd2": (0.0850, 2.3469),
    }

    trainer = AEPretrainer(
        currency=CURRENCY,
        act=ACT,
        ranges=RANGES,
        n_samples=40_000,
        noise_std=5e-4,
        rng_seed_syn=42,
        rng_seed_split=123,
        epochs=80,
        batch_size=256,
        lr=1e-3,
        verbose=True,
        ckpt_path="ae_weights.npz",   # generic; not currency-specific
    )

    test_mse = trainer.fit()
    # Later:
    # real_curve = np.array([...], dtype=np.float32)  # len == trainer.m
    # y_hat = trainer.reconstruct_real(real_curve)
