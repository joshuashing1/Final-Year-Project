"""
Variational Autoencoder (NumPy) for Yield Curve Smoothing — Higgins only
(no warmup, mean-field with unified K samples)

- Pretrain on synthetic Svensson curves
- Fine-tune on real yield data
- Plot/export smoothed curves (posterior-mean or MC-mean decoding)

Loss = Reconstruction + beta * KLD
Posterior q(z|x) is mean-field Gaussian:  N(mu, diag(exp(2*log_std)))
Reparameterization uses: z = mu + exp(log_std) * eps

Unified K:
    num_latent_samples = K  # used during training and (optionally) during inference if decode_mode="mc_mean".
"""

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

# ---------------- Path bootstrap ----------------
THIS = os.path.abspath(os.path.dirname(__file__))
PRJ_ROOT = os.path.abspath(os.path.join(THIS, ".."))
if PRJ_ROOT not in sys.path:
    sys.path.insert(0, PRJ_ROOT)

from parametric_models.yplot_historical import parse_maturities, LinearInterpolant
from parametric_models.svensson import SvenssonCurve

# ---------------- Activations ----------------
def relu(x):      return np.maximum(0.0, x)
def drelu(x):     return (x > 0.0).astype(x.dtype)
def tanh(x):      return np.tanh(x)
def dtanh(x):     y = np.tanh(x); return 1.0 - y * y
def sigmoid(x):   return 1.0 / (1.0 + np.exp(-x))
def dsigmoid(x):  s = sigmoid(x); return s * (1.0 - s)

_ACTS = {
    "relu":    (relu,    drelu),
    "tanh":    (tanh,    dtanh),
    "sigmoid": (sigmoid, dsigmoid),
    None:       (lambda x: x, lambda x: np.ones_like(x)),
}

# ---------------- Dense ----------------
class Dense:
    def __init__(self, in_dim: int, out_dim: int, activation: Optional[str], rng=None):
        self.in_dim  = int(in_dim)
        self.out_dim = int(out_dim)
        self.activation = activation
        self.f_act, self.f_dact = _ACTS[activation]
        if rng is None:
            rng = np.random.default_rng(0)
        self.rng = rng
        if activation == "relu":
            std = np.sqrt(2.0 / self.in_dim)
            self.W = self.rng.normal(0.0, std, size=(self.in_dim, self.out_dim)).astype(np.float32)
        else:
            limit = np.sqrt(6.0 / (self.in_dim + self.out_dim))
            self.W = self.rng.uniform(-limit, limit, size=(self.in_dim, self.out_dim)).astype(np.float32)
        self.b = np.zeros(self.out_dim, dtype=np.float32)
        # Adam buffers
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        # caches
        self.x = None; self.z = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z = x @ self.W + self.b
        return self.f_act(self.z)

    def backward(self, grad_out: np.ndarray):
        grad_z = grad_out * self.f_dact(self.z)
        dW = self.x.T @ grad_z
        db = grad_z.sum(axis=0)
        dx = grad_z @ self.W.T
        return dx, dW, db

# ---------------- Standardization ----------------
def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)

def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return ((X - mu) / sd).astype(np.float32)

def standardize_inverse(Z: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (Z * sd + mu).astype(np.float32)

# ---------------- Data prep ----------------
def detect_and_split_dates(df: pd.DataFrame):
    cols = df.columns.tolist()
    def is_tenor(c):
        s = str(c).strip().upper()
        if s.endswith("M") or s.endswith("Y"): return True
        try: float(s); return True
        except Exception: return False
    if len(cols) > 0 and not is_tenor(cols[0]):
        return df.iloc[:, 0].astype(str).tolist(), df.iloc[:, 1:].copy()
    return None, df.copy()

def build_dense_matrix(values_df: pd.DataFrame):
    tenor_labels = [str(c) for c in values_df.columns]
    maturities_years = parse_maturities(tenor_labels)
    X_list = []
    for _, row in values_df.iterrows():
        y = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() >= 2:
            interp = LinearInterpolant(maturities_years[mask], y[mask])
            y_filled = interp(maturities_years)
        elif mask.sum() == 1:
            y_filled = np.full_like(maturities_years, y[mask][0], dtype=float)
        else:
            y_filled = np.zeros_like(maturities_years, dtype=float)
        X_list.append(y_filled)
    X = np.vstack(X_list).astype(np.float32)
    return X, maturities_years, tenor_labels

# ---------------- Plotting ----------------
def yield_curves_plot(maturities_years, fitted_curves, title, save_path, subtitle=None):
    x_min = float(np.nanmin(maturities_years))
    x_max = float(np.nanmax(maturities_years))
    x_grid = np.linspace(x_min, x_max, 300)
    DPI = 100; W_IN = 1573 / DPI; H_IN = 750 / DPI
    fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI)
    for curve in fitted_curves:
        ax.plot(x_grid, curve(x_grid), linewidth=0.8)
    TICK_FS = 27
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_FS)
    ax.xaxis.get_offset_text().set_size(TICK_FS)
    ax.yaxis.get_offset_text().set_size(TICK_FS)
    ax.set_xlabel("Maturity (Years)", fontsize=32)
    ax.set_ylabel("Interest Rate (%)", fontsize=32)
    title_text = title if not subtitle else f"{title} — {subtitle}"
    ax.set_title(title_text, fontsize=37, fontweight="bold", pad=12)
    ax.set_ylim(-2, 10)
    ax.set_xlim(left=0, right=x_max)
    fig.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()
    print(f"Saved figure to {save_path}")

# ---------------- Synthetic Svensson ----------------
def generate_synthetic_svensson(n_samples: int, maturities_years: np.ndarray, ranges: dict, noise_std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = len(maturities_years)
    X = np.empty((n_samples, m), dtype=np.float32)
    for i in range(n_samples):
        beta1  = rng.uniform(*ranges["beta1"])
        beta2  = rng.uniform(*ranges["beta2"])
        beta3  = rng.uniform(*ranges["beta3"])
        beta4  = rng.uniform(*ranges["beta4"])
        lambd1 = rng.uniform(*ranges["lambd1"])
        lambd2 = rng.uniform(*ranges["lambd2"])
        curve = SvenssonCurve(beta1=beta1, beta2=beta2, beta3=beta3, beta4=beta4, lambd1=lambd1, lambd2=lambd2)
        y = curve(maturities_years)
        if noise_std > 0: y = y + rng.normal(0.0, noise_std, size=y.shape)
        X[i] = y.astype(np.float32)
    return X

# ---------------- NumPy Beta-VAE (Higgins only, mean-field with unified K) ----------------
class BetaVAE_MLP_NumPy:
    """
    Encoder: in_dim -> 30 -> 30 -> 13 -> (mu, log_std)
    Decoder: latent_dim -> 30 -> 30 -> in_dim
    Posterior: q(z|x) = N(mu, diag(exp(2*log_std)))
    Reparameterization: z = mu + exp(log_std) * eps
    Loss: Reconstruction + beta * KLD
    """
    num_iter = 0  # for logging

    def __init__(self,
                 in_dim: int,
                 latent_dim: int = 13,
                 hidden_dims: Optional[List[int]] = None,
                 beta: float = 2.0,
                 activation: str = 'relu',
                 rng=None):
        if rng is None: rng = np.random.default_rng(0)
        self.rng = rng
        if hidden_dims is None: hidden_dims = [30, 30, 13]
        if activation not in _ACTS: raise ValueError(f"activation must be one of {list(_ACTS.keys())}")

        self.in_dim = int(in_dim)
        self.latent_dim = int(latent_dim)
        self.beta = float(beta)

        # encoder
        dims = [self.in_dim] + hidden_dims
        self.enc_layers: List[Dense] = [Dense(dims[i], dims[i+1], activation=activation, rng=self.rng)
                                        for i in range(len(dims)-1)]
        top_dim = hidden_dims[-1]
        self.fc_mu     = Dense(top_dim, self.latent_dim, activation=None, rng=self.rng)
        self.fc_logstd = Dense(top_dim, self.latent_dim, activation=None, rng=self.rng)

        # decoder
        self.dec1 = Dense(self.latent_dim, 30, activation=activation, rng=self.rng)
        self.dec2 = Dense(30, 30, activation=activation, rng=self.rng)
        self.dec_out = Dense(30, self.in_dim, activation=None, rng=self.rng)

        # caches
        self._std = None; self._eps = None; self._z = None
        self._adam_t = 0

    # ---- forward helpers ----
    def _encode_hidden(self, x: np.ndarray) -> np.ndarray:
        h = x
        for layer in self.enc_layers: h = layer.forward(h)
        return h

    def encode(self, x: np.ndarray):
        h = self._encode_hidden(x)
        mu = self.fc_mu.forward(h)
        log_std = self.fc_logstd.forward(h)
        self._std = np.exp(log_std).astype(np.float32)
        self._eps = self.rng.normal(size=self._std.shape).astype(np.float32)
        return mu, log_std

    def reparameterize(self, mu: np.ndarray, log_std: np.ndarray) -> np.ndarray:
        std = np.exp(log_std).astype(np.float32)
        eps = self.rng.normal(size=std.shape).astype(np.float32)
        z = mu + std * eps
        self._std, self._eps, self._z = std, eps, z
        return z

    def decode(self, z: np.ndarray) -> np.ndarray:
        h = self.dec1.forward(z)
        h = self.dec2.forward(h)
        return self.dec_out.forward(h)

    def forward(self, x: np.ndarray):
        # single-sample forward; training uses multi-sample path in train_epoch
        x = x.astype(np.float32, copy=False)
        self._z = None
        mu, log_std = self.encode(x)
        z = self.reparameterize(mu, log_std)
        recons = self.decode(z)
        return [recons, x, mu, log_std]

    # ---- deterministic/MC decode (for export/plots) ----
    def reconstruct_mean(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32, copy=False)
        h = X
        for layer in self.enc_layers: h = layer.forward(h)
        mu = self.fc_mu.forward(h)
        return self.decode(mu).astype(np.float32)

    def reconstruct_mc_mean(self, X: np.ndarray, num_latent_samples: int) -> np.ndarray:
        """ Monte-Carlo mean of p(x|z) with z ~ q(z|x); uses K samples per row. """
        K = max(1, int(num_latent_samples))
        X = X.astype(np.float32, copy=False)
        h = X
        for layer in self.enc_layers: h = layer.forward(h)
        mu = self.fc_mu.forward(h)
        log_std = self.fc_logstd.forward(h)
        std = np.exp(log_std).astype(np.float32)

        xhat_sum = np.zeros((X.shape[0], self.in_dim), dtype=np.float32)
        for _ in range(K):
            eps = self.rng.normal(size=std.shape).astype(np.float32)
            z = mu + std * eps
            xhat_sum += self.decode(z)
        return (xhat_sum / float(K)).astype(np.float32)

    # ---- losses ----
    @staticmethod
    def _mse_mean(a: np.ndarray, b: np.ndarray) -> float:
        d = a - b; return float((d*d).mean())

    @staticmethod
    def _kld_mean(mu: np.ndarray, log_std: np.ndarray) -> float:
        # KLD(q||p) for diagonal Gaussian with log_std parameterization
        s2 = np.exp(2.0 * log_std)
        kld_i = -0.5 * (1.0 + 2.0 * log_std - mu*mu - s2).sum(axis=1)
        return float(kld_i.mean())

    def loss_function(self, *args, **kwargs):
        # single-sample API symmetry
        self.__class__.num_iter += 1
        recons, x, mu, log_std = args
        kld_weight = float(kwargs.get('M_N', 1.0))
        rec = self._mse_mean(recons, x)
        kld = self._kld_mean(mu, log_std)
        total = rec + (self.beta) * kld_weight * kld
        return {'loss': float(total), 'Reconstruction_Loss': float(rec), 'KLD': float(kld)}

    # ---- optimizer ----
    def parameters(self) -> List[Dense]:
        return [*self.enc_layers, self.fc_mu, self.fc_logstd, self.dec1, self.dec2, self.dec_out]

    def step_adam(self, grads: List[Tuple[np.ndarray, np.ndarray]], lr: float, t: int,
                  beta1=0.9, beta2=0.999, eps=1e-8, weight_decay: float = 0.0):
        for layer, (dW, db) in zip(self.parameters(), grads):
            if weight_decay != 0.0:
                dW = dW + weight_decay * layer.W
                db = db + weight_decay * layer.b
            layer.mW = beta1 * layer.mW + (1.0 - beta1) * dW
            layer.vW = beta2 * layer.vW + (1.0 - beta2) * (dW * dW)
            layer.mb = beta1 * layer.mb + (1.0 - beta1) * db
            layer.vb = beta2 * layer.vb + (1.0 - beta2) * (db * db)
            mW_hat = layer.mW / (1.0 - beta1**t)
            vW_hat = layer.vW / (1.0 - beta2**t)
            mb_hat = layer.mb / (1.0 - beta1**t)
            vb_hat = layer.vb / (1.0 - beta2**t)
            layer.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            layer.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def train_epoch(self, X: np.ndarray, batch_size: int, lr: float,
                    kld_weight: float = 1.0, shuffle: bool = True, weight_decay: float = 0.0,
                    num_latent_samples: int = 1):
        """
        Train for one epoch using K Monte-Carlo latent samples per datapoint (num_latent_samples >= 1).
        Reconstruction loss is averaged over K; KL is analytic and computed once.
        """
        K = max(1, int(num_latent_samples))

        X = X.astype(np.float32, copy=False)
        n = len(X)
        idx = np.arange(n)
        if shuffle: np.random.shuffle(idx)
        Xs = X[idx]
        sum_total = 0.0; sum_rec = 0.0; sum_kld = 0.0

        for i in range(0, n, batch_size):
            xb = Xs[i:i+batch_size]; B = len(xb)

            # ----- encode once -----
            h = xb
            for layer in self.enc_layers: h = layer.forward(h)
            mu = self.fc_mu.forward(h)
            log_std = self.fc_logstd.forward(h)
            std = np.exp(log_std).astype(np.float32)

            # accumulators
            rec_acc = 0.0
            # decoder grads accumulated over K samples
            dW_out_acc = np.zeros_like(self.dec_out.W); db_out_acc = np.zeros_like(self.dec_out.b)
            dW_d2_acc  = np.zeros_like(self.dec2.W);    db_d2_acc  = np.zeros_like(self.dec2.b)
            dW_d1_acc  = np.zeros_like(self.dec1.W);    db_d1_acc  = np.zeros_like(self.dec1.b)
            # grads flowing to mu and log_std from reconstruction part (sum over K)
            g_mu_sum = np.zeros_like(mu)
            g_ls_sum = np.zeros_like(log_std)

            # ----- sample K times and average recon loss -----
            for _ in range(K):
                eps = self.rng.normal(size=std.shape).astype(np.float32)
                z = mu + std * eps

                # forward decode (sets layer caches)
                xhat = self.decode(z)

                # recon loss contribution for this sample
                rec_k = self._mse_mean(xhat, xb)
                rec_acc += rec_k

                # dL/dxhat = 2/(B*K) * (xhat - x)
                dL_dxhat = (2.0 / (B * K)) * (xhat - xb)

                # backprop through decoder (accumulate)
                g, dW_out, db_out = self.dec_out.backward(dL_dxhat);  dW_out_acc += dW_out; db_out_acc += db_out
                g, dW_d2,  db_d2  = self.dec2.backward(g);            dW_d2_acc  += dW_d2;  db_d2_acc  += db_d2
                g, dW_d1,  db_d1  = self.dec1.backward(g);            dW_d1_acc  += dW_d1;  db_d1_acc  += db_d1

                # gradient wrt mu and log_std via z = mu + std * eps, std = exp(log_std)
                g_mu_sum += g
                g_ls_sum += g * (std * eps)

            # average reconstruction loss over K
            rec = rec_acc / float(K)

            # KL (analytic; computed once)
            s2 = np.exp(2.0 * log_std)
            kld_i = -0.5 * (1.0 + 2.0 * log_std - mu*mu - s2).sum(axis=1)
            kld = float(kld_i.mean())

            # scale for KL in gradients
            self.__class__.num_iter += 1
            scale = (self.beta) * kld_weight

            # add KL contributions to mu/log_std grads (divide by B for mean)
            dK_dmu =  (mu / B).astype(np.float32)
            dK_dls = ((s2 - 1.0) / B).astype(np.float32)
            g_mu_total = g_mu_sum + scale * dK_dmu
            g_ls_total = g_ls_sum + scale * dK_dls

            # backprop into μ and log_std heads
            gh_mu, dW_mu, db_mu = self.fc_mu.backward(g_mu_total)
            gh_ls, dW_ls, db_ls = self.fc_logstd.backward(g_ls_total)
            gh = gh_mu + gh_ls

            # backprop through encoder (once; grads already summed over K)
            enc_back_grads = []
            for layer in reversed(self.enc_layers):
                gh, dW_e, db_e = layer.backward(gh)
                enc_back_grads.append((dW_e, db_e))
            enc_back_grads.reverse()

            # collect grads in the same order as parameters()
            grads = enc_back_grads + [(dW_mu, db_mu), (dW_ls, db_ls),
                                      (dW_d1_acc, db_d1_acc), (dW_d2_acc, db_d2_acc), (dW_out_acc, db_out_acc)]

            # Adam step
            self._adam_t += 1
            self.step_adam(grads, lr, self._adam_t, weight_decay=weight_decay)

            total = rec + (self.beta) * kld_weight * kld
            sum_total += total * B; sum_rec += rec * B; sum_kld += kld * B

        denom = float(n)
        return sum_total/denom, sum_rec/denom, sum_kld/denom

    # convenience
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32, copy=False)
        recons, *_ = self.forward(X)
        return recons.astype(np.float32)

    def encode_mean(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32, copy=False)
        h = X
        for layer in self.enc_layers: h = layer.forward(h)
        mu = self.fc_mu.forward(h)
        return mu.astype(np.float32)

# ---------------- Pretrain ----------------
def pretrain_on_synthetic_vae(model: BetaVAE_MLP_NumPy, maturities_years: np.ndarray, syn_cfg: dict, verbose: bool = True):
    X_syn = generate_synthetic_svensson(
        n_samples=syn_cfg["n_samples"], maturities_years=maturities_years,
        ranges=syn_cfg["ranges"], noise_std=float(syn_cfg.get("noise_std", 0.0)), seed=int(syn_cfg.get("seed", 0))
    )
    mu_syn, sd_syn = standardize_fit(X_syn)
    Xz_syn = standardize_apply(X_syn, mu_syn, sd_syn)
    rng = np.random.default_rng(int(syn_cfg.get("seed", 0)))
    if syn_cfg.get("noise_std_train", 0.0) and syn_cfg["noise_std_train"] > 0:
        Xz_train = Xz_syn + rng.normal(0.0, syn_cfg["noise_std_train"], size=Xz_syn.shape).astype(np.float32)
    else:
        Xz_train = Xz_syn

    K = int(syn_cfg.get("num_latent_samples", 1))
    for ep in range(int(syn_cfg["epochs"])):
        total, rec, kld = model.train_epoch(
            Xz_train,
            batch_size=int(syn_cfg["batch_size"]),
            lr=float(syn_cfg["lr"]),
            num_latent_samples=K
        )
        if verbose and (ep+1) % 20 == 0:
            print(f"[pretrain] epoch {ep+1:03d} | total={total:.6f} | recon={rec:.6f} | kld={kld:.6f}")
    if verbose: print("[pretrain] Finished synthetic VAE pretraining.")

# ---------------- End-to-end ----------------
def process_yield_csv_vae(csv_path: str, title: str, epochs: int, batch_size: int, lr: float, activation: str,
                          noise_std: float, latent_dim: int = 13, save_latent=True, pretrain: dict | None = None,
                          weight_decay: float = 0.0, kld_weight: float = 1.0,
                          beta: float = 1.0, decode_mode: str = "mean",
                          num_latent_samples: int = 1):
    """
    num_latent_samples (K) is used for:
      - training Monte-Carlo gradient estimates, and
      - MC decoding if decode_mode == "mc_mean".
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    rng = np.random.default_rng(0)

    # 1) Load CSV
    df_raw = pd.read_csv(csv_path, header=0)
    dates, values_df = detect_and_split_dates(df_raw)

    # 2) Dense matrix
    X, maturities_years, tenor_labels = build_dense_matrix(values_df)
    n_obs, n_tenors = X.shape
    print(f"[{title}] Loaded {n_obs} rows with {n_tenors} tenors.")

    # 3) VAE (Higgins only; mean-field)
    vae = BetaVAE_MLP_NumPy(
        in_dim=n_tenors, latent_dim=latent_dim, activation=activation, rng=rng, beta=beta
    )

    # 4) Pretrain (optional)
    if pretrain is not None:
        pretrain_on_synthetic_vae(vae, maturities_years, pretrain, verbose=True)

    # 5) Fine-tune (standardize on real stats)
    mu_real, sd_real = standardize_fit(X)
    Xz_real = standardize_apply(X, mu_real, sd_real)
    X_train = Xz_real + rng.normal(0.0, noise_std, size=Xz_real.shape).astype(np.float32) if noise_std > 0 else Xz_real

    for ep in range(1, epochs + 1):
        total, rec, kld = vae.train_epoch(
            X_train, batch_size=batch_size, lr=lr,
            kld_weight=kld_weight, weight_decay=weight_decay,
            num_latent_samples=num_latent_samples
        )
        if ep % 10 == 0:
            print(f"[{title}] epoch {ep:03d} | total={total:.6f} | recon={rec:.6f} | kld={kld:.6f}")

    # 6) Reconstruct & invert standardization
    if decode_mode == "mean":
        Zhat = vae.reconstruct_mean(Xz_real)
        subtitle = "VAE (posterior mean decode)"
        tag = "vae_mean"
    elif decode_mode == "mc_mean":
        Zhat = vae.reconstruct_mc_mean(Xz_real, num_latent_samples=num_latent_samples)
        subtitle = f"VAE (MC mean, K={num_latent_samples})"
        tag = f"vae_mcmeanK{num_latent_samples}"
    else:  # "sampled"
        Zhat = vae.reconstruct(Xz_real)
        subtitle = "VAE (single sampled decode)"
        tag = "vae_sampled"

    X_smooth = standardize_inverse(Zhat.astype(np.float32), mu_real, sd_real)

    # 7) RMSE on observed quotes only
    rmses = []
    for i, row in values_df.iterrows():
        y = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() == 0:
            rmses.append(np.nan); continue
        y_obs = y[mask]; yhat_obs = X_smooth[i, mask]
        rmse = float(np.sqrt(np.mean((yhat_obs - y_obs) ** 2)))
        rmses.append(rmse)
    avg_rmse = float(np.nanmean(rmses))
    print(f"[{title}] VAE fit average RMSE on observed quotes: {avg_rmse:.6f}")

    # 8) Save smoothed CSV
    out_df = pd.DataFrame(X_smooth, columns=tenor_labels)
    if dates is not None: out_df.insert(0, "Date", dates)
    out_csv = f"{title}_yield_reconstructed_{tag}.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[{title}] Saved smoothed CSV to {out_csv}")

    # 9) Save latent means (posterior μ)
    if save_latent:
        lat_mu = vae.encode_mean(Xz_real).astype(np.float32)
        lat_df = pd.DataFrame(lat_mu, columns=[f"z{i+1}" for i in range(lat_mu.shape[1])])
        if dates is not None: lat_df.insert(0, "Date", dates)
        lat_csv = f"{title}_latent_mean_vae.csv"
        lat_df.to_csv(lat_csv, index=False)
        print(f"[{title}] Saved latent mean factors to {lat_csv}")

    # 10) Plot smoothed curves
    fitted_curves = [LinearInterpolant(maturities_years, row) for row in X_smooth]
    fig_path = f"{title}_{tag}_curve.png"
    yield_curves_plot(maturities_years, fitted_curves, title=f"{title}", save_path=fig_path, subtitle=subtitle)

    return avg_rmse

# ---------------- Main ----------------
def main():
    sv_ranges = {
        "beta1":  (3.7304, 4.4242),
        "beta2":  (-3.6421, 1.4914),
        "beta3":  (-0.8597, 2.9231),
        "beta4":  (-2.6874, 2.7582),
        "lambd1": (0.8896, 5.0500),
        "lambd2": (1.1007, 5.0500)
    }

    datasets = [{"csv_path": r"Chapter 2\Data\GLC_Yield_Final.csv", "title": "GBP"}]

    # Unified K for pretraining and fine-tuning and (optionally) inference MC mean
    K = 10

    pretrain_cfg = {
        "n_samples": 20000,
        "ranges": sv_ranges,
        "epochs": 300,
        "batch_size": 256,
        "lr": 1e-3,
        "noise_std": 0.00,
        "noise_std_train": 0.005,
        "seed": 0,
        "num_latent_samples": K,   # unified K during pretraining
    }

    for item in datasets:
        process_yield_csv_vae(
            csv_path=item["csv_path"],
            title=item["title"],
            epochs=120,
            batch_size=64,
            lr=1e-3,
            activation="relu",
            noise_std=0.00,
            latent_dim=13,
            save_latent=True,
            pretrain=pretrain_cfg,
            weight_decay=0.0,
            kld_weight=1.0,
            beta=1.0,                 # Higgins beta (no warmup), log_std parametrization
            decode_mode="mc_mean",    # "mean", "mc_mean", or "sampled"
            num_latent_samples=K      # unified K during fine-tuning and MC decoding
        )

if __name__ == "__main__":
    main()
