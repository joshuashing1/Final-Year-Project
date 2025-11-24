"""
This Python script outlines the VAE network architecture using some fundamentals of NN 
defined in nn_fn.py.
"""

import numpy as np
from typing import List

from machine_functions.nn_fn import Dense


class VariationalNN:
    """
    VAE network architecture consisting of nodes: 30 -> 30 -> 13 -> 30 -> 30
    Reparameterization eqn: z = μ + exp(log σ) * eps
    Latent dimension: 2
    Monte-carlo samples: 2 to 5 
    """

    def __init__(self, param_in: int, activation: str, latent_dim: int, rng=None):
        self.param_in = int(param_in)
        self.activation = activation
        self.latent_dim = int(latent_dim)
        self.rng = np.random.default_rng(0) if rng is None else rng

        self.encoder1 = Dense(self.param_in, 30, activation=self.activation, rng=self.rng)
        self.encoder2 = Dense(30, 30, activation=self.activation, rng=self.rng)
        self.encoder3 = Dense(30, 13, activation=self.activation, rng=self.rng)

        self.mu_head     = Dense(13, self.latent_dim, activation=None, rng=self.rng)
        self.logstd_head = Dense(13, self.latent_dim, activation=None, rng=self.rng)

        self.decoder1 = Dense(self.latent_dim, 30, activation=self.activation, rng=self.rng)
        self.decoder2 = Dense(30, 30, activation=self.activation, rng=self.rng)
        self.out      = Dense(30, self.param_in, activation=None, rng=self.rng)

    def _encode_hidden(self, x: np.ndarray) -> np.ndarray:
        h = self.encoder1.forward(x) # forward pass
        h = self.encoder2.forward(h)
        h = self.encoder3.forward(h)
        return h

    def encode(self, x: np.ndarray):
        """Return parameters μ, log σ of tractable distribution"""
        h = self._encode_hidden(x.astype(np.float32, copy=False))
        mu = self.mu_head.forward(h)
        log_std = self.logstd_head.forward(h)
        return mu, log_std

    def decode(self, z: np.ndarray) -> np.ndarray:
        h = self.decoder1.forward(z)
        h = self.decoder2.forward(h)
        return self.out.forward(h)

    def forward(self, x: np.ndarray):
        """Generate latent variables for Monte-Carlo sampling"""
        mu, log_std = self.encode(x)
        std = np.exp(log_std).astype(np.float32)
        eps = self.rng.normal(size=std.shape).astype(np.float32)
        z = (mu + std * eps).astype(np.float32)
        return self.decode(z)

    def loss_fn(self, pred: np.ndarray, y: np.ndarray):
        diff = pred - y
        return float((diff * diff).mean()), (2.0 / len(y)) * diff

    def parameters(self) -> List[Dense]:
        return [
            self.encoder1, self.encoder2, self.encoder3,
            self.mu_head, self.logstd_head,
            self.decoder1, self.decoder2, self.out
        ]

    def step_adam(self, grads, lr, t, eta1=0.9, eta2=0.999, eps=1e-8):
        """
        Adam gradient optimizer with initialized exponential decay weights.
        """
        for layer, (dW, db) in zip(self.parameters(), grads):
            layer.mW = eta1 * layer.mW + (1 - eta1) * dW
            layer.vW = eta2 * layer.vW + (1 - eta2) * (dW * dW)
            layer.mb = eta1 * layer.mb + (1 - eta1) * db
            layer.vb = eta2 * layer.vb + (1 - eta2) * (db * db)

            mW_hat = layer.mW / (1 - eta1**t); vW_hat = layer.vW / (1 - eta2**t)
            mb_hat = layer.mb / (1 - eta1**t); vb_hat = layer.vb / (1 - eta2**t)

            layer.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            layer.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    @staticmethod
    def _kld_mean(mu: np.ndarray, log_std: np.ndarray) -> float:
        s2 = np.exp(2.0 * log_std)
        kld_i = -0.5 * (1.0 + 2.0 * log_std - mu * mu - s2).sum(axis=1)
        return float(kld_i.mean())

    def train(self, X, epochs: int, batch_size: int, lr: float,
              shuffle=True, verbose=True, num_latent_samples: int = 1,
              beta_kld: float = 0.01):
        """
        Train VAE network with forward pass and Adams gradient optimizer.
        """
        X = X.astype(np.float32, copy=False)
        n = len(X); t = 0
        epoch_totals, epoch_recs, epoch_klds = [], [], []

        for ep in range(1, epochs + 1):
            idx = np.arange(n)
            if shuffle:
                np.random.shuffle(idx)
            Xs = X[idx]

            run_total = 0.0
            run_rec = 0.0
            run_kld = 0.0

            for i in range(0, n, batch_size):
                xb = Xs[i:i + batch_size]
                B = len(xb)
                if B == 0:
                    continue

                h = self._encode_hidden(xb)
                mu = self.mu_head.forward(h)
                log_std = self.logstd_head.forward(h)
                std = np.exp(log_std).astype(np.float32)

                K = max(1, int(num_latent_samples)) # monte carlo samples
                rec_acc = 0.0

                dW_out_acc = np.zeros_like(self.out.W);      db_out_acc = np.zeros_like(self.out.b)
                dW_d2_acc  = np.zeros_like(self.decoder2.W); db_d2_acc  = np.zeros_like(self.decoder2.b)
                dW_d1_acc  = np.zeros_like(self.decoder1.W); db_d1_acc  = np.zeros_like(self.decoder1.b)

                g_mu_sum = np.zeros_like(mu)
                g_ls_sum = np.zeros_like(log_std)

                for _ in range(K):
                    eps = self.rng.normal(size=std.shape).astype(np.float32)
                    z = (mu + std * eps).astype(np.float32)

                    xhat = self.decode(z)
                    rec_k, dL_dxhat = self.loss_fn(xhat, xb)
                    rec_acc += rec_k

                    dL_dxhat *= (1.0 / K)

                    g, dW_out, db_out = self.out.backward(dL_dxhat);   dW_out_acc += dW_out; db_out_acc += db_out
                    g, dW_d2,  db_d2  = self.decoder2.backward(g);     dW_d2_acc  += dW_d2;  db_d2_acc  += db_d2
                    g, dW_d1,  db_d1  = self.decoder1.backward(g);     dW_d1_acc  += dW_d1;  db_d1_acc  += db_d1

                    g_mu_sum += g
                    g_ls_sum += g * (std * eps)

                rec = rec_acc / float(K)

                s2 = np.exp(2.0 * log_std)
                kld_i = -0.5 * (1.0 + 2.0 * log_std - mu * mu - s2).sum(axis=1)
                kld = float(kld_i.mean())

                dK_dmu = (beta_kld * mu / B).astype(np.float32)
                dK_dls = (beta_kld * (s2 - 1.0) / B).astype(np.float32)

                g_mu_total = g_mu_sum + dK_dmu
                g_ls_total = g_ls_sum + dK_dls

                gh_mu, dW_mu, db_mu = self.mu_head.backward(g_mu_total)
                gh_ls, dW_ls, db_ls = self.logstd_head.backward(g_ls_total)
                gh = gh_mu + gh_ls

                g, dW_e3, db_e3 = self.encoder3.backward(gh)
                g, dW_e2, db_e2 = self.encoder2.backward(g)
                _, dW_e1, db_e1 = self.encoder1.backward(g)

                grads = []
                grads.append((dW_out_acc, db_out_acc))
                grads.append((dW_d2_acc,  db_d2_acc))
                grads.append((dW_d1_acc,  db_d1_acc))
                grads.append((dW_ls,      db_ls))
                grads.append((dW_mu,      db_mu))
                grads.append((dW_e3,      db_e3))
                grads.append((dW_e2,      db_e2))
                grads.append((dW_e1,      db_e1))
                grads = grads[::-1]

                t += 1
                self.step_adam(grads, lr, t)

                total = rec + beta_kld * kld
                run_total += total * B; run_rec += rec * B; run_kld += kld * B  # store raw KL for logging

            if verbose:
                print(
                    f"Epoch {ep:03d} | total loss={run_total / n:.6f} "
                    f"| reconstruction loss={run_rec / n:.6f} "
                    f"| kld={run_kld / n:.6f} "
                )

            epoch_totals.append(run_total / n)
            epoch_recs.append(run_rec / n)
            epoch_klds.append(run_kld / n)

        return epoch_totals, epoch_recs, epoch_klds

    def get_latent(self, X: np.ndarray):
        mu, _ = self.encode(X.astype(np.float32, copy=False))
        return mu

    def reconstruct(self, X: np.ndarray):
        return self.forward(X.astype(np.float32, copy=False))

    def reconstruct_mc_mean(self, X: np.ndarray, num_latent_samples: int):
        K = max(1, int(num_latent_samples))
        X = X.astype(np.float32, copy=False)
        mu, log_std = self.encode(X)
        std = np.exp(log_std).astype(np.float32)

        xhat_sum = np.zeros((X.shape[0], self.param_in), dtype=np.float32)
        for _ in range(K):
            eps = self.rng.normal(size=std.shape).astype(np.float32)
            z = (mu + std * eps).astype(np.float32)
            xhat_sum += self.decode(z)
        return (xhat_sum / float(K)).astype(np.float32)
