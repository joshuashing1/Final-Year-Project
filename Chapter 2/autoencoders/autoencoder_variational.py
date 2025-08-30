# variational_autoencoder.py
import numpy as np
from typing import List

from nn_fn import Dense

class VariationalNN:
    """
    Variational Autoencoder (NumPy) — mean-field with log_std parameterization

    Encoder: param_in -> 30 -> 30 -> 13  (top hidden)
             heads: top(13) -> mu(latent_dim), top(13) -> log_std(latent_dim)
    Decoder: latent_dim -> 30 -> 30 -> param_in
    Reparameterization: z = mu + exp(log_std) * eps
    Loss: total = recon_MSE + beta * kld_weight * KLD
    """

    def __init__(self,
                 param_in: int,
                 activation: str,
                 latent_dim: int = 13,
                 beta: float = 1.0,
                 rng=None):
        self.param_in = int(param_in)
        self.activation = activation
        self.latent_dim = int(latent_dim)
        self.beta = float(beta)
        self.rng = np.random.default_rng(0) if rng is None else rng

        # encoder trunk
        self.encoder1 = Dense(self.param_in, 30, activation=self.activation, rng=self.rng)
        self.encoder2 = Dense(30, 30, activation=self.activation, rng=self.rng)
        self.encoder3 = Dense(30, 13, activation=self.activation, rng=self.rng)  # top hidden dim = 13

        # heads
        self.mu_head     = Dense(13, self.latent_dim, activation=None, rng=self.rng)
        self.logstd_head = Dense(13, self.latent_dim, activation=None, rng=self.rng)

        # decoder
        self.decoder1 = Dense(self.latent_dim, 30, activation=self.activation, rng=self.rng)
        self.decoder2 = Dense(30, 30, activation=self.activation, rng=self.rng)
        self.out      = Dense(30, self.param_in, activation=None, rng=self.rng)

        self._t = 0  # Adam step counter

    # ---------- core ops ----------
    def _encode_hidden(self, x: np.ndarray) -> np.ndarray:
        h = self.encoder1.forward(x)
        h = self.encoder2.forward(h)
        h = self.encoder3.forward(h)
        return h

    def encode(self, x: np.ndarray):
        h = self._encode_hidden(x.astype(np.float32, copy=False))
        mu = self.mu_head.forward(h)
        log_std = self.logstd_head.forward(h)
        return mu, log_std

    def reparameterize(self, mu: np.ndarray, log_std: np.ndarray) -> np.ndarray:
        std = np.exp(log_std).astype(np.float32)
        eps = self.rng.normal(size=std.shape).astype(np.float32)
        return (mu + std * eps).astype(np.float32)

    def decode(self, z: np.ndarray) -> np.ndarray:
        h = self.decoder1.forward(z)
        h = self.decoder2.forward(h)
        return self.out.forward(h)

    def forward(self, x: np.ndarray):
        mu, log_std = self.encode(x)
        z = self.reparameterize(mu, log_std)
        return self.decode(z)

    def loss_fn(self, pred: np.ndarray, y: np.ndarray):
        diff = (pred - y)
        rec = float((diff * diff).mean())
        dL_dpred = (2.0 / len(y)) * diff   # mean over batch; decoder backprop handles 1/K
        return rec, dL_dpred

    def parameters(self) -> List[Dense]:
        return [
            self.encoder1, self.encoder2, self.encoder3,
            self.mu_head, self.logstd_head,
            self.decoder1, self.decoder2, self.out
        ]

    def step_adam(self, grads, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        for layer, (dW, db) in zip(self.parameters(), grads):
            layer.mW = beta1 * layer.mW + (1 - beta1) * dW
            layer.vW = beta2 * layer.vW + (1 - beta2) * (dW * dW)
            layer.mb = beta1 * layer.mb + (1 - beta1) * db
            layer.vb = beta2 * layer.vb + (1 - beta2) * (db * db)

            mW_hat = layer.mW / (1 - beta1**t); vW_hat = layer.vW / (1 - beta2**t)
            mb_hat = layer.mb / (1 - beta1**t); vb_hat = layer.vb / (1 - beta2**t)

            layer.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            layer.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    @staticmethod
    def _kld_mean(mu: np.ndarray, log_std: np.ndarray) -> float:
        s2 = np.exp(2.0 * log_std)
        kld_i = -0.5 * (1.0 + 2.0 * log_std - mu * mu - s2).sum(axis=1)
        return float(kld_i.mean())

    def train(self,
              X: np.ndarray,
              epochs: int,
              batch_size: int,
              lr: float,
              shuffle: bool = True,
              verbose: bool = True,
              kld_weight: float = 1.0,
              num_latent_samples: int = 1):
        X = X.astype(np.float32, copy=False)
        n = len(X)
        epoch_totals, epoch_recs, epoch_klds = [], [], []

        for ep in range(1, epochs + 1):
            idx = np.arange(n)
            if shuffle: np.random.shuffle(idx)
            Xs = X[idx]

            run_total = run_rec = run_kld = 0.0

            for i in range(0, n, batch_size):
                xb = Xs[i:i + batch_size]
                B = len(xb)
                if B == 0: continue

                # encode once
                h = self._encode_hidden(xb)
                mu = self.mu_head.forward(h)
                log_std = self.logstd_head.forward(h)
                std = np.exp(log_std).astype(np.float32)

                # accumulators
                K = max(1, int(num_latent_samples))
                rec_acc = 0.0
                dW_out_acc = np.zeros_like(self.out.W);      db_out_acc = np.zeros_like(self.out.b)
                dW_d2_acc  = np.zeros_like(self.decoder2.W); db_d2_acc  = np.zeros_like(self.decoder2.b)
                dW_d1_acc  = np.zeros_like(self.decoder1.W); db_d1_acc  = np.zeros_like(self.decoder1.b)
                g_mu_sum = np.zeros_like(mu)
                g_ls_sum = np.zeros_like(log_std)

                # Monte-Carlo samples
                for _ in range(K):
                    eps = self.rng.normal(size=std.shape).astype(np.float32)
                    z = (mu + std * eps).astype(np.float32)

                    xhat = self.decode(z)
                    rec_k, dL_dxhat = self.loss_fn(xhat, xb)
                    rec_acc += rec_k
                    dL_dxhat *= (1.0 / K)  # average recon grad over K

                    # decoder backprop
                    g, dW_out, db_out = self.out.backward(dL_dxhat);      dW_out_acc += dW_out; db_out_acc += db_out
                    g, dW_d2,  db_d2  = self.decoder2.backward(g);        dW_d2_acc  += dW_d2;  db_d2_acc  += db_d2
                    g, dW_d1,  db_d1  = self.decoder1.backward(g);        dW_d1_acc  += dW_d1;  db_d1_acc  += db_d1

                    # chain rule through z = mu + exp(log_std) * eps
                    g_mu_sum += g
                    g_ls_sum += g * (std * eps)

                rec = rec_acc / float(K)

                # analytic KL (batch mean)
                s2 = np.exp(2.0 * log_std)
                kld_i = -0.5 * (1.0 + 2.0 * log_std - mu * mu - s2).sum(axis=1)
                kld = float(kld_i.mean())

                # add KL grads (scaled)
                scale = self.beta * kld_weight
                dK_dmu =  (mu / B).astype(np.float32)
                dK_dls = ((s2 - 1.0) / B).astype(np.float32)
                g_mu_total = g_mu_sum + scale * dK_dmu
                g_ls_total = g_ls_sum + scale * dK_dls

                # heads and encoder backprop
                gh_mu, dW_mu, db_mu = self.mu_head.backward(g_mu_total)
                gh_ls, dW_ls, db_ls = self.logstd_head.backward(g_ls_total)
                gh = gh_mu + gh_ls

                g, dW_e3, db_e3 = self.encoder3.backward(gh)
                g, dW_e2, db_e2 = self.encoder2.backward(g)
                _, dW_e1, db_e1 = self.encoder1.backward(g)

                # >>> IMPORTANT: build grads in the SAME order as parameters()
                grads = [
                    (dW_e1,     db_e1),     # encoder1
                    (dW_e2,     db_e2),     # encoder2
                    (dW_e3,     db_e3),     # encoder3
                    (dW_mu,     db_mu),     # mu_head
                    (dW_ls,     db_ls),     # logstd_head
                    (dW_d1_acc, db_d1_acc), # decoder1
                    (dW_d2_acc, db_d2_acc), # decoder2
                    (dW_out_acc,db_out_acc) # out
                ]

                self._t += 1
                self.step_adam(grads, lr, self._t)

                total = rec + self.beta * kld_weight * kld
                run_total += total * B; run_rec += rec * B; run_kld += kld * B

            # epoch means
            epoch_totals.append(run_total / n)
            epoch_recs.append(run_rec / n)
            epoch_klds.append(run_kld / n)

            if verbose:
                print(f"Epoch {ep:03d} | total={epoch_totals[-1]:.6f} | recon={epoch_recs[-1]:.6f} | kld={epoch_klds[-1]:.6f}")

        return epoch_totals, epoch_recs, epoch_klds

    # ---------- helpers ----------
    def get_latent(self, X: np.ndarray):
        mu, _ = self.encode(X.astype(np.float32, copy=False))
        return mu

    def reconstruct(self, X: np.ndarray):
        return self.forward(X.astype(np.float32, copy=False))

    def reconstruct_mean(self, X: np.ndarray):
        mu, _ = self.encode(X.astype(np.float32, copy=False))
        return self.decode(mu).astype(np.float32)

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
