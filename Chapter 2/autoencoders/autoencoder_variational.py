import numpy as np

from nn_fn import Dense


class BetaVAE_NP:
    """
    NumPy β-VAE (MLP) with the same structure and optimizer style as your AE:

    Encoder: param_in -> 30 -> 30 -> 13 -> (mu: latent_dim, logvar: latent_dim)
    Reparam: z = mu + exp(0.5*logvar) * eps
    Decoder: latent_dim -> 30 -> 30 -> param_in

    Reconstruction loss: MSE
    KL loss: standard Gaussian prior
    Loss types:
      - 'H': recon + beta * kld_weight * KLD
      - 'B': recon + gamma * kld_weight * |KLD - C(t)|   (capacity annealing)
    """

    num_iter = 0  # global-ish counter for capacity scheduling

    def __init__(self,
                 param_in: int,
                 latent_dim: int = 13,
                 beta: float = 4.0,
                 gamma: float = 1000.0,
                 max_capacity: float = 25.0,
                 Capacity_max_iter: int = int(1e5),
                 loss_type: str = 'B',
                 rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        self.rng = rng

        self.param_in = int(param_in)
        self.latent_dim = int(latent_dim)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.loss_type = loss_type
        self.C_max = float(max_capacity)
        self.C_stop_iter = int(Capacity_max_iter)

        # ----- encoder -----
        self.e1 = Dense(self.param_in, 30, activation="relu", rng=rng)
        self.e2 = Dense(30, 30, activation="relu", rng=rng)
        self.e3 = Dense(30, 13, activation="relu", rng=rng)

        # two linear heads for mean and log-variance
        self.fc_mu = Dense(13, self.latent_dim, activation=None, rng=rng)
        self.fc_lv = Dense(13, self.latent_dim, activation=None, rng=rng)

        # ----- decoder -----
        self.d1 = Dense(self.latent_dim, 30, activation="relu", rng=rng)
        self.d2 = Dense(30, 30, activation="relu", rng=rng)
        self.out = Dense(30, self.param_in, activation=None, rng=rng)

        # caches needed for backprop through reparam
        self.mu = None
        self.logvar = None
        self.std = None
        self.eps = None
        self.z = None

    # ----- forward passes -----
    def encode_hidden(self, x):
        x = self.e1.forward(x)
        x = self.e2.forward(x)
        h = self.e3.forward(x)
        return h

    def encode(self, x):
        h = self.encode_hidden(x)
        mu = self.fc_mu.forward(h)
        logvar = self.fc_lv.forward(h)
        self.mu, self.logvar = mu, logvar
        self.std = np.exp(0.5 * logvar).astype(np.float32)
        self.eps = self.rng.normal(size=self.std.shape).astype(np.float32)
        z = mu + self.std * self.eps
        self.z = z
        return mu, logvar, z

    def decode(self, z):
        x = self.d1.forward(z)
        x = self.d2.forward(x)
        x = self.out.forward(x)
        return x

    def forward(self, x):
        mu, logvar, z = self.encode(x.astype(np.float32, copy=False))
        xhat = self.decode(z)
        return xhat, mu, logvar, z

    # ----- losses -----
    @staticmethod
    def recon_mse(pred, y):
        diff = pred - y
        # return scalar mse and grad wrt pred
        return (diff**2).mean(), (2.0 / len(y)) * diff

    @staticmethod
    def kld_terms(mu, logvar):
        # per-sample KLD:  -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        exp_lv = np.exp(logvar)
        kld_i = -0.5 * (1 + logvar - mu*mu - exp_lv).sum(axis=1)  # shape (B,)
        kld = kld_i.mean()
        # grads of mean KLD wrt mu and logvar
        B = mu.shape[0]
        dmu = (mu / B)                              # ∂/∂mu (mean KLD)
        dlogvar = (0.5 * (exp_lv - 1.0) / B)        # ∂/∂logvar (mean KLD)
        return kld, dmu.astype(np.float32), dlogvar.astype(np.float32)

    # ----- params & optimizer -----
    def parameters(self):
        # Order matters; grads will be zipped in this order
        return [self.e1, self.e2, self.e3, self.fc_mu, self.fc_lv, self.d1, self.d2, self.out]

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

    # ----- training -----
    def train(self, X, epochs: int, batch_size: int, lr: float,
              kld_weight: float = 1.0, shuffle=True, verbose=True):
        """
        kld_weight ~ M/N (minibatch scaling) if you want to mimic PyTorch impls.
        """
        X = X.astype(np.float32, copy=False)
        n = len(X); t = 0
        for ep in range(1, epochs + 1):
            idx = np.arange(n)
            if shuffle:
                np.random.shuffle(idx)
            Xs = X[idx]

            sum_loss = 0.0
            sum_rec = 0.0
            sum_kld = 0.0

            for i in range(0, n, batch_size):
                xb = Xs[i:i + batch_size]
                B = len(xb)

                # ----- forward -----
                xhat, mu, logvar, z = self.forward(xb)

                # recon loss
                rec_loss, dL_dxhat = self.recon_mse(xhat, xb)

                # kld loss (+ grads wrt mu, logvar)
                kld, dK_dmu, dK_dlv = self.kld_terms(mu, logvar)

                # capacity / weighting
                self.__class__.num_iter += 1
                if self.loss_type == 'H':
                    # loss = rec + beta * kld_weight * KLD
                    kld_factor = self.beta * kld_weight
                    kld_loss_term = kld_factor * kld
                    # grads multiplier for KLD parts
                    scale = kld_factor
                elif self.loss_type == 'B':
                    # loss = rec + gamma * kld_weight * |KLD - C(t)|
                    C = min(self.C_max, self.C_max * (self.num_iter / self.C_stop_iter))
                    diff = kld - C
                    kld_loss_term = self.gamma * kld_weight * np.abs(diff)
                    # subgradient sign for |.|; 0 if exactly equal
                    sgn = 0.0 if diff == 0 else (1.0 if diff > 0 else -1.0)
                    scale = self.gamma * kld_weight * sgn
                else:
                    raise ValueError("loss_type must be 'H' or 'B'.")

                total_loss = rec_loss + kld_loss_term

                # ----- backward -----
                grads = []

                # Backprop decoder from reconstruction loss
                g = dL_dxhat
                g, dW_out, db_out = self.out.backward(g)
                grads.append((dW_out, db_out))
                g, dW_d2, db_d2 = self.d2.backward(g)
                grads.append((dW_d2, db_d2))
                g, dW_d1, db_d1 = self.d1.backward(g)
                grads.append((dW_d1, db_d1))
                # g is now dL/dz from reconstruction

                # Reparameterization backprop pieces:
                # z = mu + std * eps, std = exp(0.5*logvar)
                std = self.std
                eps = self.eps

                # contributions from recon through z to mu/logvar
                g_mu_from_recon = g                                  # ∂z/∂mu = 1
                g_lv_from_recon = g * (0.5 * std * eps)              # ∂z/∂logvar = 0.5*std*eps

                # add KL contributions (scaled)
                g_mu_total = g_mu_from_recon + scale * dK_dmu
                g_lv_total = g_lv_from_recon + scale * dK_dlv

                # backprop through heads
                gh_mu, dW_mu, db_mu = self.fc_mu.backward(g_mu_total)
                gh_lv, dW_lv, db_lv = self.fc_lv.backward(g_lv_total)
                # combine grads into shared encoder top hidden
                gh = gh_mu + gh_lv

                # backprop through encoder trunk
                gh, dW_e3, db_e3 = self.e3.backward(gh)
                gh, dW_e2, db_e2 = self.e2.backward(gh)
                _,  dW_e1, db_e1 = self.e1.backward(gh)

                # Collect grads in parameters() order:
                grads.extend([(dW_e3, db_e3), (dW_e2, db_e2), (dW_e1, db_e1)])
                # But current list order is decoder-first; reverse to match parameters()
                # parameters(): [e1,e2,e3,fc_mu,fc_lv,d1,d2,out]
                grads = [
                    (dW_e1, db_e1), (dW_e2, db_e2), (dW_e3, db_e3),
                    (dW_mu, db_mu), (dW_lv, db_lv),
                    (dW_d1, db_d1), (dW_d2, db_d2), (dW_out, db_out)
                ]

                # Adam step
                t += 1
                self.step_adam(grads, lr, t)

                sum_loss += total_loss * B
                sum_rec  += rec_loss * B
                sum_kld  += kld * B

            if verbose:
                denom = float(n)
                print(f"Epoch {ep:03d} | Total: {sum_loss/denom:.6f} | Recon: {sum_rec/denom:.6f} | KLD: {sum_kld/denom:.6f}")

    # ----- convenience -----
    def get_latent(self, X):
        X = X.astype(np.float32, copy=False)
        _ = self.encode(X)
        return self.z  # sampled z

    def encode_mean(self, X):
        X = X.astype(np.float32, copy=False)
        h = self.encode_hidden(X)
        mu = self.fc_mu.forward(h)
        return mu

    def reconstruct(self, X):
        X = X.astype(np.float32, copy=False)
        xhat, *_ = self.forward(X)
        return xhat
