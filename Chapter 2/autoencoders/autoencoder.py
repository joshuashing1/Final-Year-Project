import numpy as np

from nn_fn import Dense

class AutoencoderNN:
    """
    Encoder: param_in -> 30 -> 30 -> 13
    Decoder: 13 -> 30 -> 30 -> param_in
    All hidden activations = ReLU
    Output layer = linear (for reconstruction with MSE).
    """
    
    def __init__(self, param_in: int, activation: str, rng=None):
        self.param_in = param_in
        self.activation = activation
        
        self.e1 = Dense(self.param_in, 30, activation=self.activation, rng=rng)
        self.e2 = Dense(30, 30, activation=self.activation, rng=rng)
        self.e3 = Dense(30, 13, activation=self.activation, rng=rng)  
        
        self.d1 = Dense(13, 30, activation=self.activation, rng=rng)
        self.d2 = Dense(30, 30, activation=self.activation, rng=rng)
        self.out = Dense(30, self.param_in, activation=None, rng=rng)  
    
    def encode(self, x: float):
        x = self.e1.forward(x)
        x = self.e2.forward(x)
        z = self.e3.forward(x)
        return z  

    def decode(self, z: float):
        x = self.d1.forward(z)
        x = self.d2.forward(x)
        x = self.out.forward(x)
        return x  

    def forward(self, x: float):
        return self.decode(self.encode(x))

    def loss_fn(self, pred, y: float):
        diff = pred - y
        return (diff**2).mean(), (2.0 / len(y)) * diff

    def parameters(self):
        return [self.e1, self.e2, self.e3, self.d1, self.d2, self.out]

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

    def train(self, X, epochs: int, batch_size: int, lr: float, shuffle=True, verbose=True):
        X = X.astype(np.float32, copy=False)
        n = len(X); t = 0
        for ep in range(1, epochs + 1):
            idx = np.arange(n)
            if shuffle:
                np.random.shuffle(idx)
            Xs = X[idx]

            running = 0.0
            for i in range(0, n, batch_size):
                xb = Xs[i:i + batch_size]

                pred = self.forward(xb)
                loss, dL_dpred = self.loss_fn(pred, xb)  
                running += loss * len(xb)

                grads = []
                g = dL_dpred
                g, dW_out, db_out = self.out.backward(g)
                grads.append((dW_out, db_out))
                g, dW_d2, db_d2 = self.d2.backward(g)
                grads.append((dW_d2, db_d2))
                g, dW_d1, db_d1 = self.d1.backward(g)
                grads.append((dW_d1, db_d1))

                g, dW_e3, db_e3 = self.e3.backward(g)
                grads.append((dW_e3, db_e3))
                g, dW_e2, db_e2 = self.e2.backward(g)
                grads.append((dW_e2, db_e2))
                _, dW_e1, db_e1 = self.e1.backward(g)
                grads.append((dW_e1, db_e1))

                grads = grads[::-1]

                t += 1
                self.step_adam(grads, lr, t)

            if verbose:
                print(f"Epoch {ep:03d} | Loss: {running / n:.6f}")

    # ----- convenience -----
    def get_latent(self, X):
        return self.encode(X.astype(np.float32, copy=False))

    def reconstruct(self, X):
        return self.forward(X.astype(np.float32, copy=False))