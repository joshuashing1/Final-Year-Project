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
        
        self.encoder1 = Dense(self.param_in, 30, activation=self.activation, rng=rng)
        self.encoder2 = Dense(30, 30, activation=self.activation, rng=rng)
        self.encoder3 = Dense(30, 12, activation=self.activation, rng=rng)  
        
        self.decoder1 = Dense(12, 30, activation=self.activation, rng=rng)
        self.decoder2 = Dense(30, 30, activation=self.activation, rng=rng)
        self.out = Dense(30, self.param_in, activation=None, rng=rng)  
    
    def encode(self, x: float):
        x = self.encoder1.forward(x)
        x = self.encoder2.forward(x)
        z = self.encoder3.forward(x)
        return z  

    def decode(self, z: float):
        x = self.decoder1.forward(z)
        x = self.decoder2.forward(x)
        x = self.out.forward(x)
        return x  

    def forward(self, x: float):
        return self.decode(self.encode(x))

    def loss_fn(self, pred, y: float):
        diff = pred - y
        return (diff**2).mean(), (2.0 / len(y)) * diff

    def parameters(self):
        return [self.encoder1, self.encoder2, self.encoder3, self.decoder1, self.decoder2, self.out]

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
        epoch_losses = []
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
                g, dW_decoder2, db_decoder2 = self.decoder2.backward(g)
                grads.append((dW_decoder2, db_decoder2))
                g, dW_decoder1, db_decoder1 = self.decoder1.backward(g)
                grads.append((dW_decoder1, db_decoder1))

                g, dW_encoder3, db_encoder3 = self.encoder3.backward(g)
                grads.append((dW_encoder3, db_encoder3))
                g, dW_encoder2, db_encoder2 = self.encoder2.backward(g)
                grads.append((dW_encoder2, db_encoder2))
                _, dW_encoder1, db_encoder1 = self.encoder1.backward(g)
                grads.append((dW_encoder1, db_encoder1))

                grads = grads[::-1]

                t += 1
                self.step_adam(grads, lr, t)

            if verbose:
                print(f"Epoch {ep:03d} | Loss: {running / n:.6f}")
                

    def get_latent(self, X):
        return self.encode(X.astype(np.float32, copy=False))

    def reconstruct(self, X):
        return self.forward(X.astype(np.float32, copy=False))