import numpy as np

def relu(x: float):
    return np.maximum(0, x)

def drelu(x: float):
    grad = np.zeros_like(x)
    grad[x > 0] = 1.0
    return grad

def tanh(x : float):
    return np.tanh(x)

def dtanh(x: float):
    y = np.tanh(x)
    return 1.0 - y * y

def sigmoid(x: float):
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(x: float):
    s = sigmoid(x)
    return s * (1.0 - s)


class Dense:
    def __init__(self, in_dim: int, out_dim: int, activation=None, rng=None):
        """
        activation fx.
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation

        if rng is None:
            rng = np.random.default_rng(0)
        if self.activation == "relu":
            self.W = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, out_dim)).astype(np.float32)
        else:
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            self.W = rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32)

        self.b = np.zeros(out_dim, dtype=np.float32)

        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        self.x = None
        self.z = None

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        if self.activation == "relu":
            return relu(self.z)
        elif self.activation == "tanh":
            return tanh(self.z)
        elif self.activation == "sigmoid":
            return sigmoid(self.z)
        elif self.activation is None:
            return self.z  
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def backward(self, grad_out):
        if self.activation == "relu":
            grad_z = grad_out * drelu(self.z)
        elif self.activation == "tanh":
            grad_z = grad_out * dtanh(self.z)
        elif self.activation == "sigmoid":
            grad_z = grad_out * dsigmoid(self.z)
        elif self.activation is None:
            grad_z = grad_out
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        dW = self.x.T @ grad_z
        db = grad_z.sum(axis=0)
        dx = grad_z @ self.W.T
        return dx, dW, db