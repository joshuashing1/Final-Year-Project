import tensorflow as tf
from tensorflow.keras import layers, Model

class BetaVAE(Model):
    """
    β-VAE in Keras with 'H' (Higgins) and 'B' (Burgess capacity) losses.
    Structure: 30 -> 30 -> 13 -> 30 -> 30
    """
    def __init__(self,
                 in_dim: int = 30,
                 latent_dim: int = 13,
                 beta: float = 4.0,
                 gamma: float = 1000.0,
                 max_capacity: float = 25.0,
                 capacity_max_iter: int = int(1e5),
                 loss_type: str = 'B',
                 use_tanh_out: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.loss_type = loss_type
        self.C_max = tf.constant(float(max_capacity), dtype=tf.float32)
        self.C_stop_iter = tf.constant(int(capacity_max_iter), dtype=tf.int64)
        self.use_tanh_out = use_tanh_out

        # global iteration counter (class-like)
        self.num_iter = tf.Variable(0, dtype=tf.int64, trainable=False)

        # ----- Encoder: 30 -> 30 -> (mu, logvar) -----
        self.enc_dense = layers.Dense(30, use_bias=True)
        self.enc_bn = layers.BatchNormalization()
        self.enc_act = layers.LeakyReLU()

        self.fc_mu = layers.Dense(latent_dim)       # μ(x)
        self.fc_logvar = layers.Dense(latent_dim)   # log σ^2(x)

        # ----- Decoder: 13 -> 30 -> 30 -----
        self.dec_dense = layers.Dense(30, use_bias=True)
        self.dec_bn = layers.BatchNormalization()
        self.dec_act = layers.LeakyReLU()
        self.dec_out = layers.Dense(in_dim)
        self.out_act = layers.Activation("tanh") if use_tanh_out else layers.Activation("linear")

        # trackers (optional, show in model.fit logs)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kld")

        # minibatch/dataset scaling (set from outside if you want)
        # e.g., model.M_N = batch_size / dataset_size
        self.M_N = tf.Variable(1.0, dtype=tf.float32, trainable=False)

    # ---------- Core ops ----------
    def encode(self, x, training=False):
        """
        x: [B, 30] -> returns (mu, logvar) each [B, 13]
        """
        h = self.enc_dense(x)
        h = self.enc_bn(h, training=training)
        h = self.enc_act(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar, training=False):
        """
        z = mu + exp(0.5*logvar) * eps, eps ~ N(0, I)
        """
        if training:
            std = tf.exp(0.5 * logvar)
            eps = tf.random.normal(shape=tf.shape(std), dtype=std.dtype)
            return mu + std * eps
        else:
            # during inference, just use the mean
            return mu

    def decode(self, z, training=False):
        """
        z: [B, 13] -> x_hat: [B, 30]
        """
        h = self.dec_dense(z)
        h = self.dec_bn(h, training=training)
        h = self.dec_act(h)
        x_hat = self.dec_out(h)
        return self.out_act(x_hat)

    def call(self, inputs, training=False):
        """
        Keras forward pass; returns (x_hat, x, mu, logvar)
        """
        mu, logvar = self.encode(inputs, training=training)
        z = self.reparameterize(mu, logvar, training=training)
        x_hat = self.decode(z, training=training)
        return x_hat, inputs, mu, logvar

    # ---------- Loss pieces ----------
    @staticmethod
    def mse_recon(x_hat, x):
        # Mean squared error over batch and dimensions
        return tf.reduce_mean(tf.reduce_mean(tf.math.squared_difference(x_hat, x), axis=-1))

    @staticmethod
    def kl_diag_gauss(mu, logvar):
        # KL(q(z|x) || N(0,I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kld_per_example = -0.5 * tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=-1)
        return tf.reduce_mean(kld_per_example)

    def compute_total_loss(self, x_hat, x, mu, logvar):
        recon = self.mse_recon(x_hat, x)
        kld = self.kl_diag_gauss(mu, logvar)

        if self.loss_type == 'H':
            total = recon + self.beta * self.M_N * kld
        elif self.loss_type == 'B':
            # capacity schedule
            self.num_iter.assign_add(1)
            # C = clamp(C_max / C_stop_iter * num_iter, 0, C_max)
            c_float = tf.cast(self.num_iter, tf.float32) * (self.C_max / tf.cast(self.C_stop_iter, tf.float32))
            C = tf.clip_by_value(c_float, 0.0, self.C_max)
            total = recon + self.gamma * self.M_N * tf.abs(kld - C)
        else:
            raise ValueError("Undefined loss type. Use 'H' or 'B'.")

        return total, recon, kld

    # ---------- Custom training loop hooks ----------
    def train_step(self, data):
        if isinstance(data, tuple) or isinstance(data, list):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            x_hat, x_in, mu, logvar = self(x, training=True)
            total, recon, kld = self.compute_total_loss(x_hat, x_in, mu, logvar)

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kld)
        return {"loss": self.total_loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "kld": self.kl_loss_tracker.result()}

    def test_step(self, data):
        if isinstance(data, tuple) or isinstance(data, list):
            x = data[0]
        else:
            x = data
        x_hat, x_in, mu, logvar = self(x, training=False)
        total, recon, kld = self.compute_total_loss(x_hat, x_in, mu, logvar)
        return {"loss": total, "recon_loss": recon, "kld": kld}

    @property
    def metrics(self):
        # reset each epoch
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    # ---------- Utilities ----------
    def sample(self, num_samples: int):
        """
        Draw z ~ N(0,I) and decode -> [num_samples, 30]
        """
        z = tf.random.normal([num_samples, self.latent_dim], dtype=tf.float32)
        return self.decode(z, training=False)

    def generate(self, x):
        """
        Reconstruct x -> [B, 30]
        """
        x_hat, _, _, _ = self(x, training=False)
        return x_hat
