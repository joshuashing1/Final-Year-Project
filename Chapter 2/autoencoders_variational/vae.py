import torch
import torch.nn as nn

from nn_fn import Encoder, LatentZ, Decoder


class VAE(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, latent_size: int = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.encoder = Encoder(input_size, hidden_size)
        self.latent_z = LatentZ(hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, input_size)

    def forward(self, x: torch.Tensor):
        p_x = self.encoder(x)
        z, logvar, mu = self.latent_z(p_x)
        q_z = self.decoder(z)
        return q_z, logvar, mu, z
