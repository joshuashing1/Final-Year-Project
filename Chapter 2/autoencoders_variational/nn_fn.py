import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_x = F.relu(self.fc1(x))
        p_x = F.relu(self.fc2(p_x))
        return p_x


class LatentZ(nn.Module):
    def __init__(self, hidden_size: int, latent_size: int):
        super().__init__()
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, p_x: torch.Tensor):
        mu = self.mu(p_x)
        logvar = self.logvar(p_x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = std * eps + mu  # reparameterization
        return z, logvar, mu


class Decoder(nn.Module):
    def __init__(self, latent_size: int, hidden_size: int, input_size: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, z_x: torch.Tensor) -> torch.Tensor:
        q_x = F.relu(self.fc1(z_x))
        q_x = torch.sigmoid(self.fc2(q_x))
        return q_x


def loss_criterion(inputs: torch.Tensor,
                   targets: torch.Tensor,
                   logvar: torch.Tensor,
                   mu: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy reconstruction + KL(q(z|x) || N(0, I))."""
    bce_loss = F.binary_cross_entropy(inputs, targets, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce_loss + kl_loss
