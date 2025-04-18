import torch
import torch.nn as nn

## inner import 
from src.models.basic_model import VaeEncoder as Encoder
from src.models.basic_model import VaeDecoder as Decoder


# VAE
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    # Key part: Reparameterization trick
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z)
        return x_recon, mean, logvar

    # Loss function
    # TODO: penalty coefficient for KL divergence
    def loss_function(x_recon, x, mean, logvar):
        MSE = nn.functional.mse_loss(x_recon, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())  # 0.5
        return MSE + KLD
