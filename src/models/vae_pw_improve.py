import torch
import torch.nn as nn

## Inner import
from src.models.basic_model import VaeEncoder as Encoder
from src.models.basic_model import VaeDecoder as Decoder
from src.models.basic_model import EmbeddingMLP, SinusoidalPositionalEmbedding


# VAE_pw
class VAE_PW_II(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE_PW_II, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim + 2, hidden_dim, 1)
        self.dltembed = SinusoidalPositionalEmbedding(10)
        self.ttmembed = SinusoidalPositionalEmbedding(10)
        self.dltemb_net = EmbeddingMLP(10, latent_dim)
        self.ttmemb_net = EmbeddingMLP(10, latent_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, surface, pw_grid):
        mean, logvar = self.encoder(surface)
        z = self.reparameterize(mean, logvar)
        delta, ttm = pw_grid[:, 0], pw_grid[:, 1]
        delta_embed, ttm_embed = self.dltembed(delta), self.ttmembed(ttm)
        delta_out, ttm_out = self.dltemb_net(delta_embed), self.ttmemb_net(ttm_embed)
        z_combined = z + delta_out + ttm_out
        # z_combined = torch.cat([z, pw_grid.view(-1,2)], dim=1)
        pred = self.decoder(z_combined)
        return pred, mean, logvar

    @staticmethod
    # Loss function
    def loss_function(pred, pw_vol, mean, logvar):
        MSE = nn.functional.mse_loss(
            pred, pw_vol, reduction="sum"
        )  
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return MSE + KLD
