import torch
import torch.nn as nn
from abc import ABC, abstractmethod
# from rotary_embedding_torch import RotaryEmbedding

## Inner import
from src.models.basic_model import VaeEncoder as Encoder
from src.models.basic_model import VaeDecoder as Decoder
from src.models.basic_model import EmbeddingMLP, SinusoidalPositionalEmbedding, PositionalEmbedding2D


class VAE_PW(nn.Module, ABC):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        '''Set up the model:
        1. Encoder/Decoder
        2. Positional Embedding
        3. Embedding MLP'''
        super(VAE_PW, self).__init__()
        
    
    @abstractmethod
    def forward(self, surface, pw_grid):
        '''Forward pass of the model'''
        pass
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    @staticmethod
    # Loss function
    def loss_function(pred, pw_vol, mean, logvar):
        MSE = nn.functional.mse_loss(
            pred, pw_vol, reduction="sum"
        )  
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return MSE + KLD

# VAE_pw
class VAE_PW_I(VAE_PW): # replication of the paper, cat k and t to the latent space
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE_PW_I, self).__init__( input_dim, hidden_dim, latent_dim)
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim + 2, hidden_dim, 1)

    def forward(self, surface, pw_grid):
        mean, logvar = self.encoder(surface)
        z = self.reparameterize(mean, logvar)
        delta, ttm = pw_grid[:, 0], pw_grid[:, 1]
        z_combined = torch.cat([z, delta.view(-1, 1), ttm.view(-1, 1)], dim=1)
        pred = self.decoder(z_combined)
        return pred, mean, logvar
    
    
class VAE_PW_II(VAE_PW): # improved version, add k&t embedding
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE_PW_II, self).__init__( input_dim, hidden_dim, latent_dim)
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, 1)
        
        self.dltemb_net = EmbeddingMLP(10, latent_dim)
        self.ttmemb_net = EmbeddingMLP(10,latent_dim)
        self.dltembed = SinusoidalPositionalEmbedding(10)
        self.ttmembed = SinusoidalPositionalEmbedding(10)
        

    def forward(self, surface, pw_grid):
        mean, logvar = self.encoder(surface)
        z = self.reparameterize(mean, logvar)
        delta, ttm = pw_grid[:, 0], pw_grid[:, 1]
        delta_embed, ttm_embed = self.dltembed(delta), self.ttmembed(ttm)
        delta_out, ttm_out = self.dltemb_net(delta_embed), self.ttmemb_net(ttm_embed)
        z_combined = z + delta_out + ttm_out
        # z_combined = torch.cat([z, delta_out, ttm_out], dim=1)
        pred = self.decoder(z_combined)
        return pred, mean, logvar

