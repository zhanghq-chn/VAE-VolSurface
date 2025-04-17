import torch
import torch.nn as nn
import numpy as np
import math

## inner import 
from src.models.basic_model import EmbeddingMLP, SinusoidalPositionalEmbedding
    
class NoisePredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim, embedding_dim):
        super(NoisePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),  # add t to predict noise
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.t_embed = SinusoidalPositionalEmbedding(embedding_dim)
        self.t_net = EmbeddingMLP(embedding_dim, latent_dim)
        
    def forward(self, z, t_norm):
        #### FIX
        t_embed = self.t_embed(t_norm.squeeze()) # shape: [batch_size, embedding_dim]
        t_out = self.t_net(t_embed)  # shape: [batch_size, latent_dim]
        z_t = z + t_out
        # predict noise
        return self.model(z_t)


class LDM(nn.Module):
    def __init__(self, autoencoder, config):
        super(LDM, self).__init__()
        # default vae model
        # FIXED PARAMS
        self.autoencoder = autoencoder
        self.autoencoder.requires_grad_(False)
        self.betas = None
        # TO TRAIN
        self.noise_predictor = NoisePredictor(config["latent_dim"], config["hidden_dim"], config["embedding_dim"])

    def forward(self, x, t):
        # encode input into latent space
        mean, logvar = self.autoencoder.encoder(x)
        noise_pred = self.noise_predictor(mean, t)
        return noise_pred

    def set_beta(self, timesteps, device):
        self.betas = nn.Parameter(self.linear_beta_schedule(timesteps).to(device), requires_grad=False)

    def get_loss(self, data, t_rand):
        z, _ = self.autoencoder.encoder(data)
        z_t, noise = self.forward_diffusion(z, t_rand, self.betas)
        noise_pred = self.noise_predictor(z_t, t_rand/len(self.betas))
        return self.loss_function(noise_pred, noise)
    
    # Diffusion Process Utilities
    @staticmethod
    def linear_beta_schedule(timesteps):
        beta_start = 1e-4
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    @staticmethod
    def forward_diffusion(z, t, betas):
        sqrt_alphas = torch.sqrt(1 - betas[t]).view(-1, 1)
        sqrt_one_minus_alphas = torch.sqrt(betas[t]).view(-1, 1)
        noise = torch.randn_like(z)
        z_t = sqrt_alphas * z + sqrt_one_minus_alphas * noise
        return z_t, noise

    # Loss Function
    @staticmethod
    def loss_function(noise_pred, noise):
        return nn.functional.mse_loss(noise_pred, noise)
