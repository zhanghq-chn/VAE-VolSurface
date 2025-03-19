import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

from src.models import vae

class NoisePredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(NoisePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim), # add t to predict noise
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
    def forward(self, z, t_norm):
        #### FIX
        t_embed = torch.sin(2*np.pi*t_norm).unsqueeze(1).to(z.device)
        z_t = torch.cat([z, t_embed], dim=1)
        # predict noise
        return self.model(z_t)

class LDM(nn.module):
    def __init__(self, autoencoder, noise_predictor):
        super(LDM, self).__init__()
        # default vae model
        self.autoencoder = autoencoder
        self.noise_predictor = noise_predictor
    
    def forward(self, x, t):
        # encode input into latent space
        mean, logvar = self.autoencoder.encoder(x)
        noise_pred = self.noise_predictor(mean, t)
        return noise_pred
    
    # Diffusion Process Utilities
    @staticmethod
    def linear_beta_schedule(timesteps):
        beta_start = 1e-4
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    @staticmethod
    def forward_diffusion(z, t, betas):
        sqrt_alphas = torch.sqrt(1 - betas[t])
        sqrt_one_minus_alphas = torch.sqrt(betas[t])
        noise = torch.randn_like(z)
        z_t = sqrt_alphas * z + sqrt_one_minus_alphas * noise
        return z_t, noise

    # Loss Function
    @staticmethod
    def loss_function(noise_pred, noise):
        return nn.functional.mse_loss(noise_pred, noise)