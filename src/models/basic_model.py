import torch
import torch.nn as nn
import numpy as np
import math

## positional embedding (mentioned by gls)
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        """
        timesteps: Tensor of shape [batch_size] or scalar, dtype int or float
        Returns: Tensor of shape [batch_size, embedding_dim]
        """
        half_dim = self.embedding_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, dtype=torch.float32) / half_dim).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb  # shape: [batch_size, embedding_dim]
    
    
    
## Encoder
class VaeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VaeEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        mean = self.fc2_mean(h)
        logvar = self.fc2_logvar(h)
        return mean, logvar
    
    
## Decoder
class VaeDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(VaeDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        x_recon = self.sigmoid(self.fc2(h))
        return x_recon
    
## MLP for embedding
## A separate MLP for timestep embedding
class EmbeddingMLP(nn.Module):
    def __init__(self, embedding_dim, latent_dim):
        super().__init__()
        self.embed_net = nn.Sequential(
            nn.Linear(embedding_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, t_emb):
        return self.net(t_emb)  
