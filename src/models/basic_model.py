import torch
import torch.nn as nn
import numpy as np
import math

## positional embedding (mentioned by gls)
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even")

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
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU):
        super().__init__()

        if not isinstance(hidden_dims, (list, tuple)):
            hidden_dims = [hidden_dims]
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        self.hidden_layers = nn.ModuleList()
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(last_dim, hidden_dim))
            last_dim = hidden_dim
        
        self.output_layer = nn.Linear(last_dim, output_dim)
        self.activation = activation()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x
    
## 2Dpositional embedding
class PositionalEmbedding2D(nn.Module):
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
        return emb
    
    
## Encoder
class VaeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, activation=nn.ReLU):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        self.mlp = MLP(input_dim, hidden_dims, 2 * latent_dim, activation)

    def forward(self, x):
        latent = self.mlp(x)
        mean = latent[:, :self.latent_dim]
        logvar = latent[:, self.latent_dim:]
        return mean, logvar
    
    
## Decoder
class VaeDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, activation=nn.ReLU):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.mlp = MLP(latent_dim, hidden_dims, output_dim, activation)

    def forward(self, z):
        return self.mlp(z)
    
## MLP for embedding
## A separate MLP for timestep embedding
class EmbeddingMLP(nn.Module):
    def __init__(self, embedding_dim, latent_dim):
        super().__init__()
        self.embed_net = MLP(
            input_dim=embedding_dim, 
            hidden_dims=latent_dim, 
            output_dim=latent_dim, 
            activation=nn.SiLU
        )

    def forward(self, t_emb):
        return self.embed_net(t_emb)  
