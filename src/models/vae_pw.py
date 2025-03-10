import torch
import torch.nn as nn


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        mean = self.fc2_mean(h)
        logvar = self.fc2_logvar(h)
        return mean, logvar


# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim+2, hidden_dim) # add K and T
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        x_recon = self.sigmoid(self.fc2(h))
        return x_recon


# VAE_pw
class VAE_PW(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE_PW, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, 1)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, k, t):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        z_combined = torch.cat([z, k, t], dim=1)
        x_recon = self.decoder(z_combined)
        return x_recon, mean, logvar


# Loss function
def loss_function(x_recon, x, mean, logvar):
    #### FIX
    BCE = nn.functional.binary_cross_entropy(x_recon, x, reduction="sum") # NEW LOSS FUNCTION NEEDED --> TO DISCUSS
    #### FIX
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD
