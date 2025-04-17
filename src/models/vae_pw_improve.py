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
        self.fc1 = nn.Linear(latent_dim, hidden_dim)  # add K and T
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        x_recon = self.sigmoid(self.fc2(h))
        return x_recon


# VAE_pw
class VAE_PW_II(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE_PW_II, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim + 2, hidden_dim, 1)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, surface, pw_grid):
        mean, logvar = self.encoder(surface)
        z = self.reparameterize(mean, logvar)
        z_combined = torch.cat([z, pw_grid.view(-1,2)], dim=1)
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
