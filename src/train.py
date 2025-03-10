import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


# inner imports
from src.models import vae
from src.utils.yaml_helper import YamlParser

# Hyperparameters
model_config = YamlParser("src/models/vae.yaml").load_yaml()

name = model_config["model"]["name"]
train_param = model_config["train"]
network_param = model_config["network"]

# other
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = train_param["batch_size"]
epochs = train_param["epochs"]
learning_rate = train_param["learning_rate"]
latent_dim = network_param["latent_dim"]
input_dim = network_param["input_dim"]
hidden_dim = network_param["hidden_dim"]

# create model
model = vae.VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# test dataset -> to our data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, input_dim).to(device)
        optimizer.zero_grad()
        x_recon, mean, logvar = model(data)
        loss = vae.loss_function(x_recon, data, mean, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}")

model.eval()
with torch.no_grad():  # --> sample result
    sample = torch.randn(64, latent_dim).to(device)
    sample = model.decoder(sample).cpu()
    save_image(sample.view(64, 1, 28, 28), "output/sample.png")
