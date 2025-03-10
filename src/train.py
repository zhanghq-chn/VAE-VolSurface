import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

import argparse


# inner imports
from src.models import vae
from src.utils.yaml_helper import YamlParser


class Trainer(object):
    def __init__(self, model_name):
        self.model_name = model_name

        # load config
        self.path = "src/models/" + model_name + ".yaml"
        try:
            self.config = YamlParser(self.path).load_yaml()
        except FileNotFoundError:
            print("Model configs not found")
            return

        # hyperparameters
        self.train_param = self.config["train"]
        self.batch_size = self.train_param["batch_size"]
        self.epochs = self.train_param["epochs"]
        self.learning_rate = self.train_param["learning_rate"]

        self.network_param = self.config["network"]

        # other
        self.model_type = self.config["model"]["type"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_model(self):
        if self.model_type == "vae":
            # check params
            for key in ["input_dim", "hidden_dim", "latent_dim"]:
                assert (
                    key in self.network_param
                ), f"Key '{key}' is missing in the network params."

            self.model = vae.VAE(
                self.network_param["input_dim"],
                self.network_param["hidden_dim"],
                self.network_param["latent_dim"],
            ).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        else:
            print("Model not found")
            return

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

    def train(self, train_loader: DataLoader):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, self.network_param["input_dim"]).to(self.device)
            self.optimizer.zero_grad()
            if self.model_type == "vae":
                x_recon, mean, logvar = self.model(data)
                loss = vae.loss_function(x_recon, data, mean, logvar)
            else:
                print("Model not found")
                return
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}")

    def evaluate(self, output_path):
        self.model.eval()
        with torch.no_grad():
            sample = torch.randn(64, self.network_param["latent_dim"]).to(self.device)
            sample = self.model.decoder(sample).cpu()
            save_image(sample.view(64, 1, 28, 28), f"{output_path}/sample.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vae_v1", help="Model name")
    parser.add_argument("--load", type=bool, default=False, help="Load model")
    parser.add_argument("--save", type=bool, default=False, help="Save model")

    args = parser.parse_args()

    trainer = Trainer(args.model)
    trainer.create_model()

    if args.load:
        trainer.load_model(f"params/{trainer.model_name}.pth")
    else:
        print("Training from scratch")

        # create dataset
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(
            root="./data", train=True, transform=transform, download=True
        )
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=trainer.batch_size, shuffle=True
        )

        # train
        for epoch in range(trainer.epochs):
            trainer.train(train_loader)

        if args.save:
            torch.save(trainer.model.state_dict(), f"params/{trainer.model_name}.pth")

        print("Training complete.")

    # evaluate
    trainer.evaluate("output")
