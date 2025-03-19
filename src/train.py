import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

import argparse


# inner imports
from src.models.vae import VAE
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

        self.hypertune_param = self.config["hypertune"]

        # other
        self.model_type = self.config["model"]["type"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #### MODEL
    # set config here only doing hypertune
    def create_model(self, hyper_config=None):
        network_param = hyper_config if hyper_config else self.network_param
        train_param = hyper_config if hyper_config else self.train_param
        if self.model_type == "vae":
            # check params
            for key in ["input_dim", "hidden_dim", "latent_dim"]:
                assert (
                    key in network_param
                ), f"Key '{key}' is missing in the network params."

            self.model = VAE(
                network_param["input_dim"],
                network_param["hidden_dim"],
                network_param["latent_dim"],
            ).to(self.device)
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=train_param["learning_rate"]
            )

        # add other model types here
        else:
            print("Model not found")
            return

    #### MODEL TRAIN
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

    def train(self, train_loader: DataLoader, echo=True):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, self.network_param["input_dim"]).to(self.device)
            self.optimizer.zero_grad()

            if self.model_type == "vae":
                x_recon, mean, logvar = self.model(data)
                loss = VAE.loss_function(x_recon, data, mean, logvar)

            # Add other model types here
            else:
                print("Model not found")
                return

            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
        if echo:
            print(
                f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}"
            )
        return train_loss

    def evaluate(self, output_path):
        self.model.eval()
        with torch.no_grad():
            sample = torch.randn(64, self.network_param["latent_dim"]).to(self.device)
            sample = self.model.decoder(sample).cpu()
            save_image(sample.view(64, 1, 28, 28), f"{output_path}/sample.png")

    #### HYPERTUNE
    # create hypertune config from yaml
    @staticmethod
    def make_hypertune_config(raw_config: dict):
        hyper_config = {key: tune.choice(value) for key, value in raw_config.items()}
        return hyper_config

    def hyper_train(self, train_loader: DataLoader, hyper_config: dict):
        assert hyper_config, "Hyperparameter config is not set."
        self.create_model(hyper_config)
        for epoch in range(self.epochs):
            ###### FIX: logic change for different dataset (eg:batchsize should be included) ######
            loss = self.train(train_loader)
            tune.report(loss=loss)

    def hypertune(self, train_loader: DataLoader):
        reporter = CLIReporter(metric_columns=["loss", "training_iteration"])
        scheduler = ASHAScheduler(metric="loss", mode="min")

        # Run the grid search
        analysis = tune.run(
            lambda config: self.hyper_train(train_loader, config),
            config=self.make_hypertune_config(self.hypertune_param),
            resources_per_trial={
                "cpu": 2
            },  # FIX: Allocate resources ---> shoule be available in yaml
            num_samples=1,  # Number of samples per configuration
            scheduler=scheduler,
            progress_reporter=reporter,
            verbose=1,
        )

        print("Best config:", analysis.get_best_config(metric="loss", mode="min"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vae_v1", help="Model name")
    ## Train
    parser.add_argument("--train", type=bool, default=False, help="Train model")
    parser.add_argument("--load", type=bool, default=False, help="Load model")
    parser.add_argument("--save", type=bool, default=False, help="Save model")

    ## Hyperparameter tuning
    parser.add_argument(
        "--hypertune", type=bool, default=False, help="Hyperparameter tuning"
    )

    args = parser.parse_args()

    trainer = Trainer(args.model)

    if args.train:
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
                torch.save(
                    trainer.model.state_dict(), f"params/{trainer.model_name}.pth"
                )

            print("Training complete.")

        # evaluate
        trainer.evaluate("output")

    if args.hypertune:
        # create dataset
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(
            root="./data", train=True, transform=transform, download=True
        )
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=trainer.batch_size, shuffle=True
        )

        # hypertune
        trainer.hypertune(train_loader)
