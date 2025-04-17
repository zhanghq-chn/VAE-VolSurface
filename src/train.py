import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

# system
import argparse
from dotenv import load_dotenv
load_dotenv()
import sys, os
sys.path.insert(0, os.getenv('SRC_PATH'))

# inner imports
from src.models.vae import VAE
from src.models.ldm import LDM, NoisePredictor
from src.models.vae_pw import VAE_PW
from src.models.vae_pw_improve import VAE_PW_II
from src.utils.yaml_helper import YamlParser
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger(__name__, level="INFO")


class Trainer(object):
    def __init__(self, model_name):
        self.model_name = model_name

        # load config
        self.path = "src/models/" + model_name + ".yaml"
        try:
            self.config = YamlParser(self.path).load_yaml()
        except FileNotFoundError:
            logger.error("Model configs not found")
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

        device_name = "cpu"
        if torch.cuda.is_available():
            device_name = "cuda"
        elif torch.mps.is_available():
            device_name = "mps"
        self.device = torch.device(device_name)
        logger.info(f"Using device: {device_name}")

    #### MODEL
    # set config here only doing hypertune
    def create_model(self, hyper_config=None):
        network_param = hyper_config if hyper_config else self.network_param
        train_param = hyper_config if hyper_config else self.train_param

        if self.model_type.startswith("vae"):
            match self.model_type:
                case "vae":
                    mdl = VAE
                case "vae_pw":
                    mdl = VAE_PW
                case "vae_pw_ii":
                    mdl = VAE_PW_II
                case _:
                    logger.error("Model not found")
                    raise AssertionError("Model not found")
            # check params
            for key in ["input_dim", "hidden_dim", "latent_dim"]:
                if key not in network_param:
                    logger.error(f"Key '{key}' is missing in the network params.")
                    raise AssertionError(
                        f"Key '{key}' is missing in the network params."
                    )

            self.model = mdl(
                network_param["input_dim"],
                network_param["hidden_dim"],
                network_param["latent_dim"],
            ).to(self.device)
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=train_param["learning_rate"]
            )

        elif self.model_type == "ldm":
            for key in ["base", "latent_dim", "hidden_dim", "embedding_dim", "timesteps"]:
                if key not in network_param:
                    logger.error(f"Key '{key}' is missing in the network params.")
                    raise AssertionError(
                        f"Key '{key}' is missing in the network params."
                    )

            # Load base model (VAE here)
            base_dict = YamlParser(
                f"src/models/{network_param['base']}.yaml"
            ).load_yaml()["network"]
            self.base_dict = base_dict
            assert (
                base_dict["latent_dim"] == network_param["latent_dim"]
            ), "Latent dim mismatch"

            self.base_encoder = VAE(
                base_dict["input_dim"],
                base_dict["hidden_dim"],
                base_dict["latent_dim"],
            )
            self.base_encoder.load_state_dict(
                torch.load(f"params/{network_param['base']}.pth")
            )
            
            self.model = LDM(autoencoder=self.base_encoder, config=network_param).to(self.device)
            self.model.set_beta(network_param["timesteps"], self.device)
            #### FIX: optimizer change
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=train_param["learning_rate"]
            )
        # add other model types here
        else:
            logger.error("Model not found")
            return

    #### MODEL TRAIN
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        logger.info(f"Model loaded from {path}")

    def train(self, train_loader: DataLoader, echo=True):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            self.optimizer.zero_grad()

            if self.model_type == "vae_pw_ii":
                pw_grid, pw_vol, surface = data
                pw_grid = pw_grid.view(-1, 2).to(self.device)
                surface = surface.view(-1, self.network_param["input_dim"]).to(self.device)
                pw_vol = pw_vol.view(-1, 1).to(self.device)
                pred, mean, logvar = self.model(surface, pw_grid)
                loss = VAE_PW_II.loss_function(pred, pw_vol, mean, logvar)

            elif self.model_type.startswith("vae"):
                data, _ = data
                data = data.view(-1, self.network_param["input_dim"]).to(self.device)
                x_recon, mean, logvar = self.model(data)
                mdl = VAE if self.model_type == "vae" else VAE_PW
                loss = mdl.loss_function(x_recon, data, mean, logvar)

            elif self.model_type == "ldm":
                data, _ = data
                data = data.view(-1, self.base_dict["input_dim"]).to(self.device)
                # sample a random timestep
                t = torch.randint(
                    0, self.network_param["timesteps"], (data.size(0),)
                ).to(self.device)
                
                # Get loss from LDM
                loss = self.model.get_loss(data, t)

            # Add other model types here
            else:
                print("Model not found")
                return

            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
        if echo:
            logger.info(
                f"Loss: {train_loss / len(train_loader.dataset):.4f}"
            )
        return train_loss

    def evaluate(self, output_path=None):
        self.model.eval()
        with torch.no_grad():
            if self.model_type == "vae" or self.model_type == "ldm":
                sample = torch.randn(64, self.network_param["latent_dim"]).to(self.device)
            elif self.model_type == "vae_pw_ii":
                # todo sampling
                pass
            else:
                sample = torch.randn(64, self.network_param["latent_dim"] + 2).to(self.device)
            sample = self.model.decoder(sample).cpu()
            # save_image(sample.view(64, 1, 28, 28), f"{output_path}/sample.png")

    #### HYPERTUNE
    # create hypertune config from yaml
    @staticmethod
    def make_hypertune_config(raw_config: dict):
        hyper_config = {key: tune.choice(value) for key, value in raw_config.items()}
        return hyper_config

    def hyper_train(self, train_loader: DataLoader, hyper_config: dict):
        if not hyper_config:
            logger.error("Hyperparameter config is not set.")
            raise AssertionError("Hyperparameter config is not set.")
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
                logger.info(f"Epoch {epoch + 1}/{trainer.epochs}, ",end='')
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
