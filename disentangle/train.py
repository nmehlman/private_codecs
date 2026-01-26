"""Main training script"""

import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
import yaml
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import os

from disentangle.codec_data import get_dataloaders
from disentangle.lightning import EmotionDisentangleModule

torch.set_warn_always(False)

def load_yaml_config(file_path: str) -> dict:
    """Loads config from yaml file
    Args:
        file_path (str): path to config file

    Returns:
        config (dict): config data
    """
    with open( file_path, 'r' ) as file:
        config = yaml.safe_load(file)

    return config

# Parse command line arguments
parser = argparse.ArgumentParser(description="PyTorch Lightning Training Script")
parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
args = parser.parse_args()

CONFIG_PATH = args.config

if __name__ == "__main__":

    # Load config, and perform general setup
    config = load_yaml_config(CONFIG_PATH)
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
    if config["random_seed"]:
        pl.seed_everything(config["random_seed"], workers=True)

    # Setup dataloaders
    dataloaders = get_dataloaders(
                                dataset_kwargs=config["dataset"],
                                **config["dataloader"]
                                )

    # Create Lightning module
    pl_model = EmotionDisentangleModule(
        **config["lightning"]
    )

    # Create logger (logs are saved to /save_dir/name/version/):
    logger = TensorBoardLogger(**config["tensorboard"])

    # Make trainer
    trainer = Trainer(logger=logger, strategy=DDPStrategy(find_unused_parameters=True), **config["trainer"])

    trainer.fit(
            pl_model,
            train_dataloaders = dataloaders["train"],
            val_dataloaders = dataloaders["val"],
            ckpt_path = config["ckpt_path"]
        )

