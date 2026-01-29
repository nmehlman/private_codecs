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
from disentangle.misc.load_stats import load_dataset_stats

torch.set_warn_always(False)

# Parse command line arguments
parser = argparse.ArgumentParser(description="PyTorch Lightning Training Script")
parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
args = parser.parse_args()


if __name__ == "__main__":

    # Load config, and perform general setup
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
    if config["random_seed"]:
        pl.seed_everything(config["random_seed"], workers=True)

    # Setup dataloaders
    dataset_name = config["dataset_name"]
    codec_name = config["codec_name"]
    input_type = config["input_type"]

    dataset_kwargs = dict(config["dataset"])
    dataset_kwargs.setdefault("input_type", input_type)

    dataloaders = get_dataloaders(
                                dataset_kwargs=dataset_kwargs,
                                **config["dataloader"]
                                )
    
    # Load dataset stats for normalization
    stats = load_dataset_stats(dataset_name, codec_name, input_type)

    # Create Lightning module
    pl_model = EmotionDisentangleModule(
        **config["lightning"],
        dataset_stats=stats
    )

    # Create logger (logs are saved to /save_dir/name/version/):
    logger = TensorBoardLogger(**config["tensorboard"])

    # Save config to tensorboard directory
    config_save_path = os.path.join(logger.log_dir, "config.yaml")
    os.makedirs(logger.log_dir, exist_ok=True)
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Make trainer
    trainer = Trainer(logger=logger, strategy=DDPStrategy(find_unused_parameters=True), **config["trainer"])

    trainer.fit(
            pl_model,
            train_dataloaders = dataloaders["train"],
            val_dataloaders = dataloaders["val"],
            ckpt_path = config["ckpt_path"]
        )
