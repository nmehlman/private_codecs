from embed_predictor.data_loading import EmbeddingDataset
from embed_predictor.embed_classifier import EmbeddingClassifier
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import os
import argparse
import yaml


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run training for embedding predictor model.")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )
    
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
    if config["random_seed"]:
        pl.seed_everything(config["random_seed"], workers=True)

    # Create datasets and dataloaders
    val_frac = config["dataset"].pop("val_frac")
    dataset = EmbeddingDataset(**config["dataset"])
    train_dataset, val_dataset = random_split(dataset,[1.0 - val_frac, val_frac])
    dataloaders = {
        "train": DataLoader(train_dataset, **config["dataloader"], shuffle=True, collate_fn=dataset.collate_function),
        "val": DataLoader(val_dataset, **config["dataloader"], shuffle=False, collate_fn=dataset.collate_function)
    }

    # Create Lightning module
    pl_model = EmbeddingClassifier(**config["model"])

    # Create logger:
    logger = TensorBoardLogger(**config["tensorboard"])

    # Make trainer
    trainer = Trainer(logger=logger, **config["trainer"])

    trainer.fit(
            pl_model,
            train_dataloaders = dataloaders["train"],
            val_dataloaders = dataloaders["val"],
            ckpt_path = config["ckpt_path"]
        )

