"""Main training script"""

import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import Callback
from pytorch_lightning.strategies.ddp import DDPStrategy
import yaml
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import os

from disentangle.codec_data import get_dataloaders
from disentangle.misc.utils import load_dataset_stats

from disentangle.lightning import EmotionDisentangleModule
from network.models import VoxProfileEmotionModel
from network.codec import HifiCodec, EnCodec, BigCodec, HIFICODEC_SR, ENCODEC_SR, BIGCODEC_SR

import torchaudio

torch.set_warn_always(False)

CODECS = {
    "encodec": (EnCodec, ENCODEC_SR),
    "hificodec": (HifiCodec, HIFICODEC_SR),
    "bigcodec": (BigCodec, BIGCODEC_SR),
}


class EpochInferenceCallback(Callback):
    """Run inference on one batch after each train epoch and log summary metrics."""

    def __init__(self, codec_name: str = "encodec", emotion_model_name: str = "whisper", device: str = "cuda", dataset_sr: int = 16000):
        super().__init__()
        self.codec_name = codec_name
        self.emotion_model_name = emotion_model_name
        self.device = device
        self.dataset_sr = dataset_sr

        # Load speech codec
        codec_class, self.codec_sr = CODECS[codec_name]
        self.codec = codec_class(device=self.device)

        # Load emotion classifier
        self.emotion_model = VoxProfileEmotionModel(device=self.device, split_models=True)

    def _resolve_dataloader(self, trainer):
        val_dataloaders = trainer.val_dataloaders
        if val_dataloaders is None:
            return None
        if isinstance(val_dataloaders, (list, tuple)):
            return val_dataloaders[0] if len(val_dataloaders) > 0 else None
        return val_dataloaders

    def _move_to_device(self, obj, device):
        if torch.is_tensor(obj):
            return obj.to(device, non_blocking=True)
        if isinstance(obj, tuple):
            return tuple(self._move_to_device(v, device) for v in obj)
        if isinstance(obj, list):
            return [self._move_to_device(v, device) for v in obj]
        if isinstance(obj, dict):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}
        return obj

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        dataloader = self._resolve_dataloader(trainer)
        if dataloader is None:
            return

        try:
            batch = next(iter(dataloader))
        except StopIteration:
            return

        batch = self._move_to_device(batch, pl_module.device)
        if not isinstance(batch, (tuple, list)) or len(batch) == 0:
            return
        
        x, _, emotion_labs, lengths = batch
        
        if not isinstance(x, torch.Tensor):
            return

        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():

            # Run privitization and map back to audio
            x_hat, _ = pl_module(x)
            codes_private, embedding_private_quantized = self.codec.quantize(x_hat)
            audio_private = self.codec.decode(codes_private)

            # Run direct recon without autoencoder
            audio_codec_only = self.codec.decode(self.codec.quantize(x)[0])

            # Resample audios to dataset sr for emotion model
            audio_private = torchaudio.functional.resample(
                audio_private, orig_freq=self.codec_sr, new_freq=self.dataset_sr
            ).unsqueeze(0)
            
            audio_codec_only = torchaudio.functional.resample(
                audio_codec_only, orig_freq=self.codec_sr, new_freq=self.dataset_sr
            ).unsqueeze(0)
            
            emotion_logits_private = self.emotion_model(
                    audio_private, sr=self.dataset_sr, return_embeddings=False, 
                    lengths=lengths
                )[f"{self.emotion_model_name}_logits"]
        
            emotion_logits_codec_only = self.emotion_model(
                audio_codec_only, sr=self.dataset_sr, return_embeddings=False,
                lengths=lengths
            )[f"{self.emotion_model_name}_logits"]

        if was_training:
            pl_module.train()

        emotion_probs_private = torch.softmax(emotion_logits_private, dim=-1)
        emotion_probs_codec_only = torch.softmax(emotion_logits_codec_only, dim=-1)

        emotion_accuracy_private = (emotion_probs_private.argmax(dim=-1) == emotion_labs).float().mean()
        emotion_accuracy_codec_only = (emotion_probs_codec_only.argmax(dim=-1) == emotion_labs).float().mean()

        emotion_entropy_private = - (emotion_probs_private * torch.log(emotion_probs_private + 1e-8)).sum(dim=-1).mean()
        emotion_entropy_codec_only = - (emotion_probs_codec_only * torch.log(emotion_probs_codec_only + 1e-8)).sum(dim=-1).mean()

        pl_module.log("epoch_inference/emotion_accuracy_private", emotion_accuracy_private, on_step=False, on_epoch=True, sync_dist=False)
        pl_module.log("epoch_inference/emotion_accuracy_codec_only", emotion_accuracy_codec_only, on_step=False, on_epoch=True, sync_dist=False)
        pl_module.log("epoch_inference/emotion_entropy_private", emotion_entropy_private, on_step=False, on_epoch=True, sync_dist=False)
        pl_module.log("epoch_inference/emotion_entropy_codec_only", emotion_entropy_codec_only, on_step=False, on_epoch=True, sync_dist=False)

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
    assert isinstance(dataloaders, dict), "Expected train/val dataloader dictionary."
    
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

    callbacks = []
    callbacks.append(EpochInferenceCallback(
        codec_name=codec_name,
        emotion_model_name=config["dataset"]["emotion_model"],
        device="cuda",
        dataset_sr=config.get("dataset_sr", 16000)
    ))

    # Make trainer
    trainer = Trainer(
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=callbacks,
        **config["trainer"],
    )

    trainer.fit(
            pl_model,
            train_dataloaders = dataloaders["train"],
            val_dataloaders = dataloaders["val"],
            ckpt_path = config["ckpt_path"]
        )
