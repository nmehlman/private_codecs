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

from disentangle.lightning import SexDisentangleModule
from network.models import VoxProfileAgeSexModel
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

    def __init__(self, codec_name: str = "encodec", device: str = "cuda", dataset_sr: int = 16000):
        super().__init__()
        self.codec_name = codec_name
        self.device = device
        self.dataset_sr = dataset_sr

        # Load speech codec
        codec_class, self.codec_sr = CODECS[codec_name]
        self.codec = codec_class(device=self.device)

        # Load emotion classifier
        self.model = VoxProfileAgeSexModel(device=self.device)

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
        
        x, sex_labs, lengths = batch
        
        if not isinstance(x, torch.Tensor):
            return

        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():

            # Run privitization and map back to audio
            x_hat, _ = pl_module(x)
            codes_private, _ = self.codec.quantize(x_hat)
            audio_private = self.codec.decode(codes_private)

            # Run direct recon without autoencoder
            codes_recon, _ = self.codec.quantize(x)
            audio_codec_only = self.codec.decode(codes_recon)

            # Convert codec-frame lengths to waveform samples for the emotion model
            codec_seq_len = max(x.size(-1), 1)
            codec_step_to_sample = audio_codec_only.shape[-1] / float(codec_seq_len)
            lengths_codec_sr = torch.clamp(
                (lengths.to(dtype=torch.float32) * codec_step_to_sample).round(),
                min=1.0,
            )
            resample_ratio = float(self.dataset_sr) / float(self.codec_sr)
            lengths_waveform = torch.clamp(
                (lengths_codec_sr * resample_ratio).round(),
                min=1.0,
            ).to(dtype=torch.long)

            # Resample audios to dataset sr for emotion model
            audio_private = torchaudio.functional.resample(
                audio_private, orig_freq=self.codec_sr, new_freq=self.dataset_sr
            )
            
            audio_codec_only = torchaudio.functional.resample(
                audio_codec_only, orig_freq=self.codec_sr, new_freq=self.dataset_sr
            )
            
            assert not torch.isnan(audio_private).any(), "NaNs detected in audio_private"
            assert not torch.isnan(audio_codec_only).any(), "NaNs detected in audio_codec_only"

            _, sex_logits_private = self.model(
                    audio_private, sr=self.dataset_sr, return_embeddings=False, 
                    lengths=lengths_waveform
                )
        
            _, sex_logits_codec_only = self.model(
                audio_codec_only, sr=self.dataset_sr, return_embeddings=False,
                lengths=lengths_waveform
            )
        
        assert not torch.isnan(sex_logits_private).any(), "NaNs detected in sex_logits_private"
        assert not torch.isnan(sex_logits_codec_only).any(), "NaNs detected in sex_logits_codec_only"
        
        if was_training:
            pl_module.train()

        sex_probs_private = torch.softmax(sex_logits_private, dim=-1)
        sex_probs_codec_only = torch.softmax(sex_logits_codec_only, dim=-1)

        sex_accuracy_private = (sex_probs_private.argmax(dim=-1) == sex_labs).float().mean()
        sex_accuracy_codec_only = (sex_probs_codec_only.argmax(dim=-1) == sex_labs).float().mean()

        sex_entropy_private = - (sex_probs_private * torch.log(sex_probs_private + 1e-8)).sum(dim=-1).mean()
        sex_entropy_codec_only = - (sex_probs_codec_only * torch.log(sex_probs_codec_only + 1e-8)).sum(dim=-1).mean()

        pl_module.log("epoch_inference/sex_accuracy_private", sex_accuracy_private, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("epoch_inference/sex_accuracy_codec_only", sex_accuracy_codec_only, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("epoch_inference/sex_entropy_private", sex_entropy_private, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("epoch_inference/sex_entropy_codec_only", sex_entropy_codec_only, on_step=False, on_epoch=True, sync_dist=True)

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
    pl_model = SexDisentangleModule(
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
