"""Main training script"""

import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
import json
import yaml
import shutil
import torch
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
import os
import json
from typing import Union

from disentangle.codec_data import get_dataloaders
from disentangle.misc.utils import load_dataset_stats

from disentangle.lightning import SexDisentangleModule
from disentangle.eval.run_asr import run_asr_eval
from network.models import VoxProfileAgeSexModel
from network.codec import HifiCodec, EnCodec, BigCodec, HIFICODEC_SR, ENCODEC_SR, BIGCODEC_SR
from disentangle.eval.eval_uninformed import process_sample, _resolve_checkpoint_path
from disentangle.misc.parse_results import parse_results
from data.vox1 import Vox1Dataset, VOX1_SR
import tqdm

import pickle
import torchaudio
import pickle

torch.set_warn_always(False)

CODECS = {
    "encodec": (EnCodec, ENCODEC_SR),
    "hificodec": (HifiCodec, HIFICODEC_SR),
    "bigcodec": (BigCodec, BIGCODEC_SR),
}

def run_eval(
        config: dict, 
        log_dir: str, 
        pl_model: SexDisentangleModule, 
        dataset_stats: dict, 
        cache_dir: Union[str, None] = None,
        num_cached_samples: int = 0,
        val_spks: Union[list, None] = None, 
        device='cuda') -> str:

    save_root = os.path.join(log_dir, "eval")
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    else:
        raise ValueError(f"Save path {save_root} already exists!")
    
    if cache_dir: # Ensure cache dir exists and clear its contents (including nested subdirs)
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        os.mkdir(os.path.join(cache_dir, "raw_audio"))
        os.mkdir(os.path.join(cache_dir, "private_audio"))
        os.mkdir(os.path.join(cache_dir, "codec_only_audio"))

    codec_name = config["codec_name"]
    sample_to_save = config.get("sample_to_save", 25)  # Number of samples to save with audio for qualitative analysis
    
    # Load disentanglement model from checkpoint
    ckpt_path = _resolve_checkpoint_path(log_dir, config.get("ckpt_name", None))
    pl_model = SexDisentangleModule.load_from_checkpoint(ckpt_path, dataset_stats=dataset_stats, **config["lightning"]).to(device).eval()
    
    # Load VP model (pretrained/fixed)
    sex_model = VoxProfileAgeSexModel(device=device)
    
    # Load speech codec
    codec_class, codec_sr = CODECS[codec_name]
    codec = codec_class(device=device)

    dataset = Vox1Dataset(**config["audio_eval_dataset"], speakers=val_spks) 
    
    # Process each sample
    for i, sample in tqdm.tqdm(enumerate(dataset), total=len(dataset), desc="Running Eval"):
        
        results = process_sample(
            sample,
            codec,
            pl_model,
            sex_model,
            dataset_sr=VOX1_SR,
            codec_sr=codec_sr,
            cache_dir=cache_dir if i < num_cached_samples else None, # Only cache the first N samples if caching is enabled
            device=device,
            filename=f"{i}_{sample['filename']}"
        )
        
        # Build save dict, optionally excluding audio to save space
        save_dict = { 
            "label": results["label"],
            "sex_logits_raw": results["sex_logits_raw"],
            "sex_logits_private": results["sex_logits_private"],
            "sex_logits_codec_only": results["sex_logits_codec_only"],
            "private_embedding_stats": results["private_embedding_stats"],
            "difference_metrics": results["difference_metrics"],
        }
        
        if i <= sample_to_save:  # Save audio only for first N samples
            save_dict["audio_raw"] = results["audio_raw"]
            save_dict["audio_private"] = results["audio_private"]
            save_dict["audio_codec_only"] = results["audio_codec_only"]
        
        save_path = os.path.join(save_root, f"{i}_{results['filename']}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)

            
    return save_root

class EpochInferenceCallback(Callback):
    """Run inference on one batch after each train epoch and log summary metrics."""

    def __init__(self, n_batches: int = 1, codec_name: str = "encodec", device: str = "cuda", dataset_sr: int = 16000):
        super().__init__()
        self.codec_name = codec_name
        self.device = device
        self.dataset_sr = dataset_sr

        # Load speech codec
        codec_class, self.codec_sr = CODECS[codec_name]
        self.codec = codec_class(device=self.device)
        self.n_batches = n_batches

        # Load classifier
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
        
        was_training = pl_module.training
        pl_module.eval()

        batch_accuracy_private = []
        batch_entropy_private = []
        for i, batch in enumerate(dataloader):
            if i >= self.n_batches:
                break

            batch = self._move_to_device(batch, pl_module.device)
            
            x, sex_labs, _, lengths = batch
            
            if not isinstance(x, torch.Tensor):
                return

            with torch.no_grad():

                # Run privitization and map back to audio
                x_hat, _ = pl_module(x)
                codes_private, _ = self.codec.quantize(x_hat)
                audio_private = self.codec.decode(codes_private)

                # Convert codec-frame lengths to waveform samples for the model
                codec_seq_len = max(x.size(-1), 1)
                codec_step_to_sample = audio_private.shape[-1] / float(codec_seq_len)
                lengths_codec_sr = torch.clamp(
                    (lengths.to(dtype=torch.float32) * codec_step_to_sample).round(),
                    min=1.0,
                )
                resample_ratio = float(self.dataset_sr) / float(self.codec_sr)
                lengths_waveform = torch.clamp(
                    (lengths_codec_sr * resample_ratio).round(),
                    min=1.0,
                ).to(dtype=torch.long)

                # Resample audios to dataset sr for model
                audio_private = torchaudio.functional.resample(
                    audio_private, orig_freq=self.codec_sr, new_freq=self.dataset_sr
                )
                
                assert not torch.isnan(audio_private).any(), "NaNs detected in audio_private"

                _, sex_logits_private = self.model(
                        audio_private, sr=self.dataset_sr, return_embeddings=False, 
                        lengths=lengths_waveform
                    )

            assert not torch.isnan(sex_logits_private).any(), "NaNs detected in sex_logits_private"

            sex_probs_private = torch.softmax(sex_logits_private, dim=-1)

            sex_accuracy_private = (sex_probs_private.argmax(dim=-1) == sex_labs).float().mean()
            sex_entropy_private = - (sex_probs_private * torch.log(sex_probs_private + 1e-8)).sum(dim=-1).mean()
            
            batch_accuracy_private.append(sex_accuracy_private.item())
            batch_entropy_private.append(sex_entropy_private.item())
        
        if was_training:
            pl_module.train()

        pl_module.log("epoch_inference/sex_accuracy_private", np.mean(batch_accuracy_private), on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("epoch_inference/sex_entropy_private", np.mean(batch_entropy_private), on_step=False, on_epoch=True, sync_dist=True)

if __name__ == "__main__":

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PyTorch Lightning Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load config, and perform general setup
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
    pl.seed_everything(config["random_seed"], workers=True)
    torch.random.manual_seed(config["random_seed"])

    # Setup dataloaders
    dataset_name = config["dataset_name"]
    codec_name = config["codec_name"]
    input_type = config["input_type"]

    # Create logger (logs are saved to /save_dir/name/version/):
    logger = TensorBoardLogger(**config["tensorboard"])

    # Save config to tensorboard directory
    log_dir = logger.log_dir
    config_save_path = os.path.join(log_dir, "config.yaml")
    os.makedirs(log_dir, exist_ok=True)
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
        
    # Maybe load predefined train/val speaker splits from json file and add to dataset kwargs
    train_val_spks_split_file = config["dataset"].pop("train_val_spks_split_file", None)
    if train_val_spks_split_file:
        with open(train_val_spks_split_file, "r") as f:
            train_val_spks = json.load(f)
    else:
        train_val_spks = None

    dataloaders = get_dataloaders(
                                train_val_spk=train_val_spks,
                                dataset_kwargs=config["dataset"],
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
    
    epoch_inf_callback = EpochInferenceCallback(
        n_batches=16,
        codec_name=codec_name,
        device="cuda", 
        dataset_sr=config.get("dataset_sr", 16000)
    )
    
    ckpt_callback = ModelCheckpoint(
        monitor="epoch_inference/sex_accuracy_private",
        every_n_epochs=1,
        mode="min",
        filename="best-{epoch}-{val_adv_acc:.3f}",
        save_top_k=1,
        save_last=True,
    )

    callbacks = [ckpt_callback, epoch_inf_callback]

    # Make trainer
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        **config["trainer"],
    )

    trainer.fit(
            pl_model,
            train_dataloaders = dataloaders["train"],
            val_dataloaders = dataloaders["val"],
            ckpt_path = config["ckpt_path"],
        )
    
    print("Training complete. Running final evaluation")
    best_ckpt_path = ckpt_callback.best_model_path

    if best_ckpt_path:
        print(f"Loading best checkpoint: {best_ckpt_path}")
        pl_model = SexDisentangleModule.load_from_checkpoint(
            best_ckpt_path,
            dataset_stats=stats,
            **config["lightning"],
        )    
    
    cache_dir = config.get("cache_dir", None)
    num_cached_samples = config.get("num_cached_samples", 0)
    results_dir = run_eval(config, log_dir, pl_model, stats, val_spks=train_val_spks["val"] if train_val_spks else None, cache_dir=cache_dir, num_cached_samples=num_cached_samples, device="cuda")

    parsed_results = parse_results(results_dir) # Compute average metrics
    
    if config.get("run_asr_eval", False):
        assert cache_dir is not None, "Cache directory must be specified for ASR evaluation"
        print("Running ASR evaluation")
        asr_results = run_asr_eval(cache_dir, device="cuda")
        for key, value in asr_results.items(): # Add to main results file
            parsed_results[key] = value
    
    for key, value in parsed_results.items():
        if value is None:
            print(f"{key}: None")
        else:
            print(f"{key}: {value:.4f}")

    json.dump(parsed_results, open(os.path.join(results_dir, "final_results.json"), "w"), indent=4)
    
    hp_metric = -parsed_results.get("accuracy_private", 0.0)
    trainer.logger.log_hyperparams(
        pl_model.hparams,
        {"hp_metric": hp_metric},
    )
