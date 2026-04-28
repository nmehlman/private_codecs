# TODO swap out prototypes from other cases

from disentangle.lightning import SexDisentangleModule
from disentangle.misc.utils import load_dataset_stats, load_emotion_prototypes
from network.models import VoxProfileAgeSexModel
from data.expresso import ExpressoDataset, EXPRESSO_SR
from data.msp_podcast import MSPPodcastDataset, MSP_SR
from data.vox1 import Vox1Dataset, VOX1_SR
from network.codec import HifiCodec, EnCodec, BigCodec, HIFICODEC_SR, ENCODEC_SR, BIGCODEC_SR

import argparse
import os
import re
import pytorch_lightning as pl
import yaml

import tqdm
import torch
import torchaudio
import pickle
import random

from disentangle.lightning import compute_difference_metric
from disentangle.eval.conditioning_ablation import compute_conditioning_ablation

def get_stats(tensor):
        return {
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "max": tensor.max().item(),
            "min": tensor.min().item(),
        }


def process_sample(sample, codec, pl_model, sex_model, dataset_sr, codec_sr, config):
    
    """Process a single sample with exhaustive strategy (all emotion prototypes)."""
    
    audio = sample["audio"].to(config["device"])
    label = sample["emotion"]
    filename = sample["filename"]
    length = sample["length"]
    
    # Get emotion embedding for raw audio
    with torch.no_grad():
        _, sex_logits_raw = sex_model(
            audio, sr=dataset_sr, return_embeddings=True, lengths=torch.tensor([length]).to(config["device"])
        )
    
    # Encode audio with codec
    with torch.no_grad():
        embedding_raw = codec.encode(audio, sr=dataset_sr)
        codes_raw, quantized_embedding_raw = codec.quantize(embedding_raw)
    
    with torch.no_grad():
        embedding_private, _ = pl_model(quantized_embedding_raw)
        codes_private, embedding_private_quantized = codec.quantize(embedding_private)
        audio_private = codec.decode(codes_private)
    
    # Codec-only reconstruction (direct decode from quantized codec embedding, no autoencoder)
    with torch.no_grad():
        audio_codec_only = codec.decode(codes_raw)
    
    # Resample audios to dataset sr for emotion model
    audio_private = torchaudio.functional.resample(
        audio_private, orig_freq=codec_sr, new_freq=dataset_sr
    ).unsqueeze(0)
    
    audio_codec_only = torchaudio.functional.resample(
        audio_codec_only, orig_freq=codec_sr, new_freq=dataset_sr
    ).unsqueeze(0)
    
    # Get emotion logits for all private audios
    with torch.no_grad():
        _, sex_logits_private = sex_model(
                audio_private, sr=dataset_sr, return_embeddings=False, 
                lengths=torch.tensor([length]).to(config["device"])
            )
       
        _, sex_logits_codec_only = sex_model(
            audio_codec_only, sr=dataset_sr, return_embeddings=False,
            lengths=torch.tensor([length]).to(config["device"])
        )    
    
    # Build results dict - note that emotion_logits_private is a dict of logits
    results = {
        "filename": filename,
        "label": label,
        "sex_logits_raw": sex_logits_raw.cpu().squeeze(),
        "sex_logits_private": sex_logits_private.cpu().squeeze(),
        "sex_logits_codec_only": sex_logits_codec_only.cpu().squeeze(),
        "raw_embedding_stats": get_stats(quantized_embedding_raw),
        "private_embedding_stats": get_stats(embedding_private_quantized),
        "audio_raw": audio.cpu().squeeze(),
        "audio_private": audio_private.cpu().squeeze(),
        "audio_codec_only": audio_codec_only.cpu().squeeze(),
        "difference_metrics": compute_difference_metric(quantized_embedding_raw, embedding_private_quantized),
    }
    
    return results


def _resolve_checkpoint_path(log_dir, ckpt_name):
    if ckpt_name:
        return os.path.join(log_dir, "checkpoints", ckpt_name)

    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    ckpt_pattern = re.compile(r"epoch=(\d+)-step=(\d+)\.ckpt$")
    candidates = []
    for filename in os.listdir(checkpoints_dir):
        match = ckpt_pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            step = int(match.group(2))
            candidates.append((epoch, step, filename))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints matching 'epoch=<int>-step=<int>.ckpt' in {checkpoints_dir}"
        )

    _, _, latest_filename = max(candidates, key=lambda item: (item[0], item[1]))
    return os.path.join(checkpoints_dir, latest_filename)


CODECS = {
    "encodec": (EnCodec, ENCODEC_SR),
    "hificodec": (HifiCodec, HIFICODEC_SR),
    "bigcodec": (BigCodec, BIGCODEC_SR),
}

DATASETS = {
    "expresso": (ExpressoDataset, EXPRESSO_SR), 
    "msp_podcast": (MSPPodcastDataset, MSP_SR),
    "vox1": (Vox1Dataset, VOX1_SR),
}

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Export codec embeddings.")
    
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
        
    log_dir = config["log_dir"]
    save_root = os.path.join(log_dir, "eval")
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    else:
        raise ValueError(f"Save path {save_root} already exists!")
    
    train_config_path = os.path.join(log_dir, "config.yaml")
    with open(train_config_path, "r") as f:
        train_config = yaml.safe_load(f)
        
    # Save config to save root
    with open(os.path.join(save_root, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    dataset_name = config["dataset_name"]
    codec_name = config["codec_name"]
    input_type = config["input_type"]
    emotion_conditioning_model = config["emotion_conditioning_model"]

    stats = load_dataset_stats("expresso", codec_name, input_type) # DEBUG
    prototypes = load_emotion_prototypes(dataset_name, "train", emotion_conditioning_model, mode=config.get("prototype_mode", "average"))
    
    # Load emotion disentanglement model from checkpoint
    ckpt_path = _resolve_checkpoint_path(log_dir, config.get("ckpt_name", None))
    pl_model = SexDisentangleModule.load_from_checkpoint(ckpt_path, dataset_stats=stats, **train_config["lightning"]).to(config["device"]).eval()
    
    # Load VP emotion model (pretrained/fixed)
    sex_model = VoxProfileAgeSexModel(device=config["device"], split_models=True)
    
    # Load speech codec
    codec_class, codec_sr = CODECS[codec_name]
    codec = codec_class(device=config["device"])
    
    # Load dataset and create dataloader
    dataset_class, dataset_sr = DATASETS[dataset_name]
    dataset = dataset_class(**config["dataset"], split="dev") # CHANGEME to test when ready
    
    # Process each sample
    for i, sample in tqdm.tqdm(enumerate(dataset), total=len(dataset), desc="Running Eval"):
        
        results = process_sample(sample, codec, pl_model, sex_model, dataset_sr, codec_sr, config)
        
        # Build save dict, optionally excluding audio to save space
        save_dict = {
            "filename": results["filename"],
            "label": results["label"],
            "sex_logits_raw": results["sex_logits_raw"],
            "sex_logits_private": results["sex_logits_private"],
            "sex_logits_codec_only": results["sex_logits_codec_only"],
            "private_embedding_stats": results["private_embedding_stats"],
            "difference_metrics": results["difference_metrics"],
        }
        
        if i <= config["num_samples_to_save"]:  # Save audio only for first N samples
            save_dict["audio_raw"] = results["audio_raw"]
            save_dict["audio_private"] = results["audio_private"]
            save_dict["audio_codec_only"] = results["audio_codec_only"]
        
        save_path = os.path.join(save_root, f"{i}_{results['filename']}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)

    
    
    
