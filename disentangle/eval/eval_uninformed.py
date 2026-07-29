from disentangle.lightning import SexDisentangleModule
from disentangle.misc.utils import load_dataset_stats
from network.models import VoxProfileAgeSexModel
from data.expresso import ExpressoDataset, EXPRESSO_SR
from data.msp_podcast import MSPPodcastDataset, MSP_SR
from data.vox1 import Vox1Dataset, VOX1_SR
from network.codec import HifiCodec, EnCodec, BigCodec, HIFICODEC_SR, ENCODEC_SR, BIGCODEC_SR

import argparse
import os
import re
from private_codecs.disentangle.eval.run_asr import run_asr_eval
import pytorch_lightning as pl # type: ignore
import yaml  # type: ignore

import tqdm  # type: ignore
import torch  # type: ignore
import torchaudio  # type: ignore
import pickle
import shutil
import json

from disentangle.lightning import compute_difference_metric

def get_stats(tensor):
        return {
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "max": tensor.max().item(),
            "min": tensor.min().item(),
        }


def _sanitize_cache_key(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def _get_audio_cache_path(cache_dir, cache_name, filename):
    return os.path.join(cache_dir, cache_name, f"{_sanitize_cache_key(filename)}.wav")


def _save_cached_audio(cache_path, audio_tensor, sr):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torchaudio.save(cache_path, audio_tensor.detach().cpu(), sample_rate=sr)


def process_sample(sample, codec, pl_model, sex_model, dataset_sr, codec_sr, cache_dir=None, device=None, filename=None):
    
    """Process a single sample."""
    
    audio = sample["audio"].to(device)
    label = sample["gender"]
    filename = sample["filename"] if filename is None else filename
    length = sample["length"]

    raw_audio_cache_path = None
    codec_only_cache_path = None
    private_audio_cache_path = None

    raw_audio = audio
    
    # Get embedding for raw audio
    with torch.no_grad():
        _, sex_logits_raw = sex_model(
            raw_audio, sr=dataset_sr, return_embeddings=False, lengths=torch.tensor([length]).to(device)
        )
    
    # Encode audio with codec
    with torch.no_grad():
        embedding_raw = codec.encode(raw_audio, sr=dataset_sr)
        codes_raw, quantized_embedding_raw = codec.quantize(embedding_raw)
    
    with torch.no_grad():
        embedding_private, _ = pl_model(quantized_embedding_raw)
        codes_private, embedding_private_quantized = codec.quantize(embedding_private)

    with torch.no_grad():
        audio_private = codec.decode(codes_private)
    
    # Codec-only reconstruction (direct decode from quantized codec embedding, no autoencoder)
    with torch.no_grad():
        audio_codec_only = codec.decode(codes_raw)
    
    # Resample audios to dataset sr for sex model
    audio_private = torchaudio.functional.resample(
        audio_private, orig_freq=codec_sr, new_freq=dataset_sr
    )
    
    audio_codec_only = torchaudio.functional.resample(
        audio_codec_only, orig_freq=codec_sr, new_freq=dataset_sr
    )
    
    # Get sex logits for all private audios
    with torch.no_grad():
        _, sex_logits_private = sex_model(
                audio_private, sr=dataset_sr, return_embeddings=False, 
                lengths=torch.tensor([length]).to(device)
            )
       
        _, sex_logits_codec_only = sex_model(
            audio_codec_only, sr=dataset_sr, return_embeddings=False,
            lengths=torch.tensor([length]).to(device)
        )  
        
    # Build results dict
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

    # Save audio to cache if paths are provided
    if cache_dir:
        raw_audio_cache_path = _get_audio_cache_path(cache_dir, "raw_audio", filename)
        codec_only_cache_path = _get_audio_cache_path(cache_dir, "codec_only_audio", filename)
        private_audio_cache_path = _get_audio_cache_path(cache_dir, "private_audio", filename)
        _save_cached_audio(raw_audio_cache_path, raw_audio, sr=dataset_sr)
        _save_cached_audio(private_audio_cache_path, audio_private, sr=dataset_sr)
        _save_cached_audio(codec_only_cache_path, audio_codec_only, sr=dataset_sr)

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
        if 'last.ckpt' in os.listdir(checkpoints_dir):
            return os.path.join(checkpoints_dir, 'last.ckpt')
        else:
            raise FileNotFoundError(
                f"No checkpoints found in {checkpoints_dir}"
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
    
    parser = argparse.ArgumentParser(description="Run eval")
    
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

    stats = load_dataset_stats(dataset_name, codec_name, input_type)
    
    # Load disentanglement model from checkpoint
    ckpt_path = _resolve_checkpoint_path(log_dir, config.get("ckpt_name", None))
    pl_model = SexDisentangleModule.load_from_checkpoint(ckpt_path, dataset_stats=stats, **train_config["lightning"]).to(config["device"]).eval()
    
    # Load VP model (pretrained/fixed)
    sex_model = VoxProfileAgeSexModel(device=config["device"])
    
    # Load speech codec
    codec_class, codec_sr = CODECS[codec_name]
    codec = codec_class(device=config["device"])

    # Maybe load predefined train/val speaker splits from json file and add to dataset kwargs
    train_val_spks_split_file = config["dataset"].pop("train_val_spks_split_file", None)
    if train_val_spks_split_file:
        with open(train_val_spks_split_file, "r") as f:
            train_val_spks = json.load(f)
    else:
        train_val_spks = None
    
    # Load dataset
    dataset_class, dataset_sr = DATASETS[dataset_name]
    dataset = dataset_class(**config["dataset"], speakers=train_val_spks['val'] if train_val_spks else None) 

    cache_dir = config.get("cache_dir", None)
    num_cached_samples = config.get("num_cached_samples", 0)

    if cache_dir: # Ensure cache dir exists and clear its contents (including nested subdirs)
        os.makedirs(cache_dir, exist_ok=True)
        # Walk the directory and remove files/dirs
        for root, dirs, files in os.walk(cache_dir, topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                except Exception:
                    pass
            for name in dirs:
                dirpath = os.path.join(root, name)
                try:
                    shutil.rmtree(dirpath)
                except Exception:
                    pass

    # Process each sample
    for i, sample in tqdm.tqdm(enumerate(dataset), total=len(dataset), desc="Running Eval"):
        
        results = process_sample(
            sample,
            codec,
            pl_model,
            sex_model,
            dataset_sr,
            codec_sr,
            cache_dir=cache_dir if i < num_cached_samples else None,
            device=config["device"],
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
        
        if i <= config["num_samples_to_save"]:  # Save audio only for first N samples
            save_dict["audio_raw"] = results["audio_raw"]
            save_dict["audio_private"] = results["audio_private"]
            save_dict["audio_codec_only"] = results["audio_codec_only"]
        
        save_path = os.path.join(save_root, f"{i}_{results['filename']}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)
            
    if config.get("run_asr_eval", False):
        assert cache_dir is not None, "Cache directory must be specified for ASR evaluation"
        print("Running ASR evaluation")
        asr_results = run_asr_eval(cache_dir, device="cuda")

    
    
    
