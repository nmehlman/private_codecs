# TODO swap out prototypes from other cases

from disentangle.lightning import EmotionDisentangleModule
from disentangle.misc.utils import load_dataset_stats, load_emotion_prototypes
from network.models import VoxProfileEmotionModel
from data.expresso import ExpressoDataset, EXPRESSO_SR
from data.msp_podcast import MSPPodcastDataset, MSP_SR
from network.codec import HifiCodec, EnCodec, BigCodec, HIFICODEC_SR, ENCODEC_SR, BIGCODEC_SR
from disentangle.codec_data import EmbeddingDataset

import argparse
import os
import re
import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader

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


def process_sample_exhaustive(sample, codec, pl_model, emotion_model, prototypes, dataset_sr, codec_sr, emotion_conditioning_model, config):
    
    """Process a single sample with exhaustive strategy (all emotion prototypes)."""
    
    audio = sample["audio"].to(config["device"])
    label = sample["emotion"]
    filename = sample["filename"]
    length = sample["length"]
    
    # Get emotion embedding from raw audio
    with torch.no_grad():
        emotion_logits_raw, emotion_embedding = emotion_model(
            audio, sr=dataset_sr, return_embeddings=True, lengths=torch.tensor([length]).to(config["device"])
        )
    
    # Encode audio with codec
    with torch.no_grad():
        embedding = codec.encode(audio, sr=dataset_sr)
        codes_raw, quantized_embedding = codec.quantize(embedding)
    
    # Generate private audio for all emotion prototypes (exhaustive strategy)
    with torch.no_grad():
        audio_private = {}
        embeddings_private = {}
        for emotion in prototypes:
            emotion_embedding_proto = prototypes[emotion].to(quantized_embedding.device).unsqueeze(0)
            embedding_private, _ = pl_model(quantized_embedding, emotion_embedding_proto)
            codes_private, _ = codec.quantize(embedding_private)
            audio_private[emotion] = codec.decode(codes_private)
            embeddings_private[emotion] = embedding_private
    
    # Run self-reconstruction for reference
    with torch.no_grad():
        embedding_self_recon, _ = pl_model(quantized_embedding, emotion_embedding[f"{emotion_conditioning_model}_embedding"])
        codes_self_recon, _ = codec.quantize(embedding_self_recon)
        audio_self_recon = codec.decode(codes_self_recon)
    
    # Codec-only reconstruction (direct decode from quantized codec embedding, no autoencoder)
    with torch.no_grad():
        audio_codec_only = codec.decode(codes_raw)
    
    # Resample audios to dataset sr for emotion model
    for emotion in audio_private:
        audio_private[emotion] = torchaudio.functional.resample(
            audio_private[emotion], orig_freq=codec_sr, new_freq=dataset_sr
        ).unsqueeze(0)
    
    audio_self_recon = torchaudio.functional.resample(
        audio_self_recon, orig_freq=codec_sr, new_freq=dataset_sr
    ).unsqueeze(0)
    
    audio_codec_only = torchaudio.functional.resample(
        audio_codec_only, orig_freq=codec_sr, new_freq=dataset_sr
    ).unsqueeze(0)
    
    # Get emotion logits for all private audios
    with torch.no_grad():
        emotion_logits_private = {}
        for emotion in audio_private:
            emotion_logits_private[emotion] = emotion_model(
                audio_private[emotion], sr=dataset_sr, return_embeddings=False, 
                lengths=torch.tensor([length]).to(config["device"])
            )
        
        emotion_logits_self_recon = emotion_model(
            audio_self_recon, sr=dataset_sr, return_embeddings=False,
            lengths=torch.tensor([length]).to(config["device"])
        )
        
        emotion_logits_codec_only = emotion_model(
            audio_codec_only, sr=dataset_sr, return_embeddings=False,
            lengths=torch.tensor([length]).to(config["device"])
        )    

    # Run conditioning ablation
    conditioning_ablation_results = compute_conditioning_ablation(
        pl_model, quantized_embedding, emotion_embedding[f"{emotion_conditioning_model}_embedding"]
    )
    
    # Build results dict - note that emotion_logits_private is a dict of logits
    results = {
        "filename": filename,
        "label": label,
        "whisper_emotion_logits_raw": emotion_logits_raw["whisper_logits"].cpu().squeeze(),
        "wavlm_emotion_logits_raw": emotion_logits_raw["wavlm_logits"].cpu().squeeze(),
        "whisper_emotion_logits_self_recon": emotion_logits_self_recon["whisper_logits"].cpu().squeeze(),
        "wavlm_emotion_logits_self_recon": emotion_logits_self_recon["wavlm_logits"].cpu().squeeze(),
        "whisper_emotion_logits_codec_only": emotion_logits_codec_only["whisper_logits"].cpu().squeeze(),
        "wavlm_emotion_logits_codec_only": emotion_logits_codec_only["wavlm_logits"].cpu().squeeze(),
        "whisper_emotion_logits_private": {emotion: logits["whisper_logits"].cpu().squeeze() for emotion, logits in emotion_logits_private.items()},
        "wavlm_emotion_logits_private": {emotion: logits["wavlm_logits"].cpu().squeeze() for emotion, logits in emotion_logits_private.items()},
        "raw_embedding_stats": get_stats(quantized_embedding),
        "self_recon_embedding_stats": get_stats(embedding_self_recon),
        "private_embedding_stats": {emotion: get_stats(embedding_private) for emotion, embedding_private in embeddings_private.items()},
        "audio_raw": audio.cpu().squeeze(),
        "audio_private": {emotion: audio.cpu().squeeze() for emotion, audio in audio_private.items()},  # dict of emotion -> audio
        "audio_self_recon": audio_self_recon.cpu().squeeze(),
        "audio_codec_only": audio_codec_only.cpu().squeeze(),
        "difference_metrics": {emotion: compute_difference_metric(embedding_self_recon, embedding_private) for emotion, embedding_private in embeddings_private.items()},
        "conditioning_ablation": conditioning_ablation_results
    }
    
    return results

def process_sample_targeted(sample, codec, pl_model, emotion_model, prototypes, dataset_sr, codec_sr, emotion_conditioning_model, config):
    """Process a single sample with a specific target emotion prototype."""
    audio = sample["audio"].to(config["device"])
    label = sample["emotion"]
    filename = sample["filename"]
    length = sample["length"]
    target_emotion = config["target_emotion"]
    
    # Get emotion embedding from raw audio
    with torch.no_grad():
        emotion_logits_raw, emotion_embedding = emotion_model(
            audio, sr=dataset_sr, return_embeddings=True, lengths=torch.tensor([length]).to(config["device"])
        )
    
    # Encode audio with codec
    with torch.no_grad():
        embedding = codec.encode(audio, sr=dataset_sr)
        codes_raw, quantized_embedding = codec.quantize(embedding)
    
    # Generate private audio with target emotion prototype only
    with torch.no_grad():
        emotion_embedding_proto = prototypes[target_emotion].to(quantized_embedding.device).unsqueeze(0)
        embedding_private, _ = pl_model(quantized_embedding, emotion_embedding_proto)
        codes_private, _ = codec.quantize(embedding_private)
        audio_private = codec.decode(codes_private)
    
    # Run self-reconstruction for reference
    with torch.no_grad():
        embedding_self_recon, _ = pl_model(quantized_embedding, emotion_embedding[f"{emotion_conditioning_model}_embedding"])
        codes_self_recon, _ = codec.quantize(embedding_self_recon)
        audio_self_recon = codec.decode(codes_self_recon)
    
    # Codec-only reconstruction (direct decode from quantized codec embedding, no autoencoder)
    with torch.no_grad():
        audio_codec_only = codec.decode(codes_raw)
    
    # Resample audios to dataset sr for emotion model
    audio_private = torchaudio.functional.resample(
        audio_private, orig_freq=codec_sr, new_freq=dataset_sr
    ).unsqueeze(0)
    
    audio_self_recon = torchaudio.functional.resample(
        audio_self_recon, orig_freq=codec_sr, new_freq=dataset_sr
    ).unsqueeze(0)
    
    audio_codec_only = torchaudio.functional.resample(
        audio_codec_only, orig_freq=codec_sr, new_freq=dataset_sr
    ).unsqueeze(0)
    
    # Get emotion logits for private audio
    with torch.no_grad():
        emotion_logits_private = emotion_model(
            audio_private, sr=dataset_sr, return_embeddings=False, 
            lengths=torch.tensor([length]).to(config["device"])
        )
        
        emotion_logits_self_recon = emotion_model(
            audio_self_recon, sr=dataset_sr, return_embeddings=False,
            lengths=torch.tensor([length]).to(config["device"])
        )
        
        emotion_logits_codec_only = emotion_model(
            audio_codec_only, sr=dataset_sr, return_embeddings=False,
            lengths=torch.tensor([length]).to(config["device"])
        )    

    # Build results dict
    results = {
        "filename": filename,
        "label": label,
        "target_emotion": target_emotion,
        "whisper_emotion_logits_raw": emotion_logits_raw["whisper_logits"].cpu().squeeze(),
        "wavlm_emotion_logits_raw": emotion_logits_raw["wavlm_logits"].cpu().squeeze(),
        "whisper_emotion_logits_self_recon": emotion_logits_self_recon["whisper_logits"].cpu().squeeze(),
        "wavlm_emotion_logits_self_recon": emotion_logits_self_recon["wavlm_logits"].cpu().squeeze(),
        "whisper_emotion_logits_codec_only": emotion_logits_codec_only["whisper_logits"].cpu().squeeze(),
        "wavlm_emotion_logits_codec_only": emotion_logits_codec_only["wavlm_logits"].cpu().squeeze(),
        "whisper_emotion_logits_private": emotion_logits_private["whisper_logits"].cpu().squeeze(),
        "wavlm_emotion_logits_private": emotion_logits_private["wavlm_logits"].cpu().squeeze(),
        "raw_embedding_stats": get_stats(quantized_embedding),
        "self_recon_embedding_stats": get_stats(embedding_self_recon),
        "private_embedding_stats": get_stats(embedding_private),
        "audio_raw": audio.cpu().squeeze(),
        "audio_private": audio_private.cpu().squeeze(),
        "audio_self_recon": audio_self_recon.cpu().squeeze(),
        "audio_codec_only": audio_codec_only.cpu().squeeze(),
        "difference_metrics": compute_difference_metric(embedding_self_recon, embedding_private)
    }
    
    return results

def process_sample_random(sample, codec, pl_model, emotion_model, prototypes, dataset_sr, codec_sr, emotion_conditioning_model, config):
    """Process a single sample with a randomly selected target emotion (different from true emotion)."""
    
    audio = sample["audio"].to(config["device"])
    label = sample["emotion"]
    filename = sample["filename"]
    length = sample["length"]
    
    # Get emotion name from label int
    emotion_names = list(prototypes.keys())
    label_emotion = emotion_names[label]
    
    # Select random target emotion different from true emotion
    available_emotions = [e for e in prototypes.keys() if e != label_emotion]
    target_emotion = random.choice(available_emotions)
    
    # Get emotion embedding from raw audio
    with torch.no_grad():
        emotion_logits_raw, emotion_embedding = emotion_model(
            audio, sr=dataset_sr, return_embeddings=True, lengths=torch.tensor([length]).to(config["device"])
        )
    
    # Encode audio with codec
    with torch.no_grad():
        embedding = codec.encode(audio, sr=dataset_sr)
        codes_raw, quantized_embedding = codec.quantize(embedding)
    
    # Generate private audio with target emotion prototype only
    with torch.no_grad():
        emotion_embedding_proto = prototypes[target_emotion].to(quantized_embedding.device).unsqueeze(0)
        embedding_private, _ = pl_model(quantized_embedding, emotion_embedding_proto)
        codes_private, _ = codec.quantize(embedding_private)
        audio_private = codec.decode(codes_private)
    
    # Run self-reconstruction for reference
    with torch.no_grad():
        embedding_self_recon, _ = pl_model(quantized_embedding, emotion_embedding[f"{emotion_conditioning_model}_embedding"])
        codes_self_recon, _ = codec.quantize(embedding_self_recon)
        audio_self_recon = codec.decode(codes_self_recon)
    
    # Codec-only reconstruction (direct decode from quantized codec embedding, no autoencoder)
    with torch.no_grad():
        audio_codec_only = codec.decode(codes_raw)
    
    # Resample audios to dataset sr for emotion model
    audio_private = torchaudio.functional.resample(
        audio_private, orig_freq=codec_sr, new_freq=dataset_sr
    ).unsqueeze(0)
    
    audio_self_recon = torchaudio.functional.resample(
        audio_self_recon, orig_freq=codec_sr, new_freq=dataset_sr
    ).unsqueeze(0)
    
    audio_codec_only = torchaudio.functional.resample(
        audio_codec_only, orig_freq=codec_sr, new_freq=dataset_sr
    ).unsqueeze(0)
    
    # Get emotion logits for private audio
    with torch.no_grad():
        emotion_logits_private = emotion_model(
            audio_private, sr=dataset_sr, return_embeddings=False, 
            lengths=torch.tensor([length]).to(config["device"])
        )
        
        emotion_logits_self_recon = emotion_model(
            audio_self_recon, sr=dataset_sr, return_embeddings=False,
            lengths=torch.tensor([length]).to(config["device"])
        )
        
        emotion_logits_codec_only = emotion_model(
            audio_codec_only, sr=dataset_sr, return_embeddings=False,
            lengths=torch.tensor([length]).to(config["device"])
        )    
    
    # Build results dict
    results = {
        "filename": filename,
        "label": label,
        "target_emotion": target_emotion,
        "whisper_emotion_logits_raw": emotion_logits_raw["whisper_logits"].cpu().squeeze(),
        "wavlm_emotion_logits_raw": emotion_logits_raw["wavlm_logits"].cpu().squeeze(),
        "whisper_emotion_logits_self_recon": emotion_logits_self_recon["whisper_logits"].cpu().squeeze(),
        "wavlm_emotion_logits_self_recon": emotion_logits_self_recon["wavlm_logits"].cpu().squeeze(),
        "whisper_emotion_logits_codec_only": emotion_logits_codec_only["whisper_logits"].cpu().squeeze(),
        "wavlm_emotion_logits_codec_only": emotion_logits_codec_only["wavlm_logits"].cpu().squeeze(),
        "whisper_emotion_logits_private": emotion_logits_private["whisper_logits"].cpu().squeeze(),
        "wavlm_emotion_logits_private": emotion_logits_private["wavlm_logits"].cpu().squeeze(),
        "raw_embedding_stats": get_stats(quantized_embedding),
        "self_recon_embedding_stats": get_stats(embedding_self_recon),
        "private_embedding_stats": get_stats(embedding_private),
        "audio_raw": audio.cpu().squeeze(),
        "audio_private": audio_private.cpu().squeeze(),
        "audio_self_recon": audio_self_recon.cpu().squeeze(),
        "audio_codec_only": audio_codec_only.cpu().squeeze(),
        "difference_metrics": compute_difference_metric(embedding_self_recon, embedding_private)
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
    strategy = config["strategy"]
    emotion_conditioning_model = config["emotion_conditioning_model"]
    
    if strategy not in ["exhaustive", "targeted", "random"]:
        raise ValueError(f"Strategy {strategy} not recognized.")

    stats = load_dataset_stats(dataset_name, codec_name, input_type)
    prototypes = load_emotion_prototypes(dataset_name, "train", emotion_conditioning_model, mode=config.get("prototype_mode", "average"))
    
    # Load emotion disentanglement model from checkpoint
    ckpt_path = _resolve_checkpoint_path(log_dir, config.get("ckpt_name", None))
    pl_model = EmotionDisentangleModule.load_from_checkpoint(ckpt_path, dataset_stats=stats, **train_config["lightning"]).to(config["device"]).eval()
    
    # Load VP emotion model (pretrained/fixed)
    emotion_model = VoxProfileEmotionModel(device=config["device"], split_models=True)
    
    # Load speech codec
    codec_class, codec_sr = CODECS[codec_name]
    codec = codec_class(device=config["device"])
    
    # Load dataset and create dataloader
    dataset = EmbeddingDataset(
        dataset_path="/project2/shrikann_35/DATA/expresso/codec_feats/",
        split="dev"
    )
    
    batch_size = 8
    
    x = [dataset[i][0] for i in range(batch_size)]
    emotion_emb = [dataset[i][1] for i in range(batch_size)]
    
    x = torch.stack(x, dim=0).to(config["device"])
    emotion_emb = torch.stack(emotion_emb, dim=0).to(config["device"])
    
    x_hat_self_recon, _ = pl_model(x, emotion_emb)
    
    _ones = torch.ones_like(emotion_emb)
    emotion_emb_shuffled = torch.stack( [torch.remainder(emotion_emb + _ones * i, 9 * _ones) for i in range(1,9)], dim=0).to(x.device) # Permute emotion embeddings
    
    recon_diffs = []
    for emotion_emb_perm in emotion_emb_shuffled:
        x_hat_perm, _ = pl_model(x, emotion_emb_perm)
        recon_diffs.append(compute_difference_metric(x_hat_self_recon, x_hat_perm))
        
    mean_diff = torch.mean(torch.tensor(recon_diffs)).to(config["device"])

    print(f"Mean difference between self-reconstruction and permuted emotion reconstructions: {mean_diff.item()}")

    
    
    
