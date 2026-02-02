# TODO embedding support

from disentangle.lightning import EmotionDisentangleModule
from disentangle.misc.utils import load_dataset_stats
from network.models import VoxProfileEmotionModel
from data.expresso import ExpressoDataset, EXPRESSO_SR
from data.msp_podcast import MSPPodcastDataset, MSP_SR
from network.codec import HifiCodec, EnCodec, BigCodec, HIFICODEC_SR, ENCODEC_SR, BIGCODEC_SR

import argparse
import os
import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader

import tqdm
import torch
import torchaudio
import pickle


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
        
    # Save config to save root
    with open(os.path.join(save_root, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    dataset_name = config["dataset_name"]
    codec_name = config["codec_name"]
    input_type = config["input_type"]
    strategy = config["strategy"]

    stats = load_dataset_stats(dataset_name, codec_name, input_type)
    
    # Load emotion disentanglement model from checkpoint
    ckpt_path = os.path.join(log_dir, "checkpoints", config["ckpt_name"])
    pl_model = EmotionDisentangleModule.load_from_checkpoint(ckpt_path, dataset_stats=stats, **config["lightning"]).to(config["device"])
    
    # Load VP emotion model (pretrained/fixed)
    emotion_model = VoxProfileEmotionModel(device=config["device"], split_models=True)
    
    # Load speech codec
    codec_class, codec_sr = CODECS[codec_name]
    codec = codec_class(device=config["device"])
    
    # Load dataset and create dataloader
    dataset_class, dataset_sr = DATASETS[dataset_name]
    dataset = dataset_class(**config["dataset"], split="dev") # CHANGEME to test when ready
    
    # Process each sample
    for i, sample in tqdm.tqdm(enumerate(dataset), total=len(dataset), desc="Running Eval"):
        
        audio = sample["audio"].to(config["device"])
        label = sample["emotion"]
        filename = sample["filename"]
        length = sample["length"]
        
        with torch.no_grad():
            emotion_logits_raw, emotion_embedding = emotion_model(
                audio, sr=dataset_sr, return_embeddings=True, lengths=torch.tensor([length]).to(config["device"])
            )

        with torch.no_grad():
            embedding = codec.encode(audio, sr=dataset_sr)
            _, quantized_embedding = codec.quantize(embedding)
            embedding_private, _ = pl_model(quantized_embedding, emotion_embedding[f"{config['emotion_conditioning_model']}_embedding"]) # CHANGEME
            codes_private, _ = codec.quantize(embedding_private)
            audio_private = codec.decode(codes_private)

        with torch.no_grad():
            embedding_self_recon, _ = pl_model(quantized_embedding, emotion_embedding[f"{config['emotion_conditioning_model']}_embedding"]) # DEBUG
            codes_self_recon, _ = codec.quantize(embedding_self_recon)
            audio_self_recon = codec.decode(codes_self_recon)

        audio_private = torchaudio.functional.resample(
                    audio_private, orig_freq=codec_sr, new_freq=dataset_sr
                ).unsqueeze(0)
        
        audio_self_recon = torchaudio.functional.resample(
                    audio_self_recon, orig_freq=codec_sr, new_freq=dataset_sr
                ).unsqueeze(0)
                
        with torch.no_grad():
            emotion_logits_private = emotion_model(
                audio_private, sr=dataset_sr, return_embeddings=False, lengths=torch.tensor([length]).to(config["device"])
            )
            
        # Compute stats for debugging
        def get_stats(tensor):
            return {
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "max": tensor.max().item(),
            "min": tensor.min().item(),
            }
        
        save_dict = {
            "filename": filename,
            "label": label,
            "whisper_emotion_logits_raw": emotion_logits_raw["whisper_logits"].cpu().squeeze(),
            "whisper_emotion_logits_private": emotion_logits_private["whisper_logits"].cpu().squeeze(),
            "wavlm_emotion_logits_raw": emotion_logits_raw["wavlm_logits"].cpu().squeeze(),
            "wavlm_emotion_logits_private": emotion_logits_private["wavlm_logits"].cpu().squeeze(),
            "raw_embedding_stats": get_stats(quantized_embedding),
            "private_embedding_stats": get_stats(embedding_private),
            "self_recon_embedding_stats": get_stats(embedding_self_recon),
        }
        
        if i <= config["num_samples_to_save"]: # Save audio only for first N samples
            save_dict["audio_raw"] = audio.cpu().squeeze()
            save_dict["audio_private"] = audio_private.cpu().squeeze()
            save_dict["audio_self_recon"] = audio_self_recon.cpu().squeeze()
        
        save_path = os.path.join(save_root, f"{i}_{filename}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)
    
    
    
