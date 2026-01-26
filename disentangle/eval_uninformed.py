# TODO embedding support

from disentangle.lightning import DisentanglementAE
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
        
    save_root = config["save_path"]
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    # Load emotion disentanglement model from checkpoint
    pl_model = DisentanglementAE.load_from_checkpoint(config["ckpt_path"]).to(config["device"])
    
    # Load VP emotion model (pretrained/fixed)
    emotion_model = VoxProfileEmotionModel(device=config["device"], split_models=True)
    
    # Load speech codec
    codec_class, codec_sr = CODECS[config["codec"]]
    codec = codec_class(device=config["device"])
    
    # Load dataset and create dataloader
    dataset_class, dataset_sr = DATASETS[config["dataset"]]
    dataset = dataset_class(**config["dataset"], split="val") # CHANGEME to test when ready
    
    # Process each sample
    for i, sample in tqdm.tqdm(enumerate(dataset), total=len(dataset), desc="Exporting Data"):
        
        audio = sample["audio"].to(config["device"])
        label = sample["emotion"]
        filename = sample["filename"]
        length = sample["length"]

        emotion_logits_raw, _ = emotion_model(
            audio, sr=dataset_sr, return_embeddings=True, lengths=torch.tensor([length]).to(config["device"])
        )
        
        with torch.no_grad():
            embeddings = codec.encode(audio, sr=dataset_sr)
            _, quantized_embeddings = codec.quantize(embeddings)
            # embeddings_private, _ = pl_model(embeddings) # DEBUG
            embeddings_private = quantized_embeddings # DEBUG
            codes_private, _ = codec.quantize(embeddings_private)
            audio_private = codec.decode(codes_private)

        audio_private = torchaudio.functional.resample(
                    audio_private, orig_freq=codec_sr, new_freq=dataset_sr
                )
        
        emotion_logits_private, _ = emotion_model(
            audio_private, sr=dataset_sr, return_embeddings=True, lengths=torch.tensor([length]).to(config["device"])
        )

        save_dict = {
            "filename": filename,
            "label": label,
            "quantized_embedding": quantized_embeddings.cpu(),
            "raw_embedding": embeddings.cpu().squeeze(),
            "whisper_emotion_logits_raw": emotion_logits_raw["whisper_logits"].cpu().squeeze(),
            "whisper_emotion_logits_private": emotion_logits_private["whisper_logits"].cpu().squeeze(),
            "wavlm_emotion_logits_raw": emotion_logits_raw["wavlm_logits"].cpu().squeeze(),
            "wavlm_emotion_logits_private": emotion_logits_private["wavlm_logits"].cpu().squeeze(),
        }
        
        if i <= config["num_samples_to_save"]: # Save audio only for first N samples
            save_dict["audio_raw"] = audio.cpu().squeeze()
            save_dict["audio_private"] = audio_private.cpu().squeeze()
        
        save_path = os.path.join(save_root, f"{filename}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)
    
    
    