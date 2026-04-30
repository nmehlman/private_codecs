from fileinput import filename
from data.expresso import ExpressoDataset, EXPRESSO_SR
from data.msp_podcast import MSPPodcastDataset, MSP_SR
from data.vox1 import Vox1Dataset, VOX1_SR
from network.codec import HifiCodec, EnCodec, BigCodec, HIFICODEC_SR, ENCODEC_SR, BIGCODEC_SR
from network.foundation import WavLMWrapper, WhisperWrapper, WAVLM_SR, WHISPER_SR
import tqdm
import torch
import argparse
import pickle
import os
import yaml

from network.models import VoxProfileAgeSexModel

CODECS = {
    "encodec": (EnCodec, ENCODEC_SR),
    "hificodec": (HifiCodec, HIFICODEC_SR),
    "bigcodec": (BigCodec, BIGCODEC_SR),
    "wavlm": (WavLMWrapper, WAVLM_SR),
    "whisper": (WhisperWrapper, WHISPER_SR)
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
    
    dataset_name = config["dataset_name"]
    codec_name = config["codec_name"]

    print(f"Exporting:\n\tCodec: {codec_name}\n\tDataset: {dataset_name}\n\tDevice: {config['device']}")
    
    save_root = config["save_path"]
    os.makedirs(save_root, exist_ok=True)

    codec_class, codec_sr = CODECS[codec_name]
    dataset_class, dataset_sr = DATASETS[dataset_name]

    codec = codec_class(device=config["device"]).to(config["device"])
    dataset = dataset_class(**config["dataset"])

    for i, sample in tqdm.tqdm(enumerate(dataset), total=len(dataset), desc="Exporting Data"):
        
        audio = sample["audio"].to(config["device"])
        # Handle both emotion and gender labels depending on dataset
        if dataset_name == "vox1":
            label = sample["gender"]
        else:
            label = sample["emotion"]
        filename = sample["filename"]
        length = sample["length"]
        
        with torch.no_grad():
            embeddings = codec.encode(audio, sr=dataset_sr)
            codes, quantized_embeddings = codec.quantize(embeddings)

        codes = codes.squeeze()
        quantized_embeddings = quantized_embeddings.squeeze()

        # Check for NaN values
        if torch.isnan(embeddings).any() or torch.isnan(codes).any() or torch.isnan(quantized_embeddings).any():
            print(f"Skipping {filename} due to NaN values in codec output")
            continue
        
        save_dict = {
                "filename": filename,
                "label": label,
                "codes": codes.cpu(),
                "quantized_embedding": quantized_embeddings.cpu(),
                "raw_embedding": embeddings.cpu().squeeze(),
            }

        
        save_path = os.path.join(save_root, f"{i}_{filename}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)
