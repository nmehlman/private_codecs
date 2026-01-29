from fileinput import filename
from data.expresso import ExpressoDataset, EXPRESSO_SR
from data.msp_podcast import MSPPodcastDataset, MSP_SR
from network.codec import HifiCodec, EnCodec, BigCodec, HIFICODEC_SR, ENCODEC_SR, BIGCODEC_SR
import tqdm
import torch
import argparse
import pickle
import os
import yaml

from network.models import VoxProfileEmotionModel

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
    
    dataset_name = config["dataset_name"]
    codec_name = config["codec_name"]

    print(f"Exporting:\n\tCodec: {codec_name}\n\tDataset: {dataset_name}\n\tDevice: {config['device']}")
    
    save_root = config["save_path"]
    os.makedirs(save_root, exist_ok=True)

    codec_class, codec_sr = CODECS[codec_name]
    dataset_class, dataset_sr = DATASETS[dataset_name]

    emotion_model = VoxProfileEmotionModel(device=config["device"], split_models=True)

    codec = codec_class(device=config["device"])
    dataset = dataset_class(**config["data_args"])

    for sample in tqdm.tqdm(dataset, total=len(dataset), desc="Exporting Data"):
        
        audio = sample["audio"].to(config["device"])
        label = sample["emotion"]
        filename = sample["filename"]
        length = sample["length"]

        emotion_logits, emotion_embedding = emotion_model(
            audio, sr=dataset_sr, return_embeddings=True, lengths=torch.tensor([length]).to(config["device"])
        )
        
        with torch.no_grad():
            embeddings = codec.encode(audio, sr=dataset_sr)
            codes, quantized_embeddings = codec.quantize(embeddings)

        codes = codes.squeeze()
        quantized_embeddings = quantized_embeddings.squeeze()

        # Check for NaN values
        if torch.isnan(embeddings).any() or torch.isnan(codes).any() or torch.isnan(quantized_embeddings).any():
            print(f"Skipping {filename} due to NaN values in codec output")
            continue
        
        if torch.isnan(emotion_logits["whisper_logits"]).any() or torch.isnan(emotion_logits["wavlm_logits"]).any():
            print(f"Skipping {filename} due to NaN values in emotion logits")
            continue

        save_dict = {
                "filename": filename,
                "label": label,
                "codes": codes.cpu(),
                "quantized_embedding": quantized_embeddings.cpu(),
                "raw_embedding": embeddings.cpu().squeeze(),
                "whisper_emotion_logits": emotion_logits["whisper_logits"].cpu().squeeze(),
                "wavlm_emotion_logits": emotion_logits["wavlm_logits"].cpu().squeeze(),
                "whisper_emotion_embedding": emotion_embedding["whisper_embedding"].cpu().squeeze(),
                "wavlm_emotion_embedding": emotion_embedding["wavlm_embedding"].cpu().squeeze(),
            }
        
        save_path = os.path.join(save_root, f"{filename}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)
