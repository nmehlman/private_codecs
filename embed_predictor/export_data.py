from fileinput import filename
from data.expresso import ExpressoDataset, EXPRESSO_SR, EXPRESSO_TO_VP_LABEL_MAPPING
from data.msp_podcast import MSPPodcastDataset, MSP_SR, MSP_TO_VP_LABEL_MAPPING
from network.codec import HifiCodec, EnCodec, BigCodec, HIFICODEC_SR, ENCODEC_SR, BIGCODEC_SR
import tqdm
import torch
import argparse
import pickle
import os
import yaml
from torch.utils.data import DataLoader

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

    parser = argparse.ArgumentParser(description="Run inference for a specified task.")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )
    
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    print(f"Exporting:\n\tCodec: {config['codec']}\n\tDataset: {config['dataset']}\n\tDevice: {config['device']}")
    
    save_root = config["save_path"]
    os.makedirs(save_root, exist_ok=True)

    codec_class, codec_sr = CODECS[config["codec"]]
    dataset_class, dataset_sr = DATASETS[config["dataset"]]

    codec = codec_class(device=config["device"])
    dataset = dataset_class(**config["data_args"])

    for sample in tqdm.tqdm(dataset, total=len(dataset), desc="Exporting Data"):
        
        audio = sample["audio"].to(config["device"])
        label = sample["emotion"]
        filename = sample["filename"]
        
        with torch.no_grad():
            embeddings = codec.encode(audio, sr=dataset_sr)
            codes, quantized_embeddings = codec.quantize(embeddings)

        codes = codes.squeeze()
        quantized_embeddings = quantized_embeddings.squeeze()

        save_dict = {
                "filename": filename,
                "label": label,
                "codes": codes.cpu(),
                "quantized_embedding": quantized_embeddings.cpu(),
                "raw_embedding": embeddings.cpu().squeeze(),
            }
        
        save_path = os.path.join(save_root, f"{filename}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)