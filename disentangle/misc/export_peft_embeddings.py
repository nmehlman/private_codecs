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
import sys
import glob

# Add peft-ser to path and import
sys.path.append("/home1/nmehlman/private_codecs/peft-ser/package")
import peft_ser

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
    
    parser.add_argument(
        "--peft_model",
        type=str,
        default="whisper-base-lora-16-conv",
        help="PEFT model name to use for embeddings."
    )
    
    parser.add_argument(
        "--peft_cache_folder",
        type=str,
        default="/project2/shrikann_35/nmehlman/models/",
        help="Cache folder for PEFT model."
    )
    
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    dataset_name = config["dataset_name"]
    codec_name = config["codec_name"]

    print(f"Adding PEFT embeddings:\n\tCodec: {codec_name}\n\tDataset: {dataset_name}\n\tPEFT Model: {args.peft_model}\n\tDevice: {config['device']}")
    
    save_root = config["save_path"]
    if not os.path.exists(save_root):
        raise ValueError(f"Save path {save_root} does not exist. Please run export_data.py first.")

    dataset_class, dataset_sr = DATASETS[dataset_name]
    dataset = dataset_class(**config["dataset"])
    
    # Load PEFT model
    print(f"Loading PEFT model: {args.peft_model}")
    peft_model = peft_ser.load_model(args.peft_model, cache_folder=args.peft_cache_folder)
    peft_model.eval()

    # Process each sample in the dataset and check if corresponding pickle file exists
    for sample in tqdm.tqdm(dataset, total=len(dataset), desc="Adding PEFT embeddings"):
        
        audio = sample["audio"].to(config["device"])
        filename = sample["filename"]
        length = sample["length"]
        
        # Check if the corresponding pickle file exists
        pickle_path = os.path.join(save_root, f"{filename}.pkl")
        if not os.path.exists(pickle_path):
            print(f"Warning: No existing data file found for {filename}, skipping")
            continue
        
        # Load existing data
        with open(pickle_path, "rb") as f:
            existing_data = pickle.load(f)
        
        # Skip if PEFT embeddings already exist
        if "peft_embedding" in existing_data:
            continue
        
        # Get PEFT embeddings
        with torch.no_grad():
            # Ensure audio is in correct format (batch dimension)
            if audio.dim() == 1:
                audio_input = audio.unsqueeze(0)
            else:
                audio_input = audio
                
            try:
                peft_output, peft_features = peft_model(audio_input, return_features=True)
                
                # Check for NaN values in PEFT output
                if torch.isnan(peft_output).any() or torch.isnan(peft_features).any():
                    print(f"Skipping {filename} due to NaN values in PEFT output")
                    continue
                
                # Add PEFT embeddings to existing data
                existing_data["peft_logits"] = peft_output.cpu().squeeze()
                existing_data["peft_embedding"] = peft_features.cpu().squeeze()
                
                # Save augmented data back to the same file
                with open(pickle_path, "wb") as f:
                    pickle.dump(existing_data, f)
                    
            except Exception as e:
                print(f"Error processing {filename} with PEFT model: {e}")
                continue
