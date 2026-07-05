# TODO swap out prototypes from other cases

from disentangle.lightning import SexDisentangleModule
from disentangle.misc.utils import load_dataset_stats
from network.models import VoxProfileAgeSexModel
from data.expresso import ExpressoDataset, EXPRESSO_SR
from data.msp_podcast import MSPPodcastDataset, MSP_SR
from data.vox1 import Vox1Dataset, VOX1_SR
from network.codec import HifiCodec, EnCodec, BigCodec, HIFICODEC_SR, ENCODEC_SR, BIGCODEC_SR
from network.asr import WhisperASR

import argparse
import os
import re
import pytorch_lightning as pl # type: ignore
import yaml  # type: ignore

import tqdm  # type: ignore
import torch  # type: ignore
import torchaudio  # type: ignore
from torch.utils.data import DataLoader
import pickle
import json
from jiwer import wer  # type: ignore

from disentangle.lightning import compute_difference_metric

def get_stats(tensor):
        return {
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "max": tensor.max().item(),
            "min": tensor.min().item(),
        }


def process_batch(batch, codec, pl_model, sex_model, dataset_sr, codec_sr, asr_model=None, device=None):
    
    """Process a batch of samples."""
    
    audios = batch["audio"].to(device)
    labels = batch["gender"]
    filenames = batch["filename"]
    lengths = batch["length"].to(device)
    
    batch_size = audios.shape[0]
    results_list = []
    
    # Get embedding for raw audio
    with torch.no_grad():
        _, sex_logits_raw = sex_model(
            audios, sr=dataset_sr, return_embeddings=False, lengths=lengths
        )
    
    # Encode audio with codec
    with torch.no_grad():
        embedding_raw = codec.encode(audios, sr=dataset_sr)
        codes_raw, quantized_embedding_raw = codec.quantize(embedding_raw)
    
    with torch.no_grad():
        embedding_private, _ = pl_model(quantized_embedding_raw)
        codes_private, embedding_private_quantized = codec.quantize(embedding_private)
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
                lengths=lengths
            )
       
        _, sex_logits_codec_only = sex_model(
            audio_codec_only, sr=dataset_sr, return_embeddings=False,
            lengths=lengths
        )  
    
    # Process ASR if provided (batch processing)
    transcriptions_raw = None
    transcriptions_private = None
    transcriptions_codec_only = None
    
    if asr_model is not None:
        transcriptions_raw = asr_model.transcribe(audios, sr=dataset_sr)
        transcriptions_private = asr_model.transcribe(audio_private, sr=dataset_sr)
        transcriptions_codec_only = asr_model.transcribe(audio_codec_only, sr=dataset_sr)
    
    # Build results list for each sample in batch
    for i in range(batch_size):
        # Get reference text if available
        if asr_model is not None:
            reference_text = batch.get("transcript", batch.get("text", batch.get("reference", [""] * batch_size)))[i] if isinstance(batch.get("transcript", batch.get("text", batch.get("reference", []))), list) else ""
            wer_raw_ref = wer(reference_text, transcriptions_raw[i]) if reference_text else None
            wer_private_ref = wer(reference_text, transcriptions_private[i]) if reference_text else None
            wer_codec_only_ref = wer(reference_text, transcriptions_codec_only[i]) if reference_text else None
            wer_private_raw = wer(transcriptions_raw[i], transcriptions_private[i]) if transcriptions_raw[i] and transcriptions_private[i] else None
            wer_private_codec_only = wer(transcriptions_codec_only[i], transcriptions_private[i]) if transcriptions_codec_only[i] and transcriptions_private[i] else None
        else:
            wer_raw_ref = wer_private_ref = wer_codec_only_ref = wer_private_raw = wer_private_codec_only = None
        
        # Build results dict
        results = {
            "filename": filenames[i],
            "label": labels[i],
            "sex_logits_raw": sex_logits_raw[i].cpu(),
            "sex_logits_private": sex_logits_private[i].cpu(),
            "sex_logits_codec_only": sex_logits_codec_only[i].cpu(),
            "raw_embedding_stats": get_stats(quantized_embedding_raw[i]),
            "private_embedding_stats": get_stats(embedding_private_quantized[i]),
            "audio_raw": audios[i].cpu(),
            "audio_private": audio_private[i].cpu(),
            "audio_codec_only": audio_codec_only[i].cpu(),
            "difference_metrics": compute_difference_metric(quantized_embedding_raw[i:i+1], embedding_private_quantized[i:i+1]),
        }
        
        if asr_model is not None:
            results["asr"] = { 
                "transcription_raw": transcriptions_raw[i],
                "transcription_private": transcriptions_private[i],
                "transcription_codec_only": transcriptions_codec_only[i],
                "wer_raw_ref": wer_raw_ref,
                "wer_private_ref": wer_private_ref,
                "wer_codec_only_ref": wer_codec_only_ref,
                "wer_private_raw": wer_private_raw,
                "wer_private_codec_only": wer_private_codec_only   
            }
        
        results_list.append(results)
    
    return results_list


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

    stats = load_dataset_stats(dataset_name, codec_name, input_type)
    
    # Load disentanglement model from checkpoint
    ckpt_path = _resolve_checkpoint_path(log_dir, config.get("ckpt_name", None))
    pl_model = SexDisentangleModule.load_from_checkpoint(ckpt_path, dataset_stats=stats, **train_config["lightning"]).to(config["device"]).eval()
    
    # Load VP model (pretrained/fixed)
    sex_model = VoxProfileAgeSexModel(device=config["device"])
    
    # Load ASR model
    asr_model = WhisperASR(device=config["device"])
    
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
    
    # Create dataloader for batch processing
    batch_size = config.get("batch_size", 4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Process each batch
    sample_idx = 0
    for batch in tqdm.tqdm(dataloader, desc="Running Eval"):
        
        results_list = process_batch(batch, codec, pl_model, sex_model, dataset_sr, codec_sr, asr_model=asr_model, device=config["device"])
        
        # Save each result in the batch
        for results in results_list:
            # Build save dict, optionally excluding audio to save space
            save_dict = { 
                "label": results["label"],
                "sex_logits_raw": results["sex_logits_raw"],
                "sex_logits_private": results["sex_logits_private"],
                "sex_logits_codec_only": results["sex_logits_codec_only"],
                "private_embedding_stats": results["private_embedding_stats"],
                "difference_metrics": results["difference_metrics"],
                "asr": results.get("asr", None)
            }
            
            if sample_idx <= config["num_samples_to_save"]:  # Save audio only for first N samples
                save_dict["audio_raw"] = results["audio_raw"]
                save_dict["audio_private"] = results["audio_private"]
                save_dict["audio_codec_only"] = results["audio_codec_only"]
            
            save_path = os.path.join(save_root, f"{sample_idx}_{results['filename']}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(save_dict, f)
            
            sample_idx += 1

    
    
    
