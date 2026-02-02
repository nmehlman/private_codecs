from pathlib import Path
import json
import torch

def load_dataset_stats(dataset_name, codec, input_type):
    
    _HERE = Path(__file__).resolve().parent
    stats_path = _HERE / f"dataset_stats_{codec}_{dataset_name}.json"
    
    with stats_path.open("r") as f:
        dataset_stats = json.load(f)
    
    return dataset_stats[input_type]

def load_emotion_prototypes(dataset_name, split, emotion_model):
    _HERE = Path(__file__).resolve().parent
    prototypes_path = _HERE / f"emotion_prototypes_{dataset_name}_{split}_{emotion_model}.pt"
    
    emotion_prototypes = torch.load(prototypes_path)
    
    return emotion_prototypes