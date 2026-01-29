from pathlib import Path
import json

def load_dataset_stats(dataset_name, codec, input_type):
    
    _HERE = Path(__file__).resolve().parent
    stats_path = _HERE / f"dataset_stats_{codec}_{dataset_name}.json"
    
    with stats_path.open("r") as f:
        dataset_stats = json.load(f)
    
    return dataset_stats[input_type]