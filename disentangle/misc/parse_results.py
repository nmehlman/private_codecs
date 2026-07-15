import os
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

def parse_results(results_dir):
    
    # Load all results
    results_files = [f for f in os.listdir(results_dir) if f.endswith(".pkl")]
    all_results = []
    for f in results_files:
        results_path = os.path.join(results_dir, f)
        results = pickle.load(open(results_path, "rb"))
        all_results.append(results)

    def mean_ignore_none(values):
        valid_values = [value for value in values if value is not None]
        return np.mean(valid_values) if valid_values else None
    
    # Aggregate results
    accuracy_raw = np.mean([r["label"] == np.argmax(r["sex_logits_raw"]) for r in all_results])
    accuracy_private = np.mean([r["label"] == np.argmax(r["sex_logits_private"]) for r in all_results])
    accuracy_codec_only = np.mean([r["label"] == np.argmax(r["sex_logits_codec_only"]) for r in all_results])

    asr_results = [r.get("asr") for r in all_results if r.get("asr") is not None]
    wer_raw_ref = mean_ignore_none([r["wer_raw_ref"] for r in asr_results])
    wer_private_ref = mean_ignore_none([r["wer_private_ref"] for r in asr_results])
    wer_codec_only_ref = mean_ignore_none([r["wer_codec_only_ref"] for r in asr_results])
    wer_private_raw = mean_ignore_none([r["wer_private_raw"] for r in asr_results])
    wer_private_codec_only = mean_ignore_none([r["wer_private_codec_only"] for r in asr_results])

    def compute_entropy(logits):
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
        return entropy
    
    entropies_raw = np.mean([compute_entropy(r["sex_logits_raw"]) for r in all_results])
    entropies_private = np.mean([compute_entropy(r["sex_logits_private"]) for r in all_results])
    entropies_codec_only = np.mean([compute_entropy(r["sex_logits_codec_only"]) for r in all_results])

    return {
        "accuracy_raw": accuracy_raw,
        "accuracy_private": accuracy_private,
        "accuracy_codec_only": accuracy_codec_only,
        "entropy_raw": entropies_raw,
        "entropy_private": entropies_private,
        "entropy_codec_only": entropies_codec_only,
        "wer_raw_ref": wer_raw_ref if wer_raw_ref is not None else None,
        "wer_private_ref": wer_private_ref if wer_private_ref is not None else None,
        "wer_codec_only_ref": wer_codec_only_ref if wer_codec_only_ref is not None else None,
        "wer_private_raw": wer_private_raw if wer_private_raw is not None else None,
        "wer_private_codec_only": wer_private_codec_only if wer_private_codec_only is not None else None
    }

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing the results .pt files")
    args = parser.parse_args()
    
    parsed_results = parse_results(args.results_dir)
    for key, value in parsed_results.items():
        if value is None:
            print(f"{key}: None")
        else:
            print(f"{key}: {value:.4f}")
