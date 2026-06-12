import os
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

# save_dict = {
#             "filename": results["filename"],
#             "label": results["label"],
#             "sex_logits_raw": results["sex_logits_raw"],
#             "sex_logits_private": results["sex_logits_private"],
#             "sex_logits_codec_only": results["sex_logits_codec_only"],
#             "private_embedding_stats": results["private_embedding_stats"],
#             "difference_metrics": results["difference_metrics"],
#         }

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing the results .pt files")
    args = parser.parse_args()
    
    # Load all results
    results_files = [f for f in os.listdir(args.results_dir) if f.endswith(".pkl")]
    all_results = []
    for f in results_files:
        results_path = os.path.join(args.results_dir, f)
        results = pickle.load(open(results_path, "rb"))
        all_results.append(results)
    
    # Aggregate results
    accuracy_raw = np.mean([r["label"] == np.argmax(r["sex_logits_raw"]) for r in all_results])
    accuracy_private = np.mean([r["label"] == np.argmax(r["sex_logits_private"]) for r in all_results])
    accuracy_codec_only = np.mean([r["label"] == np.argmax(r["sex_logits_codec_only"]) for r in all_results])

    print(f"Raw Accuracy: {accuracy_raw:.4f}")
    print(f"Private Accuracy: {accuracy_private:.4f}")
    print(f"Codec-Only Accuracy: {accuracy_codec_only:.4f}")

    def compute_entropy(logits):
        probs = torch.softmax(torch.tensor(logits), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
        return entropy
    
    entropies_raw = [compute_entropy(r["sex_logits_raw"]) for r in all_results]
    entropies_private = [compute_entropy(r["sex_logits_private"]) for r in all_results]
    entropies_codec_only = [compute_entropy(r["sex_logits_codec_only"]) for r in all_results]

    print(f"Average Entropy - Raw: {np.mean(entropies_raw):.4f} +/- {np.std(entropies_raw):.4f}")
    print(f"Average Entropy - Private: {np.mean(entropies_private):.4f} +/- {np.std(entropies_private):.4f}")
    print(f"Average Entropy - Codec-Only: {np.mean(entropies_codec_only):.4f} +/- {np.std(entropies_codec_only):.4f}")