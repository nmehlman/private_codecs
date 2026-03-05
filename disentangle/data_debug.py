# Paste into a script/run in the same environment your training runs in.
import sys, numpy as np, torch
from torch.utils.data import DataLoader
from disentangle.codec_data import get_dataloaders

def bytes_per_sample(sample):
    total = 0
    def add(obj):
        nonlocal total
        if isinstance(obj, torch.Tensor):
            total += obj.element_size() * obj.nelement()
        elif hasattr(obj, "nbytes"):
            total += int(getattr(obj, "nbytes"))
        elif isinstance(obj, (bytes, bytearray)):
            total += len(obj)
        else:
            try:
                total += sys.getsizeof(obj)
            except Exception:
                pass
    if isinstance(sample, dict):
        for v in sample.values(): add(v)
    elif isinstance(sample, (list, tuple)):
        for v in sample: add(v)
    else:
        add(sample)
    return total

# === CONFIG ===
probe_samples = 1024           # how many samples to measure
batch_sizes_to_test = [8, 16, 32, 64]  # edit as needed
num_workers_assumed = 4       # set to your common setting
prefetch_factor = 2           # torch default; set 0 or 1 to simulate lower prefetch

# === MEASURE ===
dl = get_dataloaders(
        {"dataset_path": "/project2/shrikann_35/DATA/expresso/codec_feats/",
        "codec": "hificodec",
        "emotion_model": "wavlm"}
        , batch_size=1, num_workers=num_workers_assumed, prefetch_factor=prefetch_factor
    )['train']

sizes = []
for i, s in enumerate(dl):
    sizes.append(bytes_per_sample(s))
    if i+1 >= probe_samples:
        break

mean_sample = float(np.mean(sizes))
median_sample = float(np.median(sizes))
print(f"avg_bytes_per_sample = {mean_sample:,} bytes, median = {median_sample:,} bytes")

# === ESTIMATE HOST RAM ===
def estimate_host_ram(batch_size, W=num_workers_assumed, pf=prefetch_factor,
                      dataset_ram_est=0, pinned_overhead=0):
    # dataset_ram_est: if you preload dataset into RAM, estimate its bytes (0 if not)
    batch_bytes = batch_size * mean_sample
    # assume prefetch keeps pf batches per worker (PyTorch uses prefetch_factor per worker)
    # conservative host batch buffer factor:
    host_batch_buffer = batch_bytes * (1 + pf)  
    worker_overhead = W * (mean_sample * 2)  # heuristic: each worker may hold ~2 samples' worth
    total = dataset_ram_est + worker_overhead + host_batch_buffer + pinned_overhead
    return total

for b in batch_sizes_to_test:
    est = estimate_host_ram(b)
    print(f"batch {b:2d} -> estimated host RAM ≈ {int(est/1024**2):6d} MB")
