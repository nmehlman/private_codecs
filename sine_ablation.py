"""Passes sine waves of linearly spaced frequencies through HiFiCodec and saves the mean embeddings."""

import os
import torch
import tqdm

from network.codec import HifiCodec, HIFICODEC_SR

SAVE_DIR = "/home1/nmehlman/private_codecs/sine-ablation"

NUM_FREQS = 100
MIN_FREQ = 100.0
MAX_FREQ = 5000.0
DURATION = 1.0  # seconds

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec = HifiCodec(device=device)
    codec.model.eval()

    freqs = torch.linspace(MIN_FREQ, MAX_FREQ, NUM_FREQS)
    t = torch.arange(int(DURATION * HIFICODEC_SR)) / HIFICODEC_SR

    mean_embeds = []
    with torch.no_grad():
        for freq in tqdm.tqdm(freqs, desc="Encoding sine waves"):
            sine = torch.sin(2 * torch.pi * freq * t).unsqueeze(0)  # (1, T)
            embeds = codec.encode(sine, sr=HIFICODEC_SR)  # (1, D, T')
            mean_embeds.append(embeds.mean(dim=-1).squeeze(0).cpu())  # (D,)

    mean_embeds = torch.stack(mean_embeds, dim=0)  # (NUM_FREQS, D)

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "sine_mean_embeddings.pt")
    torch.save({"frequencies": freqs, "mean_embeddings": mean_embeds}, save_path)
    print(f"Saved {tuple(mean_embeds.shape)} embeddings to {save_path}")
