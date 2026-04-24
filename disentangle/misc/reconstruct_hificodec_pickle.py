import argparse
import pickle
from pathlib import Path

import torch
import torchaudio

from network.codec import HifiCodec, HIFICODEC_SR


def describe_value(value):
    if torch.is_tensor(value):
        if value.numel() <= 16:
            return value
        return f"tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
    return value


def load_codes(value):
    codes = value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)

    if codes.ndim == 2:
        codes = codes.unsqueeze(0)
    elif codes.ndim == 3 and codes.shape[0] != 1:
        codes = codes[:1]

    return codes.long()


def main():
    parser = argparse.ArgumentParser(description="Load an exported pickle, print its fields, and reconstruct audio with HifiCodec.")
    parser.add_argument("pickle_path", type=Path, help="Path to a .pkl file produced by export_data.py")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save the reconstructed wav")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for reconstruction")
    parser.add_argument("--config-path", type=str, default="/home1/nmehlman/private_codecs/AcademiCodec/egs/HiFi-Codec-16k-320d/config_16k_320d.json")
    parser.add_argument("--model-path", type=str, default="/project2/shrikann_35/nmehlman/models/HiFi-Codec-16k-320d")
    args = parser.parse_args()

    with args.pickle_path.open("rb") as handle:
        sample = pickle.load(handle)

    print(f"Loaded: {args.pickle_path}")
    for key, value in sample.items():
        if key == "codes":
            continue
        print(f"{key}: {describe_value(value)}")

    codes = load_codes(sample["codes"])

    codec = HifiCodec(
        config_path=args.config_path,
        model_path=args.model_path,
        device=args.device,
    )

    with torch.no_grad():
        recon_audio = codec.decode(codes).detach().cpu()

    output_dir = args.output_dir or args.pickle_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.pickle_path.stem}_recon.wav"

    torchaudio.save(str(output_path), recon_audio.unsqueeze(0), sample_rate=HIFICODEC_SR)
    print(f"Saved reconstructed audio to: {output_path}")


if __name__ == "__main__":
    main()