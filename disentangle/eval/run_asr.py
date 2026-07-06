from network.asr import WhisperASR
import os
import json
from jiwer import wer  # type: ignore
import torch
from typing import Union
import numpy as np

def _compute_wer(reference_text: dict, transcriptions: dict):
    
    wer_results = []
    for filename, ref_text in reference_text.items():
        if filename not in transcriptions:
            raise ValueError(f"Transcription for {filename} not found.")
        
        hyp_text = transcriptions[filename]
        error = wer(ref_text, hyp_text)
        wer_results.append(error)

    return np.mean(wer_results)

def run_asr_eval(cache_dir: str, save_path: str, device: str = "cuda", reference_text: Union[dict, None] = None):

    asr = WhisperASR(device=device)
    raw_audio_path = os.path.join(cache_dir, "raw_audio")
    codec_only_audio_path = os.path.join(cache_dir, "codec_only_audio")
    private_audio_path = os.path.join(cache_dir, "private_audio")

    # Run transcription
    print("Running ASR on raw audio")
    raw_transcriptions = asr.transcribe_dir(raw_audio_path, os.path.join(cache_dir, "raw_audio_transcriptions.json"))
    
    print("Running ASR on codec-only audio")
    codec_only_transcriptions = asr.transcribe_dir(codec_only_audio_path, os.path.join(cache_dir, "codec_only_audio_transcriptions.json"))
    
    print("Running ASR on private audio")
    private_transcriptions = asr.transcribe_dir(private_audio_path, os.path.join(cache_dir, "private_audio_transcriptions.json"))

    # Calculate WER
    if reference_text is not None:
        print("Calculating WER against reference text")
        wer_raw_ref = _compute_wer(reference_text, raw_transcriptions)
        wer_codec_only_ref = _compute_wer(reference_text, codec_only_transcriptions)
        wer_private_ref = _compute_wer(reference_text, private_transcriptions)

    else:
        print("No reference text provided, skipping WER against reference.")
        wer_raw_ref = None
        wer_codec_only_ref = None
        wer_private_ref = None

    # Calculate WER between private and raw, and private and codec-only
    print("Calculating WER between private and raw audio")
    wer_private_raw = _compute_wer(raw_transcriptions, private_transcriptions)
    print("Calculating WER between private and codec-only audio")
    wer_private_codec_only = _compute_wer(codec_only_transcriptions, private_transcriptions)
    print("Calculating WER between codec-only and raw audio")
    wer_codec_only_raw = _compute_wer(raw_transcriptions, codec_only_transcriptions) 

    results = {
        "wer_raw_ref": wer_raw_ref,
        "wer_codec_only_ref": wer_codec_only_ref,
        "wer_private_ref": wer_private_ref,
        "wer_private_raw": wer_private_raw,
        "wer_private_codec_only": wer_private_codec_only,
        "wer_codec_only_raw": wer_codec_only_raw,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True, help="Directory containing cached audio files")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the WER results JSON")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run ASR on (default: cuda)")
    parser.add_argument("--reference_text_path", type=str, default=None, help="Path to reference text JSON for WER calculation (optional)")
    args = parser.parse_args()

    if args.reference_text_path:
        with open(args.reference_text_path, "r", encoding="utf-8") as f:
            reference_text = json.load(f)
    else:
        reference_text = None

    run_asr_eval(args.cache_dir, args.save_path, device=args.device, reference_text=reference_text)