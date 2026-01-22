#!/usr/bin/env bash

set -e  # exit immediately if any command fails

vals=(0.0 1.0 5.0 10.0 30.0 75.0 100.0 125.0)

for val in "${vals[@]}"; do
    echo "Running sigma=${val}"
    python run.py --config "configs/informed_noise_masking/masked_emotion_inference_encodec_expresso_sigma=${val}.yaml"
done