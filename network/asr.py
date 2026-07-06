import torch
import torch.nn as nn
import torchaudio
import os
import json
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class WhisperASR(nn.Module):
    def __init__(self, device="cpu", pretrain_model="openai/whisper-large-v3"):
        
        super().__init__()
        self.device = device
        self.pretrain_model = pretrain_model

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.pretrain_model, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.pretrain_model)
        self.sample_rate = self.processor.feature_extractor.sampling_rate
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
        )

    def transcribe(self, x: torch.Tensor, sr: int):
        """Return transcription for a single sample."""
        
        if x.ndim == 2:
            assert x.shape[0] == 1, "Currently only supports single-sample batches"
        
        if sr != self.sample_rate:
            x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=self.sample_rate)
        
        x = x.squeeze().numpy()
        
        results = self.pipe(x, generate_kwargs={"language": "en"})
        return results["text"]
    
    def transcribe_dir(self, audio_dir: str, save_path: str):
        """Transcribe all audio files in a directory and save results to a text file."""
        
        audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith((".wav", ".flac", ".mp3"))]
        
        def data_generator(file_paths):
            for path in file_paths:
                array, sr = librosa.load(path, sr=self.sample_rate)  # librosa resamples internally
                yield {"raw": array, "sampling_rate": self.sample_rate}

        results = []
        for out in self.pipe(data_generator(audio_files), batch_size=8, generate_kwargs={"language": "en"}):
            results.append(out["text"])

        # Save transcriptions as JSON mapping filename -> transcription
        trans_dict = {os.path.basename(file_path): transcription for file_path, transcription in zip(audio_files, results)}
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(trans_dict, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = WhisperASR(device=device, pretrain_model="openai/whisper-large-v3")
    model.transcribe_dir('/home1/nmehlman/private_codecs/private_codecs/disentangle/eval/test_audio', "./test_transcriptions.txt")
