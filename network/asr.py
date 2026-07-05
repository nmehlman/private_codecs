import torch
import torch.nn as nn
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class WhisperASR(nn.Module):
    def __init__(self, device="cpu", pretrain_model="openai/whisper-large-v3", language='en'):
        
        super().__init__()
        self.device = device
        self.pretrain_model = pretrain_model
        self.language = language

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
        
        kwargs = {}
        if self.language is not None:
            kwargs["generate_kwargs"] = {"language": self.language}
        
        results = self.pipe(x, **kwargs)
        return results["text"]
    
if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    sample = torch.randn(16000 * 5).numpy()  # 5 seconds of random noise at 16kHz

    result = pipe(sample)
    print(result["text"])
