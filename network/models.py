import sys
import torch
import torch.nn.functional as F
import torchaudio

sys.path.append("/home/nmehlman/private-codecs/vox-profile")
from src.model.emotion.wavlm_emotion import WavLMWrapper # type: ignore
from src.model.emotion.whisper_emotion import WhisperWrapper # type: ignore

VOX_PROFILE_SR = 16000

VP_EMOTION_LABELS = [
        'Anger', 
        'Contempt', 
        'Disgust', 
        'Fear', 
        'Happiness', 
        'Neutral', 
        'Sadness', 
        'Surprise', 
        'Other'
    ]

class VoxProfileEmotionModel:
        
    def __init__(self, device: str = "cpu") -> None:
        
        self.device = device
        self.sample_rate = VOX_PROFILE_SR

        self.wavlm_model = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-categorical-emotion").to(device)
        self.wavlm_model.eval()

        self.whisper_model = WhisperWrapper.from_pretrained("tiantiaf/whisper-large-v3-msp-podcast-emotion").to(device)
        self.whisper_model.eval()   

    def __call__(self, audio: torch.Tensor, sr: int, return_embeddings: bool = False, lengths: torch.Tensor = None):
        
        audio = audio.to(self.device)

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sample_rate)
            if lengths is not None:
                lengths = lengths * int(self.sample_rate / sr)

        if lengths is None:
            lengths = torch.tensor([audio.shape[1]] * audio.shape[0]).to(self.device)

        whisper_logits, whisper_embedding, _, _, _, _ = self.whisper_model(
            audio, return_feature=True, length=lengths
        )
        
        wavlm_logits, wavlm_embedding, _, _, _, _ = self.wavlm_model(
            audio, return_feature=True, length=lengths
        )

        logits = 1/2 * (whisper_logits + wavlm_logits) # DEBUG
        embedding = torch.cat([whisper_embedding, wavlm_embedding], dim=1)
        
        if return_embeddings:
            return logits, embedding
        else:
            return logits
        
if __name__ == "__main__":
    model = VoxProfileEmotionModel(device="cpu")
    dummy_audio = torch.randn(4, 16000 * 5)
    dummy_lengths = torch.tensor([16000 * 5, 16000 * 5, 16000 * 5, 16000 * 5])
    logits, embedding = model(dummy_audio, sr=16000, return_embeddings=True, lengths=None)
    print("Logits shape:", logits.shape)
    print("Embedding shape:", embedding.shape)