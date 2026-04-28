import sys
import torch
import torch.nn.functional as F
import torchaudio

from src.model.age_sex.wavlm_demographics import WavLMWrapper

VOX_PROFILE_SR = 16000     

class VoxProfileAgeSexModel:
        
    def __init__(self, device: str = "cpu") -> None:
        
        self.device = device
        self.sample_rate = VOX_PROFILE_SR

        self.model = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-age-sex").to(device)
        self.model.eval()

    def __call__(self, audio: torch.Tensor, sr: int, return_embeddings: bool = False, lengths: torch.Tensor = None):
        
        audio = audio.to(self.device)

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sample_rate)
            if lengths is not None:
                lengths = lengths * int(self.sample_rate / sr)

        if lengths is None:
            lengths = torch.tensor([audio.shape[1]] * audio.shape[0]).to(self.device)

        age, sex, embedding = self.model(
            audio, return_feature=True, length=lengths
        )
        
        sex = sex.flip(-1)  # Flip to [male, female]

        if return_embeddings:
            return age, sex, embedding
        else:
            return age, sex
        
if __name__ == "__main__":
    model = VoxProfileAgeSexModel(device="cpu")
    dummy_audio = torch.randn(4, 16000 * 5)
    dummy_lengths = torch.tensor([16000 * 5, 16000 * 5, 16000 * 5, 16000 * 5])
    age, sex, embedding = model(dummy_audio, sr=16000, return_embeddings=True, lengths=None)
    print("Age shape:", age.shape)
    print("Sex shape:", sex.shape)
    print("Embedding shape:", embedding.shape)