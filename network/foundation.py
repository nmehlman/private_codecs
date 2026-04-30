from torch import nn
from transformers import Wav2Vec2FeatureExtractor
from transformers import WavLMModel
from speechbrain.lobes.models.huggingface_transformers.huggingface import make_padding_masks


class WavLMWrapper(nn.Module):
    """Simple WavLM wrapper that loads a pretrained model and returns pooled embeddings.

    Usage:
      model = WavLMWrapper(pretrain_model='wavlm_large')
      embeddings = model(batch_waveforms)

    The wrapper accepts either a list of 1D arrays/tensors or a batched 2D tensor.
    """
    def __init__(
        self,
        pretrain_model="wavlm_large",
        device=None,
    ):
        super().__init__()
        self.pretrain_model = pretrain_model
        self.device = device

        if self.pretrain_model == "wavlm":
            self.backbone_model = WavLMModel.from_pretrained(
                "microsoft/wavlm-base-plus",
                output_hidden_states=False,
            )
            self.processor = None
        else:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
            self.backbone_model = WavLMModel.from_pretrained(
                "microsoft/wavlm-large",
                output_hidden_states=False,
            )

        if self.device is not None:
            self.to(self.device)

        # default to eval and no grad (pretrained inference)
        self.backbone_model.eval()
        for p in self.backbone_model.parameters():
            p.requires_grad = False

    def forward(self, x, length=None):
        """Return per-sample embeddings of shape (B, D).

        x: either a list of 1D arrays/tensors (cpu) or a tensor of shape (B, T)
        """
        if self.pretrain_model == "wavlm_large":  
            with torch.no_grad():
                signal, attention_mask = list(), list()
                if length is not None: attention_mask = make_padding_masks(x, wav_len=length/length.max()).to(x.device)
                else: attention_mask = make_padding_masks(x, wav_len=torch.tensor([1]*len(x)).to(x.device)).to(x.device)

                for idx in range(len(x)):
                    input = self.processor(x[idx], sampling_rate=16_000, return_tensors="pt", padding=True)
                    signal.append(input["input_values"][0].to(x.device))
                signal = torch.stack(signal)
        
        # 2. get length and mask
        if length is not None:
            length = self.get_feat_extract_output_lengths(length.detach().cpu())
            length = length.cuda()

        if self.pretrain_model == "wavlm": 
            z = self.backbone_model(
                x, 
                output_hidden_states=True
            ).hidden_states
        else: 
            z = self.backbone_model(
                signal, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            ).hidden_states
        
        return z
        
    
if __name__ == "__main__":

    import torch

    model = WavLMWrapper(pretrain_model="wavlm_large", device="cuda")
    dummy_audio = torch.randn(4, 16000 * 5)
    embeddings = model(dummy_audio)
    print("Embeddings shape:", embeddings.shape)
