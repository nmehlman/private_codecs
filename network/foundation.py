from torch import nn
from transformers import Wav2Vec2FeatureExtractor
from transformers import WavLMModel
from speechbrain.lobes.models.huggingface_transformers.huggingface import make_padding_masks

import copy
import torch
import loralib as lora
import transformers.models.whisper.modeling_whisper as whisper

from torch import nn
from transformers.activations import ACT2FN
from huggingface_hub import PyTorchModelHubMixin
from transformers import WhisperModel, AutoFeatureExtractor


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
            length = length.to(self.device)

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
        
        return z[-1] # Last layer only 

class WhisperWrapper(nn.Module):
    """Simple Whisper wrapper that loads a pretrained Whisper backbone and returns
    the last encoder hidden states (no finetuning, pretrained inference only).

    Usage:
      model = WhisperWrapper(pretrain_model='whisper_base')
      embeddings = model(batch_waveforms)

    Accepts either a list of 1D arrays/tensors (cpu) or a tensor of shape (B, T).
    """
    def __init__(self, pretrain_model="whisper_base", device=None):
        super().__init__()
        self.pretrain_model = pretrain_model
        self.device = device

        # choose model id
        if self.pretrain_model == "whisper_tiny":
            model_id = "openai/whisper-tiny"
            feat_id = "openai/whisper-tiny"
        elif self.pretrain_model == "whisper_base":
            model_id = "openai/whisper-base"
            feat_id = "openai/whisper-base"
        elif self.pretrain_model == "whisper_small":
            model_id = "openai/whisper-small"
            feat_id = "openai/whisper-small"
        elif self.pretrain_model == "whisper_medium":
            model_id = "openai/whisper-medium"
            feat_id = "openai/whisper-medium"
        else:
            model_id = "openai/whisper-large-v3"
            feat_id = "openai/whisper-large-v3"

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feat_id)
        self.backbone_model = WhisperModel.from_pretrained(model_id, output_hidden_states=True)

        if self.device is not None:
            self.to(self.device)

        # pretrained inference mode
        self.backbone_model.eval()
        for p in self.backbone_model.parameters():
            p.requires_grad = False

    def forward(self, x, length=None):
        """Return last encoder hidden states: Tensor of shape (B, T', D).

        x: list of 1D arrays/tensors (cpu) or tensor of shape (B, T)
        """
        # prepare list of numpy arrays for the feature extractor
        new_x = []
        if isinstance(x, torch.Tensor):
            for idx in range(x.shape[0]):
                new_x.append(x[idx].detach().cpu().numpy())
        else:
            for idx in range(len(x)):
                # assume already CPU numpy or tensor
                xi = x[idx]
                if torch.is_tensor(xi):
                    new_x.append(xi.detach().cpu().numpy())
                else:
                    new_x.append(xi)

        features = self.feature_extractor(new_x, return_tensors="pt", sampling_rate=16000, padding=True)
        inputs = features.input_features.to(self.device) if self.device is not None else features.input_features

        with torch.no_grad():
            z = self.backbone_model(inputs, output_hidden_states=True).encoder_hidden_states

        # return last encoder layer hidden states
        return z[-1]
    
if __name__ == "__main__":

    import torch

    model = WavLMWrapper(pretrain_model="wavlm_large", device="cuda")
    dummy_audio = torch.randn(4, 16000 * 5, device="cuda")
    embeddings = model(dummy_audio)
    print("Embeddings shape:", embeddings.shape)
