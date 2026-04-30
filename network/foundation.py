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
    def __init__(self, pretrain_model="whisper_large", device=None):
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
        self.backbone_model = WhisperModel.from_pretrained(model_id, output_hidden_states=True, ignore_mismatched_sizes=True,
                max_source_positions=750)
        self.embed_positions = copy.deepcopy(self.backbone_model.encoder.embed_positions.weight)
        self.embed_positions.requires_grad = False

        if self.device is not None:
            self.to(self.device)

        # pretrained inference mode
        self.backbone_model.eval()
        for p in self.backbone_model.parameters():
            p.requires_grad = False

    def forward(self, x, length=None):
        """Return last encoder hidden states: Tensor of shape (B, T', D).

        x: list of 1D arrays/tensors (cpu) or tensor of shape (B, T)
        length: optional sequence lengths for attention mask
        """
        # 1. feature extraction and projections
        if length is not None:
            max_audio_len = length.max().item()
            # Append to list for feature_extractor to work
            new_x = list()
            for idx in range(len(length)):
                new_x.append(x[idx].detach().cpu().numpy())
            
            # Max length is max audio len in a batch
            features = self.feature_extractor(
                new_x,
                return_tensors="pt", 
                sampling_rate=16000,
                max_length=max_audio_len
            )
            features = features.input_features.to(x.device)
        else:
            max_audio_len = 15*16000
            new_x = list()
            for idx in range(x.shape[0]):
                new_x.append(x[idx].detach().cpu().numpy())
            
            # Max length is max audio len in a batch
            features = self.feature_extractor(
                new_x,
                return_tensors="pt", 
                sampling_rate=16000,
                max_length=max_audio_len
            )
            features = features.input_features.to(x.device)
        
        # pdb.set_trace()
        # 2. get length and mask
        if length is not None:
            length = self._get_feat_extract_output_lengths(length.detach().cpu())
            # Replace positional embeddings
            self.backbone_model.encoder.embed_positions = self.backbone_model.encoder.embed_positions.from_pretrained(self.embed_positions[:750])
        else:
            # Replace positional embeddings
            length = torch.tensor([len(x[0])])
            length = self._get_feat_extract_output_lengths(length)
            self.backbone_model.encoder.embed_positions = self.backbone_model.encoder.embed_positions.from_pretrained(self.embed_positions[:750])
            
        # 3. transformer encoding features
        # compute reduced attention_mask corresponding to feature vectors
        features = self.backbone_model.encoder(
            features, output_hidden_states=True
        ).hidden_states

        features = torch.stack(features, dim=0)[-1]

        return features

    def _get_feat_extract_output_lengths(self, input_lengths):
        """Computes the output length of the convolutional layers"""
        input_lengths = input_lengths // 160
        input_lengths = (input_lengths - 1) // 2 + 1
        return input_lengths
    
if __name__ == "__main__":

    import torch

    device = 'cpu'

    model = WhisperWrapper(pretrain_model="whisper_large", device=device)
    dummy_audio = torch.randn(4, 16000 * 5, device=device)
    embeddings = model(dummy_audio)
    print("Embeddings shape:", embeddings.shape)
