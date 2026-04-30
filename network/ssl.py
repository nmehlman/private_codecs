from torch import nn
from transformers import Wav2Vec2FeatureExtractor
from transformers import WavLMModel


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

    def forward(self, x, sampling_rate=16000):
        """Return per-sample embeddings of shape (B, D).

        x: either a list of 1D arrays/tensors (cpu) or a tensor of shape (B, T)
        """
        # Prepare inputs via processor when available
        if self.processor is not None:
            # processor can handle list of numpy arrays or tensors
            inputs = self.processor(x, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
            input_values = inputs["input_values"]
            attention_mask = inputs.get("attention_mask", None)
            if self.device is not None:
                input_values = input_values.to(self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

            outputs = self.backbone_model(input_values, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state
            if attention_mask is not None:
                # compute masked mean over time dim
                mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
                summed = (last_hidden * mask).sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / lengths
            else:
                pooled = last_hidden.mean(dim=1)

            return pooled

        else:
            # assume x is a tensor (B, T)
            if self.device is not None:
                x = x.to(self.device)
            outputs = self.backbone_model(x)
            last_hidden = outputs.last_hidden_state
            pooled = last_hidden.mean(dim=1)
            return pooled

    # kept for compatibility if other code calls it
    def get_feat_extract_output_lengths(self, input_length):
        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1
        cfg = getattr(self.backbone_model.config, "conv_kernel", None)
        stride = getattr(self.backbone_model.config, "conv_stride", None)
        if cfg is None or stride is None:
            return input_length
        for kernel_size, s in zip(cfg, stride):
            input_length = _conv_out_length(input_length, kernel_size, s)
        return input_length
    
if __name__ == "__main__":

    import torch

    model = WavLMWrapper(pretrain_model="wavlm_large", device="cuda")
    dummy_audio = torch.randn(4, 16000 * 5)
    embeddings = model(dummy_audio, sampling_rate=16000)
    print("Embeddings shape:", embeddings.shape)