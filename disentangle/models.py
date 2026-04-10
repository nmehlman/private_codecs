import sys
from pytorch_tcn.tcn import TCN 
import torch
import torch.nn as nn

class DisentanglementAE(nn.Module):
    
    def __init__(self, 
                 codec_dim: int, 
                 latent_dim: int, 
                 enc_channels: list,
                 dec_channels: list, 
                 conditioning_dim: int, 
                 conditioning_dropout: float = 0.0, 
                 **kwargs):
        
        super(DisentanglementAE, self).__init__()
        
        self.encoder = TCN(codec_dim, enc_channels + [latent_dim], causal=False, **kwargs)
        self.decoder = TCN(latent_dim, dec_channels + [codec_dim], conditioning_dim=conditioning_dim, causal=False, disable_final_activation=True, **kwargs)

        self.conditioning_dropout = conditioning_dropout

    def forward(self, codec_output: torch.Tensor, emotion_embed: torch.Tensor) -> tuple:
        
        z = self.encoder(codec_output)
        
        if self.training and self.conditioning_dropout > 0: # Randomly shuffle conditioning embeddings
            b = emotion_embed.size(0)
            do_swap = (torch.rand(b, device=emotion_embed.device) < self.conditioning_dropout)
            perm = torch.randperm(b, device=emotion_embed.device)
            swapped = emotion_embed[perm]
            emotion_embed = torch.where(do_swap[:, None], swapped, emotion_embed)
        
        x_hat = self.decoder(z, conditioning=emotion_embed)
        return x_hat, z
    
class AttentionPooling(torch.nn.Module):
    def __init__(self, input_dim: int):
        super(AttentionPooling, self).__init__()
        self.score = torch.nn.Linear(input_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, T, D)
        scores = self.score(x).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weighted_sum = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return weighted_sum

class AdversarialClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, channels: list = [64, 64, 64], **kwargs):
        
        super(AdversarialClassifier, self).__init__()
        
        self.encoder = TCN(
            num_inputs=input_dim,
            num_channels=channels,
            kernel_size=3,
            use_norm="layer_norm",
            dropout=0.2,
            causal=False,
            use_skip_connections=True,
            output_projection=None,
            **kwargs
        )

        self.pooling = AttentionPooling(input_dim=channels[-1])
        self.fc = torch.nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        
        T = x.size(2)
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        assert torch.isnan(mask).sum() == 0

        encoded = self.encoder(x)
        assert encoded.dim() == 3
        assert torch.isnan(encoded).sum() == 0

        pooled = self.pooling(encoded, mask)
        assert torch.isnan(pooled).sum() == 0

        logits = self.fc(pooled)

        return logits


if __name__ == "__main__":

    codec_dim = 128
    batch_size = 8
    seq_len = 512
    latent_dim = 16
    conditioning_dim = 32

    num_channels_enc = [128]
    num_channels_dec = [128]

    ae = DisentanglementAE(codec_dim, latent_dim, num_channels_enc, num_channels_dec, conditioning_dim)

    x = torch.randn(batch_size, codec_dim, seq_len)
    cond = torch.randn(batch_size, conditioning_dim) 

    x_hat, z = ae(x, cond)

    print(f"Input shape: {x.shape}")
    print(f"Conditioning shape: {cond.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Output shape: {x_hat.shape}")