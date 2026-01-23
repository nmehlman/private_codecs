import sys
sys.path.append('../../pytorch-tcn')  # TODO: add to path
from pytorch_tcn.tcn import TCN # TODO: add to path
import torch
import torch.nn as nn

class DisentanglementAE(nn.Module):
    
    def __init__(self, codec_dim, latent_dim, enc_channels, dec_channels, conditioning_dim: int, kernel_size=3, dropout=0.2, **kwargs):
        
        super(DisentanglementAE, self).__init__()
        
        self.encoder = TCN(codec_dim, enc_channels + [latent_dim], kernel_size=kernel_size, dropout=dropout, **kwargs)
        self.decoder = TCN(latent_dim, dec_channels + [codec_dim], kernel_size=kernel_size, dropout=dropout, conditioning_dim=conditioning_dim, **kwargs)

    def forward(self, codec_output, emotion_embed):
        z = self.encoder(codec_output)
        x_hat = self.decoder(z, conditioning=emotion_embed)
        
        return x_hat, z

if __name__ == "__main__":

    codec_dim = 128
    batch_size = 8
    seq_len = 128
    latent_dim = 16
    conditioning_dim = 32

    num_channels_enc = [128, 64, latent_dim]
    num_channels_dec = [latent_dim, 64, codec_dim]

    ae = DisentanglementAE(codec_dim, latent_dim, num_channels_enc, num_channels_dec, conditioning_dim)

    x = torch.randn(batch_size, seq_len, codec_dim)
    cond = torch.randn(batch_size, conditioning_dim) 

    x_hat, z = ae(x, cond)

    print(f"Input shape: {x.shape}")
    print(f"Conditioning shape: {cond.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Output shape: {x_hat.shape}")