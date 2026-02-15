import torch
from pytorch_lightning import LightningModule
from disentangle.lightning import compute_difference_metric

def compute_conditioning_ablation(
        pl_model: LightningModule,
        embedding: torch.Tensor,
        emotion_label: torch.Tensor,
        delta: float = 1e-2,
        alpha: float = 10.0
    ):

    ae = pl_model.ae

        
    z = ae.encoder(embedding)
    
    emotion_embed = ae.embedding_table(emotion_label)
    emotion_embed = emotion_embed.unsqueeze(-1).expand(-1, -1, z.size(2))  # (B, C, T)
        
    x_hat = ae.decoder(torch.cat([z, emotion_embed], dim=1))

    correct_embedding_recon = ae.decoder(torch.cat([z, emotion_embed], dim=1))
    zeros_embedding_recon = ae.decoder(torch.cat([z, torch.zeros_like(emotion_embed)], dim=1))  
    scaled_embedding_recon = ae.decoder(torch.cat([z, alpha * emotion_embed], dim=1))
    perturned_embedding_recon = ae.decoder(torch.cat([z, emotion_embed + delta * torch.randn_like(emotion_embed)], dim=1))

    diff_zeros = compute_difference_metric(correct_embedding_recon, zeros_embedding_recon)
    diff_scaled = compute_difference_metric(correct_embedding_recon, scaled_embedding_recon)
    diff_perturbed = compute_difference_metric(correct_embedding_recon, perturned_embedding_recon)

    return {
        'diff_zeros': diff_zeros,
        'diff_scaled': diff_scaled,
        'diff_perturbed': diff_perturbed
    }





