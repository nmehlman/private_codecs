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

    correct_embedding_recon = ae.decoder(z, emotion_embed)
    zeros_embedding_recon = ae.decoder(z, torch.zeros_like(emotion_embed))  
    scaled_embedding_recon = ae.decoder(z, alpha * emotion_embed)
    perturned_embedding_recon = ae.decoder(z, emotion_embed + delta * torch.randn_like(emotion_embed))

    diff_zeros = compute_difference_metric(correct_embedding_recon, zeros_embedding_recon)
    diff_scaled = compute_difference_metric(correct_embedding_recon, scaled_embedding_recon)
    diff_perturbed = compute_difference_metric(correct_embedding_recon, perturned_embedding_recon)

    return {
        'diff_zeros': diff_zeros,
        'diff_scaled': diff_scaled,
        'diff_perturbed': diff_perturbed
    }





