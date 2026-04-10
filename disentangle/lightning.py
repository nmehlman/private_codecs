from math import sin
from typing import Any, Optional

import pytorch_lightning as pl # pyright: ignore[reportMissingImports]
import torch
import torch.nn.functional as F
from torch.autograd import Function

from disentangle.models import AdversarialClassifier, DisentanglementAE

def compute_difference_metric(emb_self_recon, emb_private):
    """Compute some metric between self-reconstructed and private embeddings."""
    metric = torch.norm(emb_self_recon - emb_private, p=2).item()/(torch.norm(emb_self_recon, p=2).item() + 1e-8)
    return metric

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_outputs): # type: ignore
        return -ctx.lambd * grad_outputs, None


def grl(x, lambd: float):
    return GradReverse.apply(x, lambd)


class EmotionDisentangleModule(pl.LightningModule):
    def __init__(
        self,
        codec_dim: int,
        latent_dim: int,
        enc_channels: list,
        dec_channels: list,
        emotion_dim: int,
        ae_kwargs: dict = {},
        adversarial_channels: list = [128, 128, 128],
        adversarial_kwargs: dict = {},
        adv_loss_weight: float = 1.0,
        learning_rate: float = 1e-3,
        adv_learning_rate: Optional[float] = None,
        adv_annealing_steps: int = 0,
        adv_update_factor: int = 1,
        weight_decay: float = 0,
        normalize_input: bool = True,
        dataset_stats: dict = {},
        use_adversarial: bool = True,
        lr_scheduling: bool = True,
        gradient_clip_val: float = 0.0,
        tau: float = 0.07,    ):
        super().__init__()

        self.ae = DisentanglementAE(
            codec_dim=codec_dim,
            latent_dim=latent_dim,
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            **ae_kwargs,
        )

        self.use_adversarial = use_adversarial
        self.adv_classifier: Optional[AdversarialClassifier]
        if self.use_adversarial:
            self.adv_classifier = AdversarialClassifier(
                input_dim=latent_dim,
                emotion_dim=emotion_dim,
                channels=adversarial_channels,
                tau=tau,
                **adversarial_kwargs,
            )
        else:
            self.adv_classifier = None

        self.learning_rate = learning_rate
        self.adv_learning_rate = adv_learning_rate
        self.weight_decay = weight_decay
        self.adv_loss_weight = adv_loss_weight
        self.adv_annealing_steps = adv_annealing_steps
        self.adv_update_factor = adv_update_factor
        self.normalize_input = normalize_input
        self.dataset_stats = dataset_stats
        self.lr_scheduling = lr_scheduling
        self.gradient_clip_val = gradient_clip_val

        if self.normalize_input:
            assert self.dataset_stats, "Dataset stats must be provided for normalization."
            
            mean = torch.tensor(self.dataset_stats["mean"])
            std = torch.tensor(self.dataset_stats["std"]).clamp_min(1e-6)

            assert mean.shape[0] == codec_dim, "Mean stats dimension does not match codec_dim."
            assert std.shape[0] == codec_dim, "Std stats dimension does not match codec_dim."
            
            self.register_buffer("ds_mean", mean.view(1, -1, 1))
            self.register_buffer("ds_std", std.view(1, -1, 1))

    def forward(self, x):
        
        if self.normalize_input:
            x = self._normalize(x)

        x_hat, z = self.ae(x)
        if self.normalize_input:
            x_hat = self._denormalize(x_hat)

        return x_hat, z

    def _normalize(self, x):
        mean = self.ds_mean  # (1, codec_dim, 1)
        std = self.ds_std  # (1, codec_dim, 1)
        return (x - mean) / std

    def _denormalize(self, x):
        assert self.dataset_stats, "Dataset stats must be provided for denormalization."
        mean = self.ds_mean  # (1, codec_dim, 1)
        std = self.ds_std  # (1, codec_dim, 1)
        return x * std + mean

    def _compute_adv_loss_weight(self):
        # Apply sine annealing for adversarial loss weight
        if self.global_step < self.adv_annealing_steps:
            frac = self.global_step / self.adv_annealing_steps
            return self.adv_loss_weight * sin(frac * (3.14159265 / 2)) ** 2
        else:
            return self.adv_loss_weight

    def _scheduler_t_max(self) -> int:
        max_epochs = self.trainer.max_epochs
        return max(max_epochs if max_epochs is not None else 1, 1)

    def _get_adv_classifier(self) -> AdversarialClassifier:
        if self.adv_classifier is None:
            raise RuntimeError("Adversarial classifier is disabled for this module.")
        return self.adv_classifier

    def _freeze(self, module: torch.nn.Module):
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)

    def _unfreeze(self, module: torch.nn.Module):
        module.train()
        for param in module.parameters():
            param.requires_grad_(True)

    def _check_nan(self, loss, loss_name):
        """Check if loss contains NaN and log warning."""
        if torch.isnan(loss):
            self.log(f"{loss_name}_is_nan", 1.0, on_step=True, on_epoch=False)
            return True
        return False

    def _compute_grad_norm(self, parameters):
        """Compute total gradient norm across parameters."""
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def _clip_gradients(self, parameters):
        """Apply gradient clipping if enabled."""
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip_val)
        
    def training_step(self, batch, batch_idx):
        
        self._freeze(self.ae)

        x, emotion_embs, _, lengths = batch
        B = x.size(0) # batch size
        
        _, opt_adv = self.optimizers() # type: ignore
        adv_classifier = self._get_adv_classifier()

        with torch.no_grad():
            _, z = self(x)

        self.toggle_optimizer(opt_adv)
        adv_logits = adv_classifier(z, emotion_embs, lengths)
        targets = torch.arange(adv_logits.size(0), device=adv_logits.device)
        adv_loss = F.cross_entropy(adv_logits, targets) 
        
        # Check for NaN
        self._check_nan(adv_loss, "train_adv_loss")

        adv_acc = torch.mean((adv_logits.argmax(dim=1) == targets).float())
    
        self.log_dict(
            {
                "train_adv_loss": adv_loss.detach(),
                "train_adv_acc": adv_acc.detach(),
            },
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        return adv_loss.detach()

    def validation_step(self, batch, batch_idx):
        x, emotion_embs, _, lengths = batch
        _, z = self(x)
        adv_classifier = self._get_adv_classifier()
        adv_logits = adv_classifier(z, emotion_embs, lengths)
        targets = torch.arange(adv_logits.size(0), device=adv_logits.device)
        adv_acc = torch.mean((adv_logits.argmax(dim=1) == targets).float())
        self.log(
            "val_adv_acc",
            adv_acc.detach(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def on_before_optimizer_step(self, optimizer):
            """Hook called before optimizer step - used for gradient logging in automatic optimization mode."""
            if not self.use_adversarial:
                # Log gradient norm for AE-only training (automatic optimization)
                grad_norm = self._compute_grad_norm(self.ae.parameters())
                self.log("grad_norm_autoencoder", grad_norm, on_step=True, on_epoch=False, sync_dist=True)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
            """Configure gradient clipping for automatic optimization mode."""
            if not self.use_adversarial and self.gradient_clip_val > 0:
                # Apply gradient clipping for AE-only training (automatic optimization)
                self.clip_gradients(optimizer, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")

    def load_state_dict(self, state_dict, strict=True):
        """Load only autoencoder weights from checkpoint, ignoring adversarial classifier."""
        # Filter state_dict to only include autoencoder keys
        ae_state_dict = {k: v for k, v in state_dict.items() if k.startswith("ae.")}
        
        # Load only the filtered state dict
        return super().load_state_dict(ae_state_dict, strict=False)

    def configure_optimizers(self) -> Any:
        
        opt_ae = torch.optim.Adam(
            self.ae.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        
        if not self.use_adversarial:
            # AE-only training: return single optimizer and scheduler
            if self.lr_scheduling:
                sched_ae = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt_ae, T_max=self._scheduler_t_max()
                )
                return {"optimizer": opt_ae, "lr_scheduler": sched_ae}
            else:
                return {"optimizer": opt_ae}
        
        else:
            # Adversarial training: return both optimizers and schedulers
            adv_classifier = self._get_adv_classifier()
            opt_adv = torch.optim.Adam(adv_classifier.parameters(), lr=self.adv_learning_rate if self.adv_learning_rate else self.learning_rate)
            
            # Store optimizer names for logging
            self.optimizer_names = ["autoencoder", "adversarial"]
            
            if self.lr_scheduling:
                sched_ae = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt_ae, T_max=self._scheduler_t_max()
                )
                sched_adv = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt_adv, T_max=self._scheduler_t_max()
                )
                return [
                    {"optimizer": opt_ae, "lr_scheduler": sched_ae},
                    {"optimizer": opt_adv, "lr_scheduler": sched_adv}
                ]
            else:
                return [opt_ae, opt_adv]

    def on_train_epoch_end(self):
        if not self.use_adversarial:
            # Automatic optimization handles scheduler stepping
            return
        
        schedulers = self.lr_schedulers()
        
        if isinstance(schedulers, (list, tuple)):
            for i, scheduler in enumerate(schedulers):
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(0.0)
                else:
                    scheduler.step()
                # Use stored optimizer names for logging
                name = self.optimizer_names[i] if hasattr(self, 'optimizer_names') and i < len(self.optimizer_names) else f"optimizer_{i}"
                self.log(f"lr_{name}", scheduler.get_last_lr()[0], on_epoch=True, sync_dist=True)
        elif schedulers is not None:
            if isinstance(schedulers, torch.optim.lr_scheduler.ReduceLROnPlateau):
                schedulers.step(0.0)
            else:
                schedulers.step()
            self.log("lr_scheduler", schedulers.get_last_lr()[0], on_epoch=True, sync_dist=True)
        
