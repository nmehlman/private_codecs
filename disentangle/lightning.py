from math import sin

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torchmetrics import Accuracy

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
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grl(x, lambd: float):
    return GradReverse.apply(x, lambd)


class EmotionDisentangleModule(pl.LightningModule):
    def __init__(
        self,
        codec_dim: int,
        latent_dim: int,
        enc_channels: list,
        dec_channels: list,
        conditioning_dim: int,
        conditioning_dropout: float = 0.0,
        num_emotion_classes: int = 9,
        ae_kwargs: dict = {},
        adversarial_channels: list = [128, 128, 128],
        adversarial_kwargs: dict = {},
        adv_loss_weight: float = 1.0,
        learning_rate: float = 1e-3,
        adv_annealing_steps: int = 0,
        adv_update_factor: int = 1,
        weight_decay: float = 0,
        normalize_input: bool = True,
        dataset_stats: dict = {},
        use_adversarial: bool = True,
        lr_scheduling: bool = True,
        gradient_clip_val: float = 0.0,
        emotion_utilization_weight: float = 0.0,
        emotion_utilization_margin: float = 0.0,
    ):
        super().__init__()

        self.ae = DisentanglementAE(
            codec_dim=codec_dim,
            latent_dim=latent_dim,
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            conditioning_dim=conditioning_dim,
            conditioning_dropout=conditioning_dropout,
            **ae_kwargs,
        )

        self.use_adversarial = use_adversarial
        if self.use_adversarial:
            self.adv_classifier = AdversarialClassifier(
                input_dim=latent_dim,
                num_classes=num_emotion_classes,
                channels=adversarial_channels,
                **adversarial_kwargs,
            )
        else:
            self.adv_classifier = None

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adv_loss_weight = adv_loss_weight
        self.automatic_optimization = not use_adversarial  # Use automatic optimization when no adversarial training
        self.adv_annealing_steps = adv_annealing_steps
        self.adv_update_factor = adv_update_factor
        self.normalize_input = normalize_input
        self.dataset_stats = dataset_stats
        self.lr_scheduling = lr_scheduling
        self.gradient_clip_val = gradient_clip_val
        self.emotion_utilization_weight = emotion_utilization_weight
        self.emotion_utilization_margin = emotion_utilization_margin
        
        if self.use_adversarial:
            self.train_accuracy = Accuracy(task="multiclass", num_classes=num_emotion_classes)
            self.validation_accuracy = Accuracy(task="multiclass", num_classes=num_emotion_classes)

        if self.normalize_input:
            assert self.dataset_stats, "Dataset stats must be provided for normalization."
            
            mean = torch.tensor(self.dataset_stats["mean"])
            std = torch.tensor(self.dataset_stats["std"]).clamp_min(1e-6)

            assert mean.shape[0] == codec_dim, "Mean stats dimension does not match codec_dim."
            assert std.shape[0] == codec_dim, "Std stats dimension does not match codec_dim."
            
            self.register_buffer("ds_mean", mean.view(1, -1, 1))
            self.register_buffer("ds_std", std.view(1, -1, 1))


    

    def forward(self, x, cond):
        if self.normalize_input:
            x = self._normalize(x)

        x_hat, z = self.ae(x, cond)

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
        x, emotion_emb, emotion_labs, lengths = batch
        
        if not self.use_adversarial:
            # AE-only training with automatic optimization
            x_hat, z = self(x, emotion_emb)
            recon_loss = F.mse_loss(x_hat, x)
            
            total_loss = recon_loss

            # Check for NaN
            self._check_nan(recon_loss, "train_recon_loss")
            self._check_nan(total_loss, "train_total_loss")
            
            self.log_dict(
                {
                    "train_recon_loss": recon_loss.detach(),
                    "train_total_loss": total_loss.detach(),
                },
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            
            return total_loss
        
        else:
        
            # Adversarial training with manual optimization
            opt_ae, opt_adv = self.optimizers()

            with torch.no_grad():
                _, z = self(x, emotion_emb)

            # Adversary update steps
            for _ in range(self.adv_update_factor):
                self.toggle_optimizer(opt_adv)
                adv_logits = self.adv_classifier(z, lengths)
                adv_loss = F.cross_entropy(adv_logits, emotion_labs)
                
                # Check for NaN
                self._check_nan(adv_loss, "train_adv_loss")
                
                opt_adv.zero_grad()
                self.manual_backward(adv_loss)
                
                # Compute and log gradient norm
                grad_norm_adv = self._compute_grad_norm(self.adv_classifier.parameters())
                self.log("grad_norm_adversarial", grad_norm_adv, on_step=True, on_epoch=False, sync_dist=True)
                
                # Apply gradient clipping
                self._clip_gradients(self.adv_classifier.parameters())
                
                opt_adv.step()
                opt_adv.zero_grad(set_to_none=True)
                self.untoggle_optimizer(opt_adv)
            
            adv_acc = self.train_accuracy(adv_logits, emotion_labs)

            # AE update step
            self._freeze(self.adv_classifier)
            self.toggle_optimizer(opt_ae)
            x_hat, z = self(x, emotion_emb)
            adv_loss_weight = self._compute_adv_loss_weight()
            fool_logits = self.adv_classifier(grl(z, adv_loss_weight), lengths)

            recon_loss = F.mse_loss(x_hat, x)
            fool_loss = F.cross_entropy(fool_logits, emotion_labs)

            ae_loss = recon_loss + fool_loss
            
            # Check for NaN
            self._check_nan(recon_loss, "train_recon_loss")
            self._check_nan(fool_loss, "train_fool_loss")
            self._check_nan(ae_loss, "train_ae_loss")

            opt_ae.zero_grad()
            self.manual_backward(ae_loss)
            
            # Compute and log gradient norm
            grad_norm_ae = self._compute_grad_norm(self.ae.parameters())
            self.log("grad_norm_autoencoder", grad_norm_ae, on_step=True, on_epoch=False, sync_dist=True)
            
            # Apply gradient clipping
            self._clip_gradients(self.ae.parameters())
            
            opt_ae.step()
            self.untoggle_optimizer(opt_ae)
            self._unfreeze(self.adv_classifier)

            self.log_dict(
                {
                    "train_recon_loss": recon_loss.detach(),
                    "train_adv_loss": adv_loss.detach(),
                    "train_ae_loss": ae_loss.detach(),
                    "train_adv_acc": adv_acc.detach(),
                    "train_fool_loss": fool_loss.detach(),
                    "train_total_loss": ae_loss.detach(),
                },
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )

            return ae_loss.detach()

    def validation_step(self, batch, batch_idx):
        x, emotion_emb, emotion_labs, lengths = batch
        x_hat, z = self(x, emotion_emb)
        recon_loss = F.mse_loss(x_hat, x)
        self.log(
            "val_recon_loss",
            recon_loss.detach(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        if self.use_adversarial:
            adv_logits = self.adv_classifier(z, lengths)
            adv_acc = self.validation_accuracy(adv_logits, emotion_labs)
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

    def configure_optimizers(self):
        
        opt_ae = torch.optim.Adam(
            self.ae.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        
        if not self.use_adversarial:
            # AE-only training: return single optimizer and scheduler
            if self.lr_scheduling:
                sched_ae = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt_ae, T_max=max(self.trainer.max_epochs, 1)
                )
                return {"optimizer": opt_ae, "lr_scheduler": sched_ae}
            else:
                return {"optimizer": opt_ae}
        
        else:
            # Adversarial training: return both optimizers and schedulers
            opt_adv = torch.optim.Adam(self.adv_classifier.parameters(), lr=self.learning_rate)
            
            # Store optimizer names for logging
            self.optimizer_names = ["autoencoder", "adversarial"]
            
            if self.lr_scheduling:
                sched_ae = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt_ae, T_max=max(self.trainer.max_epochs, 1)
                )
                sched_adv = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt_adv, T_max=max(self.trainer.max_epochs, 1)
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
                scheduler.step()
                # Use stored optimizer names for logging
                name = self.optimizer_names[i] if hasattr(self, 'optimizer_names') and i < len(self.optimizer_names) else f"optimizer_{i}"
                self.log(f"lr_{name}", scheduler.get_last_lr()[0], on_epoch=True, sync_dist=True)
        elif schedulers is not None:
            schedulers.step()
            self.log("lr_scheduler", schedulers.get_last_lr()[0], on_epoch=True, sync_dist=True)
            
            
        
