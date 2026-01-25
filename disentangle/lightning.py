import pytorch_lightning as pl
from disentangle.models import DisentanglementAE, AdversarialClassifier
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy

class EmotionDisentangleModule(pl.LightningModule):
    def __init__(self, 
                 codec_dim: int, 
                 latent_dim: int, 
                 enc_channels: list,
                 dec_channels: list, 
                 conditioning_dim: int,
                 ae_kwargs: dict,
                 num_emotion_classes: int,
                 adversarial_kwargs: dict,
                 adv_loss_weight: float = 1.0,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 0):
        
        super(EmotionDisentangleModule, self).__init__()
        
        self.ae = DisentanglementAE(
            codec_dim=codec_dim,
            latent_dim=latent_dim,
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            conditioning_dim=conditioning_dim,
            **ae_kwargs
        )

        self.adv_classifier = AdversarialClassifier(input_dim=latent_dim,
                                                    num_classes=num_emotion_classes,
                                                    **adversarial_kwargs)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adv_loss_weight = adv_loss_weight
        self.automatic_optimization = False

        self.accuracy = Accuracy(task="multiclass", num_classes=num_emotion_classes)

    def forward(self, x, cond):
        return self.ae(x, cond)

    def training_step(self, batch, batch_idx):
        
        x, emotion_emb, emotion_labs, lengths = batch

        opt_ae, opt_adv = self.optimizers()
        
        x_hat, z = self(x, emotion_emb)
        recon_loss = F.mse_loss(x_hat, x)

        # Train adversarial classifier to predict the emotions from the latent code
        self.toggle_optimizer(opt_adv)
        adv_logits = self.adv_classifier(z.detach(), lengths)
        adv_acc = self.accuracy(adv_logits, emotion_labs)
        adv_loss = F.cross_entropy(adv_logits, emotion_labs)
        self.manual_backward(adv_loss)
        opt_adv.step()
        opt_adv.zero_grad()
        self.untoggle_optimizer(opt_adv)

        # Train AE to reconstruct and simultaneously confuse the adversary
        self.toggle_optimizer(opt_ae)
        fool_logits = self.adv_classifier(z, lengths)
        fool_loss = F.cross_entropy(fool_logits, emotion_labs)
        ae_loss = recon_loss - self.adv_loss_weight * fool_loss
        self.manual_backward(ae_loss)
        opt_ae.step()
        opt_ae.zero_grad()
        self.untoggle_optimizer(opt_ae)

        self.log_dict(
            {
                'train_recon_loss': recon_loss.detach(),
                'train_adv_loss': adv_loss.detach(),
                'train_ae_loss': ae_loss.detach(),
                'train_adv_acc': adv_acc.detach(),
                'train_fool_loss': fool_loss.detach(),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        return ae_loss.detach()


    def validation_step(self, batch, batch_idx):
        
        x, emotion_emb, emotion_labs, lengths = batch
        x_hat, z = self(x, emotion_emb)
        recon_loss = F.mse_loss(x_hat, x)
        self.log('val_recon_loss', recon_loss.detach(), prog_bar=True, on_step=False, on_epoch=True)

        adv_logits = self.adv_classifier(z, lengths)
        adv_acc = self.accuracy(adv_logits, emotion_labs)
        self.log('val_adv_acc', adv_acc.detach(), prog_bar=True, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(self.ae.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        opt_adv = torch.optim.Adam(self.adv_classifier.parameters(), lr=self.learning_rate)
        return [opt_ae, opt_adv]
