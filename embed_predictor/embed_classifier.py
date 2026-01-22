from pytorch_tcn import TCN
import torch
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, F1Score

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

class EmbeddingClassifier(LightningModule):
    def __init__(self, input_dim: int, num_classes: int, lr: float = 1e-3, **kwargs):
        
        super(EmbeddingClassifier, self).__init__()
        
        self.encoder = TCN(
            num_inputs=input_dim,
            num_channels=[64, 64, 64],
            kernel_size=3,
            use_norm="layer_norm",
            dropout=0.2,
            causal=False,
            use_skip_connections=True,
            output_projection=None,
            **kwargs
        )

        self.lr = lr

        self.pooling = AttentionPooling(input_dim=64)
        self.fc = torch.nn.Linear(64, num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        
        x, y, lengths = batch
        logits = self.forward(x, lengths)
        
        loss = torch.nn.functional.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        f1 = self.f1(logits, y)
        
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.log("train_f1", f1)

        return loss
    
    def validation_step(self, batch, batch_idx):
        
        x, y, lengths = batch
        logits = self.forward(x, lengths)
        
        loss = torch.nn.functional.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        f1 = self.f1(logits, y)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)