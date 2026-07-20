"""All-in-one training script for the contrastive adversary ONLY.

Trains an AdversarialClassifier to align pooled latent representations with
sex embeddings via the in-batch contrastive loss. No autoencoder training,
no GRL, no audio eval.

By default the adversary is trained directly on (normalized) codec features.
If `ae_ckpt` is set in the config, a frozen pretrained SexDisentangleModule
is loaded and the adversary is trained on its latents instead.

Usage:
    python train_adversary.py --config configs/train/adversary_only.yaml
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
import tqdm
import yaml
from torch.utils.tensorboard import SummaryWriter

from disentangle.codec_data import get_dataloaders
from disentangle.lightning import SexDisentangleModule
from disentangle.misc.utils import load_dataset_stats
from disentangle.models import AdversarialClassifier


def compute_contrastive_loss(adv_logits, embeddings, soft_loss_weight=0.0, tau_st=0.07):
    targets = torch.arange(adv_logits.size(0), device=adv_logits.device)
    ce_loss = F.cross_entropy(adv_logits, targets)

    if soft_loss_weight > 0:
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        soft_targets = (torch.inner(embeddings_norm, embeddings_norm) / tau_st).softmax(dim=1)
        log_probs = F.log_softmax(adv_logits, dim=1)
        soft_loss = F.kl_div(log_probs, soft_targets, reduction="batchmean")
        return (1 - soft_loss_weight) * ce_loss + soft_loss_weight * soft_loss

    return ce_loss


def main():

    parser = argparse.ArgumentParser(description="Contrastive adversary-only training script")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.get("gpus", "0")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(config.get("random_seed", 42))

    # Setup logging/checkpoint directory
    log_dir = config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    writer = SummaryWriter(log_dir=log_dir)

    # Setup dataloaders (same format as main train script)
    train_val_spks_split_file = config["dataset"].pop("train_val_spks_split_file", None)
    if train_val_spks_split_file:
        with open(train_val_spks_split_file, "r") as f:
            train_val_spks = json.load(f)
    else:
        train_val_spks = None

    dataloaders = get_dataloaders(
        train_val_spk=train_val_spks,
        dataset_kwargs=config["dataset"],
        **config["dataloader"],
    )
    train_loader, val_loader = dataloaders["train"], dataloaders["val"]

    # Load dataset stats for input normalization
    stats = load_dataset_stats(config["dataset_name"], config["codec_name"], config["input_type"])
    ds_mean = torch.tensor(stats["mean"]).view(1, -1, 1).to(device)
    ds_std = torch.tensor(stats["std"]).clamp_min(1e-6).view(1, -1, 1).to(device)

    model_cfg = config["model"]
    normalize_input = model_cfg.get("normalize_input", True)

    # Optionally load a frozen pretrained AE to train the adversary on its latents
    ae_ckpt = config.get("ae_ckpt", None)
    if ae_ckpt:
        pl_model = SexDisentangleModule.load_from_checkpoint(
            ae_ckpt, dataset_stats=stats, **config["model"]
        ).to(device).eval()
        for p in pl_model.parameters():
            p.requires_grad_(False)
        adv_input_dim = pl_model.hparams["latent_dim"]
        print(f"Loaded frozen AE from {ae_ckpt}; training adversary on latents (dim={adv_input_dim})")
    else:
        pl_model = None
        adv_input_dim = model_cfg["codec_dim"]
        print(f"No AE checkpoint given; training adversary directly on codec features (dim={adv_input_dim})")

    if config['dataset'].get("use_vg_sex_embeddings", False):
        embedding_dim = 192
    else:
        embedding_dim = model_cfg["emotion_dim"]
    adversary = AdversarialClassifier(
        input_dim=adv_input_dim,
        emotion_dim=embedding_dim,
        channels=model_cfg.get("adversarial_channels", [128, 128, 128]),
        tau=model_cfg.get("tau_cl", 0.07),
        **model_cfg.get("adversarial_kwargs", {}),
    ).to(device)

    soft_loss_weight = model_cfg.get("soft_loss_weight", 0.0)
    tau_st = model_cfg.get("tau_st", 0.07)
    gradient_clip_val = model_cfg.get("gradient_clip_val", 0.0)

    optimizer = torch.optim.Adam(
        adversary.parameters(),
        lr=model_cfg.get("learning_rate", 1e-3),
        weight_decay=model_cfg.get("weight_decay", 0.0),
    )

    max_epochs = config.get("max_epochs", 100)
    scheduler = None
    if model_cfg.get("lr_scheduling", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    def get_adversary_input(x):
        """Normalize codec features and (optionally) map through the frozen AE encoder."""
        if normalize_input:
            x = (x - ds_mean) / ds_std
        if pl_model is not None:
            with torch.no_grad():
                z = pl_model.ae.encoder(x)
            return z
        return x

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(max_epochs):

        # Training
        adversary.train()
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            x, _, embeddings, lengths = batch
            x = x.to(device)
            embeddings = embeddings.to(device)
            lengths = lengths.to(device)

            z = get_adversary_input(x)
            logits = adversary(z, embeddings, lengths)
            loss = compute_contrastive_loss(logits, embeddings, soft_loss_weight, tau_st)

            optimizer.zero_grad()
            loss.backward()
            if gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(adversary.parameters(), gradient_clip_val)
            optimizer.step()

            targets = torch.arange(logits.size(0), device=device)
            acc = (logits.argmax(dim=1) == targets).float().mean()

            writer.add_scalar("train/adv_loss", loss.item(), global_step)
            writer.add_scalar("train/adv_acc", acc.item(), global_step)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.3f}")
            global_step += 1

        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        # Validation
        adversary.eval()
        val_loss_total, val_acc_total, val_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, _, embeddings, lengths = batch
                x = x.to(device)
                embeddings = embeddings.to(device)
                lengths = lengths.to(device)

                z = get_adversary_input(x)
                logits = adversary(z, embeddings, lengths)
                loss = compute_contrastive_loss(logits, embeddings, soft_loss_weight, tau_st)

                targets = torch.arange(logits.size(0), device=device)
                acc = (logits.argmax(dim=1) == targets).float().mean()

                val_loss_total += loss.item()
                val_acc_total += acc.item()
                val_batches += 1

        val_loss = val_loss_total / max(val_batches, 1)
        val_acc = val_acc_total / max(val_batches, 1)
        writer.add_scalar("val/adv_loss", val_loss, epoch)
        writer.add_scalar("val/adv_acc", val_acc, epoch)
        print(f"Epoch {epoch}: val_adv_loss={val_loss:.4f}, val_adv_acc={val_acc:.3f}")

        # Checkpointing
        ckpt = {
            "epoch": epoch,
            "adversary_state_dict": adversary.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "config": config,
        }
        torch.save(ckpt, os.path.join(log_dir, "last.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(log_dir, "best.pt"))

    writer.close()
    print(f"Training complete. Best val_adv_loss={best_val_loss:.4f}. Checkpoints saved to {log_dir}")


if __name__ == "__main__":
    main()
