from __future__ import annotations

# =========================
# Pix2Pix training utilities
# =========================

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def create_patch_labels(
    prediction: torch.Tensor,
    target_is_real: bool,
) -> torch.Tensor:
    """
    Create real or fake labels with the same shape as the discriminator output.
    """
    if target_is_real:
        return torch.ones_like(prediction)
    return torch.zeros_like(prediction)


def train_pix2pix_one_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    adversarial_criterion: nn.Module,
    reconstruction_criterion: nn.Module,
    device: torch.device,
    lambda_l1: float = 100.0,
) -> dict:
    """
    Train Pix2Pix for one epoch.

    The generator is optimized with:
    - adversarial loss, to fool the discriminator;
    - L1 reconstruction loss, to keep the generated image close to the target.

    The discriminator is optimized to distinguish real pairs from generated pairs.
    """
    generator.train()
    discriminator.train()

    running_g_loss = 0.0
    running_g_adv_loss = 0.0
    running_g_l1_loss = 0.0
    running_d_loss = 0.0

    num_batches = len(dataloader)

    for batch_index, batch in enumerate(dataloader, start=1):
        label_maps = batch["label"].to(device)
        real_images = batch["real"].to(device)

        # -------------------------
        # Train discriminator
        # -------------------------

        optimizer_d.zero_grad(set_to_none=True)

        with torch.no_grad():
            fake_images = generator(label_maps)

        real_prediction = discriminator(label_maps, real_images)
        fake_prediction = discriminator(label_maps, fake_images.detach())

        real_targets = create_patch_labels(real_prediction, target_is_real=True)
        fake_targets = create_patch_labels(fake_prediction, target_is_real=False)

        d_real_loss = adversarial_criterion(real_prediction, real_targets)
        d_fake_loss = adversarial_criterion(fake_prediction, fake_targets)
        d_loss = 0.5 * (d_real_loss + d_fake_loss)

        d_loss.backward()
        optimizer_d.step()

        # -------------------------
        # Train generator
        # -------------------------

        optimizer_g.zero_grad(set_to_none=True)

        fake_images = generator(label_maps)
        fake_prediction = discriminator(label_maps, fake_images)

        real_targets_for_generator = create_patch_labels(fake_prediction, target_is_real=True)

        g_adv_loss = adversarial_criterion(fake_prediction, real_targets_for_generator)
        g_l1_loss = reconstruction_criterion(fake_images, real_images)
        g_loss = g_adv_loss + lambda_l1 * g_l1_loss

        g_loss.backward()
        optimizer_g.step()

        # -------------------------
        # Logging
        # -------------------------

        running_g_loss += g_loss.item()
        running_g_adv_loss += g_adv_loss.item()
        running_g_l1_loss += g_l1_loss.item()
        running_d_loss += d_loss.item()

        if batch_index % 10 == 0 or batch_index == num_batches:
            print(
                f"Batch {batch_index:03d}/{num_batches:03d} | "
                f"G: {g_loss.item():.4f} | "
                f"G L1: {g_l1_loss.item():.4f} | "
                f"D: {d_loss.item():.4f}"
            )

    return {
        "generator_loss": running_g_loss / num_batches,
        "generator_adversarial_loss": running_g_adv_loss / num_batches,
        "generator_l1_loss": running_g_l1_loss / num_batches,
        "discriminator_loss": running_d_loss / num_batches,
    }

@torch.no_grad()
def validate_pix2pix(
    generator: nn.Module,
    dataloader: DataLoader,
    reconstruction_criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Validate the generator using reconstruction loss.

    During validation, we only evaluate the generator because the final goal is
    image generation from label maps.
    """
    generator.eval()

    running_l1_loss = 0.0
    num_batches = len(dataloader)

    for batch in dataloader:
        label_maps = batch["label"].to(device)
        real_images = batch["real"].to(device)

        fake_images = generator(label_maps)
        l1_loss = reconstruction_criterion(fake_images, real_images)

        running_l1_loss += l1_loss.item()

    return {
        "val_l1_loss": running_l1_loss / num_batches,
    }


def save_checkpoint(
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    epoch: int,
    history: list[dict],
    checkpoint_path: Path,
) -> None:
    """
    Save a training checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "history": history,
        },
        checkpoint_path,
    )


def load_generator_weights(
    generator: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """
    Load generator weights from a Pix2Pix checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.to(device)
    generator.eval()

    return generator