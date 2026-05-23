from __future__ import annotations

# =========================
# Experiment helpers
# =========================

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.models import (
    UNetGenerator,
    PatchGANDiscriminator,
    MultiScalePatchGANDiscriminator,
)
from src.training import (
    train_pix2pix_one_epoch,
    validate_pix2pix,
    save_checkpoint,
    load_history_from_checkpoint,
    build_adversarial_criterion,
)
from src.utils import initialize_weights, set_seed


def run_pix2pix_experiment(
    name: str,
    train_loader,
    val_loader,
    checkpoint_dir: Path,
    device: torch.device,
    loss_type: str = "bce",
    use_multiscale_d: bool = False,
    num_epochs: int = 30,
    base_channels: int = 64,
    learning_rate: float = 2e-4,
    betas: tuple = (0.5, 0.999),
    lambda_l1: float = 100.0,
    seed: int = 42,
) -> tuple[list[dict], Path]:
    """
    Train one Pix2Pix ablation variant and return its history and best checkpoint path.

    If a best checkpoint already exists on disk for this experiment name, the
    function skips training and loads the stored history. This allows us to
    re-run the notebook for figures, tables, and evaluation without retraining
    models that have already been trained.

    To force retraining, delete the corresponding checkpoint files.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = checkpoint_dir / f"{name}_best.pt"
    last_ckpt = checkpoint_dir / f"{name}_last.pt"

    print(f"\n{'=' * 70}")
    print(f"Experiment: {name}")
    print(f"  loss_type = {loss_type}, multi_scale_d = {use_multiscale_d}")
    print(f"  epochs = {num_epochs}, lambda_l1 = {lambda_l1}")
    print(f"{'=' * 70}")

    # -------------------------
    # Short-circuit if a checkpoint already exists
    # -------------------------

    if best_ckpt.exists():
        print(
            f"Best checkpoint already exists at {best_ckpt}. "
            f"Loading history without retraining."
        )
        print("To force retraining, delete the corresponding checkpoint files.")

        history = load_history_from_checkpoint(best_ckpt, device=device)

        best_val_l1 = min(
            (item["val_l1_loss"] for item in history),
            default=float("inf"),
        )

        print(f"Loaded history with {len(history)} epochs.")
        print(f"Best Val L1 (from history): {best_val_l1:.4f}")

        return history, best_ckpt

    # -------------------------
    # Otherwise, train from scratch
    # -------------------------

    set_seed(seed)

    generator = UNetGenerator(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
    ).to(device)

    if use_multiscale_d:
        discriminator = MultiScalePatchGANDiscriminator(
            in_channels=3,
            base_channels=base_channels,
            num_scales=2,
        ).to(device)
    else:
        discriminator = PatchGANDiscriminator(
            in_channels=3,
            base_channels=base_channels,
        ).to(device)

    initialize_weights(generator)
    initialize_weights(discriminator)

    adversarial_criterion = build_adversarial_criterion(loss_type)
    reconstruction_criterion = nn.L1Loss()

    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=learning_rate,
        betas=betas,
    )

    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=learning_rate,
        betas=betas,
    )

    history = []
    best_val_l1 = float("inf")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_metrics = train_pix2pix_one_epoch(
            generator=generator,
            discriminator=discriminator,
            dataloader=train_loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            adversarial_criterion=adversarial_criterion,
            reconstruction_criterion=reconstruction_criterion,
            device=device,
            lambda_l1=lambda_l1,
        )

        val_metrics = validate_pix2pix(
            generator=generator,
            dataloader=val_loader,
            reconstruction_criterion=reconstruction_criterion,
            device=device,
        )

        epoch_metrics = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
        }

        history.append(epoch_metrics)

        print(
            f"G loss: {epoch_metrics['generator_loss']:.4f} | "
            f"G adv: {epoch_metrics['generator_adversarial_loss']:.4f} | "
            f"G L1: {epoch_metrics['generator_l1_loss']:.4f} | "
            f"D loss: {epoch_metrics['discriminator_loss']:.4f} | "
            f"Val L1: {epoch_metrics['val_l1_loss']:.4f} | "
            f"Val PSNR: {epoch_metrics['val_psnr']:.2f} dB | "
            f"Val SSIM: {epoch_metrics['val_ssim']:.4f}"
        )

        save_checkpoint(
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            epoch=epoch,
            history=history,
            checkpoint_path=last_ckpt,
        )

        if epoch_metrics["val_l1_loss"] < best_val_l1:
            best_val_l1 = epoch_metrics["val_l1_loss"]

            save_checkpoint(
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                epoch=epoch,
                history=history,
                checkpoint_path=best_ckpt,
            )

            print(f"  -> New best model saved (Val L1 = {best_val_l1:.4f})")

    elapsed = (time.time() - start_time) / 60.0

    print(f"\nExperiment '{name}' completed in {elapsed:.2f} min.")
    print(f"Best Val L1: {best_val_l1:.4f}")
    print(f"Best checkpoint: {best_ckpt}")

    return history, best_ckpt