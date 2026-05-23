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


def _as_prediction_list(prediction) -> list[torch.Tensor]:
    """
    Normalize a discriminator output to a list of patch prediction tensors.

    A single-scale PatchGAN returns a tensor; a MultiScalePatchGANDiscriminator
    returns a list of tensors. Wrapping the single-scale case in a one-element
    list lets the rest of the training code treat both uniformly.
    """
    if isinstance(prediction, (list, tuple)):
        return list(prediction)
    return [prediction]


def compute_adversarial_loss(
    prediction,
    target_is_real: bool,
    criterion: nn.Module,
) -> torch.Tensor:
    """
    Compute the adversarial loss for a discriminator output that may come from
    either a single-scale or a multi-scale discriminator.

    For multi-scale discriminators, the loss at each scale is computed
    independently and then averaged across scales. Averaging (rather than
    summing) keeps the loss magnitude comparable to the single-scale case, so
    the same lambda_l1 weighting works without retuning.
    """
    predictions = _as_prediction_list(prediction)
    losses = []
    for pred in predictions:
        targets = create_patch_labels(pred, target_is_real=target_is_real)
        losses.append(criterion(pred, targets))
    return torch.stack(losses).mean()


def build_adversarial_criterion(loss_type: str) -> nn.Module:
    """
    Build the adversarial criterion for Pix2Pix training.

    Supported values:
    - "bce": original Pix2Pix formulation with BCEWithLogitsLoss. Treats the
      adversarial game as binary classification of patches.
    - "lsgan": Least Squares GAN formulation (Mao et al., 2017) with MSELoss.
      Penalizes how far the discriminator output is from the target label,
      which gives smoother gradients and tends to be more stable than BCE.
    """
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    if loss_type == "lsgan":
        return nn.MSELoss()
    raise ValueError(f"Unknown adversarial loss type: {loss_type!r}. Expected 'bce' or 'lsgan'.")


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

        # Single generator forward pass per batch, reused for both D and G
        # updates. For the D step we detach to avoid gradients flowing back
        # into the generator; for the G step we re-run the discriminator on
        # the same fake_images (now without detach).
        fake_images = generator(label_maps)

        # -------------------------
        # Train discriminator
        # -------------------------

        optimizer_d.zero_grad(set_to_none=True)

        real_prediction = discriminator(label_maps, real_images)
        fake_prediction = discriminator(label_maps, fake_images.detach())

        d_real_loss = compute_adversarial_loss(
            real_prediction, target_is_real=True, criterion=adversarial_criterion
        )
        d_fake_loss = compute_adversarial_loss(
            fake_prediction, target_is_real=False, criterion=adversarial_criterion
        )
        d_loss = 0.5 * (d_real_loss + d_fake_loss)

        d_loss.backward()
        optimizer_d.step()

        # -------------------------
        # Train generator
        # -------------------------

        optimizer_g.zero_grad(set_to_none=True)

        fake_prediction = discriminator(label_maps, fake_images)

        g_adv_loss = compute_adversarial_loss(
            fake_prediction, target_is_real=True, criterion=adversarial_criterion
        )
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
    Validate the generator using reconstruction loss and image quality metrics.

    During validation, we only evaluate the generator because the final goal is
    image generation from label maps.

    Reported metrics:
    - L1 loss (in normalized [-1, 1] space).
    - PSNR (in dB, computed in the [0, 1] image space).
    - SSIM (computed in the [0, 1] image space).
    """
    generator.eval()

    running_l1_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    num_batches = len(dataloader)

    for batch in dataloader:
        label_maps = batch["label"].to(device)
        real_images = batch["real"].to(device)

        fake_images = generator(label_maps)
        l1_loss = reconstruction_criterion(fake_images, real_images)

        # PSNR and SSIM are computed in the [0, 1] image space, which is the
        # standard convention. The model outputs and targets live in [-1, 1],
        # so we denormalize first.
        fake_01 = torch.clamp((fake_images * 0.5) + 0.5, 0.0, 1.0)
        real_01 = torch.clamp((real_images * 0.5) + 0.5, 0.0, 1.0)

        running_l1_loss += l1_loss.item()
        running_psnr += compute_psnr(fake_01, real_01).item()
        running_ssim += compute_ssim(fake_01, real_01).item()

    return {
        "val_l1_loss": running_l1_loss / num_batches,
        "val_psnr": running_psnr / num_batches,
        "val_ssim": running_ssim / num_batches,
    }


def compute_psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the Peak Signal-to-Noise Ratio between two image batches.

    Both tensors are expected to be in the [0, 1] range and to have shape
    (N, C, H, W). The PSNR is averaged across the batch.
    """
    mse = torch.mean((prediction - target) ** 2, dim=[1, 2, 3])
    # Guard against perfect reconstruction to avoid log(0).
    mse = torch.clamp(mse, min=1e-10)
    psnr_per_image = 10.0 * torch.log10(1.0 / mse)
    return psnr_per_image.mean()


def compute_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Compute the Structural Similarity Index between two image batches.

    Both tensors are expected to be in the [0, 1] range and to have shape
    (N, C, H, W). The SSIM is averaged across batch and channels.
    """
    import torch.nn.functional as F

    device = prediction.device
    dtype = prediction.dtype
    num_channels = prediction.shape[1]

    # Build a 2D Gaussian kernel and replicate it per channel for depthwise conv
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    gaussian_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
    window = gaussian_2d.expand(num_channels, 1, window_size, window_size).contiguous()

    padding = window_size // 2

    mu_pred = F.conv2d(prediction, window, padding=padding, groups=num_channels)
    mu_target = F.conv2d(target, window, padding=padding, groups=num_channels)

    mu_pred_sq = mu_pred * mu_pred
    mu_target_sq = mu_target * mu_target
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(prediction * prediction, window, padding=padding, groups=num_channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=padding, groups=num_channels) - mu_target_sq
    sigma_pred_target = F.conv2d(prediction * target, window, padding=padding, groups=num_channels) - mu_pred_target

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = (
        (2.0 * mu_pred_target + c1) * (2.0 * sigma_pred_target + c2)
    ) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    )

    return ssim_map.mean()


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


def load_history_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> list[dict]:
    """
    Load the training history (list of per-epoch metric dicts) from a Pix2Pix
    checkpoint.

    Used together with load_generator_weights to skip retraining when a
    checkpoint already exists: we restore both the weights (for evaluation and
    qualitative inspection) and the history (for plotting the loss curves
    without having to retrain).
    """
    if device is None:
        device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint.get("history", [])


@torch.no_grad()
def evaluate_on_test_set(
    generator: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate the generator on the held-out test set.

    Reports the same set of metrics as the validation function: L1 (in
    normalized space), PSNR (dB) and SSIM. This is the function that produces
    the numbers reported in the final results table of the report.
    """
    reconstruction_criterion = nn.L1Loss()

    metrics = validate_pix2pix(
        generator=generator,
        dataloader=test_loader,
        reconstruction_criterion=reconstruction_criterion,
        device=device,
    )

    # Rename keys so they make sense when printed alongside validation metrics
    return {
        "test_l1_loss": metrics["val_l1_loss"],
        "test_psnr": metrics["val_psnr"],
        "test_ssim": metrics["val_ssim"],
    }