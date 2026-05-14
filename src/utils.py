# =========================
# Utility functions
# =========================

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds to make the experiments as reproducible as possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """
    Return the available computation device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image_paths(dataset_dir: Path, extension: str = "*.png") -> list[Path]:
    """
    Return all image paths from a dataset directory.
    """
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found at: {dataset_dir}. "
            "Please make sure the dataset folder is located at the project root."
        )

    image_paths = sorted(dataset_dir.glob(extension))

    if len(image_paths) == 0:
        raise RuntimeError(f"No images with extension {extension} were found in {dataset_dir}.")

    return image_paths


def split_image_paths(
    image_paths: list[Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Split image paths into training, validation, and test subsets.
    """
    paths = list(image_paths)

    rng = random.Random(seed)
    rng.shuffle(paths)

    n_total = len(paths)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_paths = paths[:n_train]
    val_paths = paths[n_train:n_train + n_val]
    test_paths = paths[n_train + n_val:]

    return train_paths, val_paths, test_paths


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert an image tensor from [-1, 1] back to [0, 1] for visualization.
    """
    return torch.clamp((tensor * 0.5) + 0.5, 0.0, 1.0)


def show_paired_batch(batch: dict, num_samples: int = 4) -> None:
    """
    Display input label maps and target real images from a batch.
    """
    label_maps = denormalize_image(batch["label"][:num_samples]).cpu()
    real_images = denormalize_image(batch["real"][:num_samples]).cpu()

    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3 * num_samples))

    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        axes[i, 0].imshow(label_maps[i].permute(1, 2, 0))
        axes[i, 0].set_title("Input label map")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(real_images[i].permute(1, 2, 0))
        axes[i, 1].set_title("Target real image")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


def show_original_paired_sample(image_path: Path) -> None:
    """
    Display the original paired image, the real image, and the label map.
    """
    from PIL import Image

    paired_image = Image.open(image_path).convert("RGB")

    width, height = paired_image.size
    half_width = width // 2

    real_image = paired_image.crop((0, 0, half_width, height))
    label_map = paired_image.crop((half_width, 0, width, height))

    print(f"Sample file: {image_path.name}")
    print(f"Full paired image size: {paired_image.size}")
    print(f"Real image size: {real_image.size}")
    print(f"Label map size: {label_map.size}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(paired_image)
    axes[0].set_title("Original paired image")
    axes[0].axis("off")

    axes[1].imshow(real_image)
    axes[1].set_title("Real histology image")
    axes[1].axis("off")

    axes[2].imshow(label_map)
    axes[2].set_title("Input label map")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def initialize_weights(model: torch.nn.Module) -> None:
    """
    Initialize convolutional and normalization layers following the Pix2Pix convention.
    """
    import torch.nn as nn

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight.data, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0.0)

        elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if module.weight is not None:
                nn.init.normal_(module.weight.data, mean=1.0, std=0.02)

            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0.0)
                

def show_generated_batch(
    generator: torch.nn.Module,
    dataloader,
    device: torch.device,
    num_samples: int = 4,
) -> None:
    """
    Display label maps, generated images, and target real images.
    """
    generator.eval()

    batch = next(iter(dataloader))

    label_maps = batch["label"].to(device)
    real_images = batch["real"].to(device)

    with torch.no_grad():
        generated_images = generator(label_maps)

    label_maps = denormalize_image(label_maps[:num_samples]).cpu()
    generated_images = denormalize_image(generated_images[:num_samples]).cpu()
    real_images = denormalize_image(real_images[:num_samples]).cpu()

    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))

    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        axes[i, 0].imshow(label_maps[i].permute(1, 2, 0))
        axes[i, 0].set_title("Input label map")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(generated_images[i].permute(1, 2, 0))
        axes[i, 1].set_title("Generated image")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(real_images[i].permute(1, 2, 0))
        axes[i, 2].set_title("Target real image")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


def plot_training_history(history: list[dict]) -> None:
    """
    Plot the main training and validation losses.
    """
    epochs = [item["epoch"] for item in history]

    generator_loss = [item["generator_loss"] for item in history]
    discriminator_loss = [item["discriminator_loss"] for item in history]
    val_l1_loss = [item["val_l1_loss"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, generator_loss, marker="o", label="Generator loss")
    plt.plot(epochs, discriminator_loss, marker="o", label="Discriminator loss")
    plt.plot(epochs, val_l1_loss, marker="o", label="Validation L1 loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline Pix2Pix smoke-test training history")
    plt.legend()
    plt.grid(True)
    plt.show()