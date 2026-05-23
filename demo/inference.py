"""
Inference utilities for the Pix2Pix MoNuSeg demo.

This module is imported by both the CLI script (demo.py) and the Jupyter
notebook (demo_notebook.ipynb). It contains all the logic to:
1. Download trained generator weights from Hugging Face Hub (if not local).
2. Load a generator architecture and its state dict.
3. Preprocess an input paired image into the format expected by the generator.
4. Run inference.
5. Produce a side-by-side visualization (label / generated / real).

The demo loads only the generator (not the discriminator), since inference is
purely deterministic and only uses the generator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

# Make the project root importable so we can find models.py (one directory up).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models import UNetGenerator  # noqa: E402


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
#
# Maps each model name (used in --model) to its checkpoint filename on
# Hugging Face Hub. The first listed model is the default.

HUGGINGFACE_REPO_ID = "hugogs-7/pix2pix-monuseg"

AVAILABLE_MODELS = {
    "baseline": "baseline_pix2pix_best.pt",
    "aug": "improved_aug_best.pt",
    "lsgan": "improved_lsgan_best.pt",
    "multi": "improved_multi_best.pt",
    "improved": "improved_all_best.pt",
}

DEFAULT_MODEL = "improved"


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------


def resolve_checkpoint_path(
    model_name: str,
    local_checkpoints_dir: Path | None = None,
) -> Path:
    """
    Return a local path to the requested checkpoint.

    Resolution order:
    1. If `local_checkpoints_dir` is provided and the file exists there, use it.
    2. Otherwise, download the file from Hugging Face Hub on first use, then
       cache it under ~/.cache/huggingface/hub for subsequent calls.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(AVAILABLE_MODELS)}"
        )

    checkpoint_filename = AVAILABLE_MODELS[model_name]

    if local_checkpoints_dir is not None:
        local_path = Path(local_checkpoints_dir) / checkpoint_filename
        if local_path.exists():
            return local_path

    # Lazy import so that users with local checkpoints do not need to install
    # huggingface_hub.
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download checkpoints. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    downloaded_path = hf_hub_download(
        repo_id=HUGGINGFACE_REPO_ID,
        filename=checkpoint_filename,
    )
    return Path(downloaded_path)


# ---------------------------------------------------------------------------
# Generator loading
# ---------------------------------------------------------------------------


def load_generator(
    checkpoint_path: Path,
    device: torch.device | None = None,
    base_channels: int = 64,
) -> tuple[UNetGenerator, torch.device]:
    """
    Build a U-Net generator and load weights from `checkpoint_path`.

    Returns the generator (in eval mode) and the device it lives on.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = UNetGenerator(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Checkpoints saved by training.py contain a dict with 'generator_state_dict'.
    # We also support raw state_dict files (e.g. exported, stripped versions).
    if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
        generator.load_state_dict(checkpoint["generator_state_dict"])
    else:
        generator.load_state_dict(checkpoint)

    generator.eval()
    return generator, device


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def split_paired_image(
    paired_image: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    """
    Split a MoNuSeg paired image into (real, label).

    The MoNuSeg paired format puts the real histology image on the left half
    and the label map on the right half. This function returns both as
    independent PIL images.
    """
    width, height = paired_image.size
    half = width // 2
    real_image = paired_image.crop((0, 0, half, height))
    label_map = paired_image.crop((half, 0, width, height))
    return real_image, label_map


def preprocess_label_map(
    label_map: Image.Image,
    image_size: int = 256,
) -> torch.Tensor:
    """
    Resize and normalize a label map into the [-1, 1] range expected by the
    generator. Returns a (1, 3, H, W) tensor on CPU.
    """
    label_map = TF.resize(
        label_map,
        size=[image_size, image_size],
        interpolation=InterpolationMode.NEAREST,
    )
    tensor = TF.to_tensor(label_map)
    tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return tensor.unsqueeze(0)  # add batch dimension


def preprocess_real_image(
    real_image: Image.Image,
    image_size: int = 256,
) -> torch.Tensor:
    """
    Resize and normalize the ground-truth real image for visualization only.
    """
    real_image = TF.resize(
        real_image,
        size=[image_size, image_size],
        interpolation=InterpolationMode.BICUBIC,
    )
    tensor = TF.to_tensor(real_image)
    tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return tensor.unsqueeze(0)


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor from [-1, 1] back to [0, 1] for visualization.
    """
    return torch.clamp((tensor * 0.5) + 0.5, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_from_label_map(
    generator: UNetGenerator,
    label_map_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Run a single inference pass and return the generated image as a
    (1, 3, H, W) tensor in [-1, 1].
    """
    label_map_tensor = label_map_tensor.to(device)
    return generator(label_map_tensor)


@torch.no_grad()
def run_inference_on_paired_image(
    paired_image_path: Path,
    generator: UNetGenerator,
    device: torch.device,
    image_size: int = 256,
) -> dict:
    """
    Full inference pipeline on one paired image from the MoNuSeg test set.

    Returns a dict with all three images (label, generated, real) in [0, 1]
    HWC numpy format, ready for plotting or saving.
    """
    paired_image = Image.open(paired_image_path).convert("RGB")
    real_image, label_map = split_paired_image(paired_image)

    label_tensor = preprocess_label_map(label_map, image_size=image_size)
    real_tensor = preprocess_real_image(real_image, image_size=image_size)

    generated_tensor = generate_from_label_map(generator, label_tensor, device)

    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        img = denormalize_image(tensor.squeeze(0).cpu())
        return img.permute(1, 2, 0).numpy()

    return {
        "label": to_numpy(label_tensor),
        "generated": to_numpy(generated_tensor),
        "real": to_numpy(real_tensor),
        "source_filename": Path(paired_image_path).name,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def make_side_by_side_figure(
    inference_result: dict,
    model_name: str | None = None,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    Build a label / generated / real side-by-side figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(inference_result["label"])
    axes[0].set_title("Input label map")
    axes[0].axis("off")

    title_generated = "Generated"
    if model_name is not None:
        title_generated = f"Generated ({model_name})"
    axes[1].imshow(inference_result["generated"])
    axes[1].set_title(title_generated)
    axes[1].axis("off")

    axes[2].imshow(inference_result["real"])
    axes[2].set_title("Real (ground truth)")
    axes[2].axis("off")

    plt.tight_layout()
    return fig


def save_side_by_side(
    inference_result: dict,
    output_path: Path,
    model_name: str | None = None,
) -> None:
    """
    Save the side-by-side figure to disk.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = make_side_by_side_figure(inference_result, model_name=model_name)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)