# =========================
# Dataset definitions
# =========================

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class MoNuSegPairedDataset(Dataset):
    """
    PyTorch dataset for the MoNuSeg paired image-to-image translation task.

    Each PNG file contains:
    - Left half: realistic histology image.
    - Right half: corresponding label map.

    The model input is the label map and the target output is the realistic image.
    """

    def __init__(self, image_paths: list[Path], image_size: int = 256):
        self.image_paths = list(image_paths)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        image_path = self.image_paths[index]

        paired_image = Image.open(image_path).convert("RGB")
        width, height = paired_image.size
        half_width = width // 2

        real_image = paired_image.crop((0, 0, half_width, height))
        label_map = paired_image.crop((half_width, 0, width, height))

        label_map = TF.resize(
            label_map,
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        real_image = TF.resize(
            real_image,
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.BICUBIC,
        )

        label_map = TF.to_tensor(label_map)
        real_image = TF.to_tensor(real_image)

        label_map = TF.normalize(label_map, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        real_image = TF.normalize(real_image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return {
            "label": label_map,
            "real": real_image,
            "filename": image_path.name,
        }


class MoNuSegPairedDatasetAug(Dataset):
    """
    Augmented variant of MoNuSegPairedDataset for the improved Pix2Pix training.

    The augmentations are applied synchronously to both the label map and the
    target real image, so the paired correspondence between input and output is
    preserved. This is critical for paired image-to-image translation: an
    augmentation that altered only one side of the pair would destroy the
    supervision signal.

    Augmentations applied (only at training time):
    - Random horizontal flip (probability 0.5).
    - Random vertical flip (probability 0.5).
    - Random 90 degree rotations (0, 90, 180 or 270 degrees).
    - Mild color jitter on the real histology image only (brightness, contrast
      and saturation). H&E stained tissue has a meaningful color distribution,
      so the jitter is kept subtle. The label map is NOT color-jittered because
      its colors carry semantic information (different colors = different
      classes); altering them would change the supervision signal.

    The augmentations are well suited to histology: there is no canonical
    orientation in microscopy slides, so flips and rotations are physically
    meaningful and do not introduce label noise.
    """

    def __init__(
        self,
        image_paths: list[Path],
        image_size: int = 256,
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.5,
        rotation_prob: float = 0.75,
        color_jitter_strength: float = 0.1,
    ):
        self.image_paths = list(image_paths)
        self.image_size = image_size
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_prob = rotation_prob
        self.color_jitter_strength = color_jitter_strength

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        image_path = self.image_paths[index]

        paired_image = Image.open(image_path).convert("RGB")
        width, height = paired_image.size
        half_width = width // 2

        real_image = paired_image.crop((0, 0, half_width, height))
        label_map = paired_image.crop((half_width, 0, width, height))

        label_map = TF.resize(
            label_map,
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        real_image = TF.resize(
            real_image,
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.BICUBIC,
        )

        # ---------------------------------------------------------------
        # Synchronized geometric augmentations (applied to both images)
        # ---------------------------------------------------------------

        if random.random() < self.horizontal_flip_prob:
            label_map = TF.hflip(label_map)
            real_image = TF.hflip(real_image)

        if random.random() < self.vertical_flip_prob:
            label_map = TF.vflip(label_map)
            real_image = TF.vflip(real_image)

        if random.random() < self.rotation_prob:
            angle = random.choice([90, 180, 270])
            label_map = TF.rotate(
                label_map,
                angle=angle,
                interpolation=InterpolationMode.NEAREST,
            )
            real_image = TF.rotate(
                real_image,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR,
            )

        # ---------------------------------------------------------------
        # Photometric augmentation on the real image only.
        # We intentionally do not jitter the label map.
        # ---------------------------------------------------------------

        if self.color_jitter_strength > 0:
            jitter = self.color_jitter_strength
            brightness_factor = 1.0 + random.uniform(-jitter, jitter)
            contrast_factor = 1.0 + random.uniform(-jitter, jitter)
            saturation_factor = 1.0 + random.uniform(-jitter, jitter)

            real_image = TF.adjust_brightness(real_image, brightness_factor)
            real_image = TF.adjust_contrast(real_image, contrast_factor)
            real_image = TF.adjust_saturation(real_image, saturation_factor)

        # ---------------------------------------------------------------
        # Tensor conversion and normalization to [-1, 1]
        # ---------------------------------------------------------------

        label_map = TF.to_tensor(label_map)
        real_image = TF.to_tensor(real_image)

        label_map = TF.normalize(label_map, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        real_image = TF.normalize(real_image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return {
            "label": label_map,
            "real": real_image,
            "filename": image_path.name,
        }


def create_dataloaders(
    train_paths: list[Path],
    val_paths: list[Path],
    test_paths: list[Path],
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create training, validation, and test DataLoaders.
    """
    train_dataset = MoNuSegPairedDataset(train_paths, image_size=image_size)
    val_dataset = MoNuSegPairedDataset(val_paths, image_size=image_size)
    test_dataset = MoNuSegPairedDataset(test_paths, image_size=image_size)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def create_dataloaders_aug(
    train_paths: list[Path],
    val_paths: list[Path],
    test_paths: list[Path],
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for the improved Pix2Pix training.

    Augmentations are applied ONLY to the training set. Validation and test
    use the same deterministic dataset as the baseline, so the comparison
    between baseline and improved models is fair: the metrics are computed on
    exactly the same val/test images that the baseline was evaluated on.
    """
    train_dataset = MoNuSegPairedDatasetAug(train_paths, image_size=image_size)
    val_dataset = MoNuSegPairedDataset(val_paths, image_size=image_size)
    test_dataset = MoNuSegPairedDataset(test_paths, image_size=image_size)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader