# =========================
# Dataset definitions
# =========================

from __future__ import annotations

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