from __future__ import annotations

# =========================
# Pix2Pix model definitions
# =========================

import torch
import torch.nn as nn


class UNetDownBlock(nn.Module):
    """
    Downsampling block used in the U-Net encoder.
    """

    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not use_norm,
            )
        ]

        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUpBlock(nn.Module):
    """
    Upsampling block used in the U-Net decoder.
    """

    def __init__(self, in_channels: int, out_channels: int, use_dropout: bool = False):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        ]

        if use_dropout:
            layers.append(nn.Dropout(0.5))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = torch.cat([x, skip_connection], dim=1)
        return x


class UNetGenerator(nn.Module):
    """
    U-Net generator for Pix2Pix.

    The generator receives a label map and produces a realistic histology image.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
    ):
        super().__init__()

        # Encoder
        self.down1 = UNetDownBlock(in_channels, base_channels, use_norm=False)
        self.down2 = UNetDownBlock(base_channels, base_channels * 2)
        self.down3 = UNetDownBlock(base_channels * 2, base_channels * 4)
        self.down4 = UNetDownBlock(base_channels * 4, base_channels * 8)
        self.down5 = UNetDownBlock(base_channels * 8, base_channels * 8)
        self.down6 = UNetDownBlock(base_channels * 8, base_channels * 8)
        self.down7 = UNetDownBlock(base_channels * 8, base_channels * 8)
        self.down8 = UNetDownBlock(base_channels * 8, base_channels * 8, use_norm=False)

        # Decoder
        self.up1 = UNetUpBlock(base_channels * 8, base_channels * 8, use_dropout=True)
        self.up2 = UNetUpBlock(base_channels * 16, base_channels * 8, use_dropout=True)
        self.up3 = UNetUpBlock(base_channels * 16, base_channels * 8, use_dropout=True)
        self.up4 = UNetUpBlock(base_channels * 16, base_channels * 8)
        self.up5 = UNetUpBlock(base_channels * 16, base_channels * 4)
        self.up6 = UNetUpBlock(base_channels * 8, base_channels * 2)
        self.up7 = UNetUpBlock(base_channels * 4, base_channels)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 2,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder with skip connections
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bottleneck = self.down8(d7)

        # Decoder
        u1 = self.up1(bottleneck, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for conditional image-to-image translation.

    The discriminator receives the concatenation of the label map and either
    a real image or a generated image.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        def discriminator_block(
            in_filters: int,
            out_filters: int,
            stride: int = 2,
            use_norm: bool = True,
        ) -> list[nn.Module]:
            layers = [
                nn.Conv2d(
                    in_filters,
                    out_filters,
                    kernel_size=4,
                    stride=stride,
                    padding=1,
                    bias=not use_norm,
                )
            ]

            if use_norm:
                layers.append(nn.InstanceNorm2d(out_filters, affine=True))

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, base_channels, use_norm=False),
            *discriminator_block(base_channels, base_channels * 2),
            *discriminator_block(base_channels * 2, base_channels * 4),
            *discriminator_block(base_channels * 4, base_channels * 8, stride=1),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, label_map: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        x = torch.cat([label_map, image], dim=1)
        return self.model(x)


class MultiScalePatchGANDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN discriminator for the improved Pix2Pix model.

    Two independent PatchGAN discriminators evaluate the conditional pair at
    different spatial resolutions:
    - D1 operates at the original image size (256 x 256), capturing fine local
      texture realism.
    - D2 operates on a downsampled version (128 x 128 via average pooling),
      capturing larger structural patterns and global appearance.

    Combining discriminators at multiple scales encourages the generator to
    produce outputs that are realistic both at the fine-grained pixel-texture
    level and at coarser structural levels. This is particularly useful in
    histology, where realism depends on both nuclei textures and tissue-wide
    color/contrast distributions.

    The forward method returns a list of patch prediction tensors, one per
    scale, so the training loop can sum (or average) the adversarial losses
    across scales.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_scales: int = 2,
    ):
        super().__init__()

        self.num_scales = num_scales
        self.discriminators = nn.ModuleList(
            [
                PatchGANDiscriminator(
                    in_channels=in_channels,
                    base_channels=base_channels,
                )
                for _ in range(num_scales)
            ]
        )

        # AvgPool with kernel=3, stride=2, padding=1 is the standard downsampler
        # used in pix2pixHD for the multi-scale discriminator. It is mild enough
        # to preserve structure while reducing the resolution by a factor of 2.
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, label_map: torch.Tensor, image: torch.Tensor) -> list[torch.Tensor]:
        predictions = []

        current_label = label_map
        current_image = image

        for index, discriminator in enumerate(self.discriminators):
            predictions.append(discriminator(current_label, current_image))

            # Downsample for the next (coarser) scale, except after the last D
            if index < self.num_scales - 1:
                current_label = self.downsample(current_label)
                current_image = self.downsample(current_image)

        return predictions