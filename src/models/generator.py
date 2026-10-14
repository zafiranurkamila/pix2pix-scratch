import torch
from torch import nn


class DownBlock(nn.Module):
    """Satu blok encoder U-Net: Conv 4x4 stride 2 -> (BatchNorm) -> LeakyReLU.

    Shape: [B, in_channels, H, W] -> [B, out_channels, H/2, W/2]
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
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Satu blok decoder U-Net: ConvTranspose 4x4 stride 2 -> BatchNorm -> (Dropout) -> ReLU,
    lalu hasilnya digabung (concat) dengan skip connection dari encoder.

    Shape:
        x    : [B, in_channels, H, W]
        skip : [B, skip_channels, 2H, 2W]
        out  : [B, out_channels + skip_channels, 2H, 2W]
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
            nn.BatchNorm2d(out_channels),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return torch.cat([x, skip], dim=1)
