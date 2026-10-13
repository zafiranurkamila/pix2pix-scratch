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
