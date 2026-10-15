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


class UNetGenerator(nn.Module):
    """Generator U-Net Pix2Pix: 8 blok encoder, 7 blok decoder + skip, layer final tanh.

    Shape: [B, in_channels, 256, 256] -> [B, out_channels, 256, 256], range [-1, 1]
    """

    # Kelipatan `features` untuk tiap blok encoder (e1..e7 + bottleneck)
    ENCODER_MULTS = (1, 2, 4, 8, 8, 8, 8, 8)
    # Jumlah blok decoder terdalam yang memakai dropout (u1, u2, u3)
    NUM_DROPOUT_BLOCKS = 3

    def __init__(self, in_channels: int = 3, out_channels: int = 3, features: int = 64):
        super().__init__()

        enc_channels = [features * m for m in self.ENCODER_MULTS]
        last = len(enc_channels) - 1

        # ----- Encoder -----
        self.downs = nn.ModuleList()
        ch = in_channels
        for i, out_ch in enumerate(enc_channels):
            use_norm = 0 < i < last          # e1 dan bottleneck tanpa BatchNorm
            self.downs.append(DownBlock(ch, out_ch, use_norm=use_norm))
            ch = out_ch

        # ----- Decoder -----
        skip_channels = list(reversed(enc_channels[:-1]))   # [512, 512, 512, 512, 256, 128, 64]
        self.ups = nn.ModuleList()
        for i, skip_ch in enumerate(skip_channels):
            use_dropout = i < self.NUM_DROPOUT_BLOCKS
            self.ups.append(UpBlock(ch, skip_ch, use_dropout=use_dropout))
            ch = skip_ch + skip_ch           # output blok + skip yang ditempel

        # ----- Layer final -----
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ch, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        self.num_downs = len(self.downs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        factor = 2 ** self.num_downs
        h, w = x.shape[-2:]
        if h % factor != 0 or w % factor != 0:
            raise ValueError(
                f"Ukuran input {h}x{w} harus kelipatan {factor} (kontrak: 256x256). "
                f"Cek preprocessing di dataset."
            )

        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # output terakhir adalah bottleneck itu sendiri, bukan skip
        skips = skips[:-1]

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        return self.final(x)
