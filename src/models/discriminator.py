"""PatchGAN Discriminator (70x70) dan inisialisasi bobot untuk Pix2Pix.

Sesuai KONTRAK.md bagian 5 dan 6:
- D menerima dua gambar (cond, img), masing-masing [B, 3, 256, 256].
- Concat dilakukan di dalam D -> [B, 6, 256, 256].
- Output berupa logits mentah [B, 1, 30, 30], TANPA sigmoid
  (sigmoid sudah ada di dalam BCEWithLogitsLoss pada training loop).
"""

import torch
import torch.nn as nn
from torch import Tensor


def _down_block(in_ch: int, out_ch: int, stride: int, use_norm: bool) -> nn.Sequential:
    """Satu blok Conv(k=4, p=1) -> [BatchNorm] -> LeakyReLU(0.2).

    Kalau memakai BatchNorm, bias conv dimatikan karena BatchNorm
    sudah punya parameter geser (beta) sendiri, jadi bias conv mubazir.
    """
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=not use_norm)
    ]
    if use_norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class PatchGANDiscriminator(nn.Module):
    """Discriminator PatchGAN dengan receptive field 70x70.

    Alur ukuran spasial untuk input 256x256:
        256 -(s2)-> 128 -(s2)-> 64 -(s2)-> 32 -(s1)-> 31 -(s1)-> 30
    """

    def __init__(self, in_channels: int = 3, features: int = 64):
        super().__init__()
        f = features
        self.model = nn.Sequential(
            # Layer pertama tanpa BatchNorm (sesuai paper Pix2Pix).
            _down_block(in_channels * 2, f,     stride=2, use_norm=False),  # 256 -> 128
            _down_block(f,               f * 2, stride=2, use_norm=True),   # 128 -> 64
            _down_block(f * 2,           f * 4, stride=2, use_norm=True),   # 64  -> 32
            _down_block(f * 4,           f * 8, stride=1, use_norm=True),   # 32  -> 31
            # Layer terakhir: 1 channel, tanpa norm, tanpa aktivasi, TANPA sigmoid.
            nn.Conv2d(f * 8, 1, kernel_size=4, stride=1, padding=1),       # 31  -> 30
        )

    def forward(self, cond: Tensor, img: Tensor) -> Tensor:
        if cond.shape != img.shape:
            raise ValueError(
                f"cond dan img harus ber-shape sama, dapat {tuple(cond.shape)} "
                f"dan {tuple(img.shape)}"
            )
        x = torch.cat([cond, img], dim=1)  # [B, 6, H, W]
        return self.model(x)               # [B, 1, 30, 30] logits mentah


def init_weights(model: nn.Module) -> None:
    """Inisialisasi bobot ala paper Pix2Pix/DCGAN.

    - Conv2d / ConvTranspose2d: weight ~ Normal(0, 0.02), bias = 0
    - BatchNorm2d:              weight ~ Normal(1, 0.02), bias = 0

    Dipakai untuk Generator maupun Discriminator: init_weights(G); init_weights(D)
    """

    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, mean=1.0, std=0.02)
            nn.init.zeros_(m.bias)

    model.apply(_init)


if __name__ == "__main__":
    # Smoke test (KONTRAK.md bagian 10): python -m src.models.discriminator
    torch.manual_seed(42)
    D = PatchGANDiscriminator()
    init_weights(D)

    cond = torch.rand(2, 3, 256, 256) * 2 - 1
    img = torch.rand(2, 3, 256, 256) * 2 - 1
    out = D(cond, img)

    n_params = sum(p.numel() for p in D.parameters())
    print(f"Output shape : {tuple(out.shape)}")
    print(f"Output range : [{out.min().item():.3f}, {out.max().item():.3f}]")
    print(f"Parameter    : {n_params:,}")
    assert out.shape == (2, 1, 30, 30), "Shape output D salah"
    assert not isinstance(D.model[-1], nn.Sigmoid), "Layer terakhir tidak boleh sigmoid"
    print("Smoke test discriminator: LOLOS")
