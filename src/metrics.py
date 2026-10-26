"""Metrik evaluasi gambar, ditulis sendiri dengan PyTorch (tanpa library tambahan).

Semua fungsi menerima batch [B, C, H, W] dengan nilai [0, 1] dan mengembalikan
nilai PER GAMBAR (tensor [B]), supaya bisa dirata-rata atau dilihat satu per satu.

Dua kelompok metrik, dan keduanya perlu dibaca bersama:
- Kemiripan dengan foto asli: L1, PSNR, SSIM. Model yang "main aman"
  (hasil blur, rata-rata) sering unggul di sini.
- Ketajaman: variance of Laplacian. Menangkap blur yang tidak terlihat
  di metrik kemiripan.
"""
import torch
import torch.nn.functional as F


def l1_per_image(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Rata-rata |fake - real| per gambar. Makin kecil makin mirip."""
    return (fake - real).abs().mean(dim=(1, 2, 3))


def psnr_per_image(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio dalam dB, untuk nilai [0, 1]. Makin besar makin mirip.

    PSNR = 10 * log10(MAX^2 / MSE), dengan MAX = 1.
    Pada dasarnya MSE dalam skala logaritmik, jadi sifatnya mirip L1/L2:
    menghukum perbedaan piksel, bukan menilai struktur.
    """
    mse = ((fake - real) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return 10.0 * torch.log10(1.0 / mse)


def _gaussian_window(size: int, sigma: float, channels: int,
                     device: torch.device) -> torch.Tensor:
    """Kernel Gaussian 2D [channels, 1, size, size] untuk konvolusi per channel."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]              # outer product -> 2D
    return window.expand(channels, 1, size, size).contiguous()


def ssim_per_image(fake: torch.Tensor, real: torch.Tensor,
                   window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Structural Similarity Index (Wang dkk., 2004). Rentang ~[0, 1], 1 = identik.

    Beda dengan L1/PSNR yang membandingkan piksel satu per satu, SSIM
    membandingkan jendela kecil 11x11 dari tiga sisi: kecerahan rata-rata (mu),
    kontras (sigma), dan struktur (kovarians). Hasil akhirnya rata-rata semua
    jendela dan semua channel RGB.
    """
    c = fake.size(1)
    w = _gaussian_window(window_size, sigma, c, fake.device)
    C1, C2 = 0.01 ** 2, 0.03 ** 2   # konstanta penstabil untuk data range 1

    def blur(x):  # rata-rata lokal berbobot Gaussian, per channel
        return F.conv2d(x, w, groups=c)

    mu_x, mu_y = blur(fake), blur(real)
    var_x = blur(fake * fake) - mu_x ** 2
    var_y = blur(real * real) - mu_y ** 2
    cov_xy = blur(fake * real) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2))
    return ssim_map.mean(dim=(1, 2, 3))


_LAPLACIAN = torch.tensor([[0.0, 1.0, 0.0],
                           [1.0, -4.0, 1.0],
                           [0.0, 1.0, 0.0]])


def sharpness_per_image(x: torch.Tensor) -> torch.Tensor:
    """Ketajaman = variance dari respons filter Laplacian pada versi grayscale.

    Laplacian adalah detektor tepi: nilainya besar di tempat yang berubah
    tajam (garis jendela, tepi bata), dan mendekati 0 di area yang halus.
    Gambar blur punya sedikit tepi tajam, sehingga variance-nya kecil.

    Angka mentahnya sulit diartikan sendirian. Bandingkan dengan ketajaman
    foto asli: rasio fake/real ~1 berarti setajam foto asli, jauh di bawah 1
    berarti blur. Rasio > 1 juga patut dicurigai: biasanya artinya noise atau
    artefak kotak-kotak, bukan detail asli.
    """
    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]   # [B, H, W]
    k = _LAPLACIAN.to(x.device).view(1, 1, 3, 3)
    response = F.conv2d(gray.unsqueeze(1), k)                       # [B, 1, H-2, W-2]
    return response.var(dim=(1, 2, 3))
