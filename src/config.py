"""
Konfigurasi terpusat untuk proyek Pix2Pix.

Semua angka yang bisa diubah (path, hyperparameter, jadwal simpan) dikumpulkan
di satu tempat. Tujuannya supaya tidak ada "magic number" yang tersebar di
dataset.py, file model, atau train.py.

Pemilik file: Zafira (Model & Training).

Contoh pemakaian:
    from src.config import Config

    cfg = Config()                                   # pakai nilai default
    cfg = Config(lr_gen=1e-4, run_name="lr1e-4")     # override untuk eksperimen
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_root: str = "data/facades"
    image_size: int = 256      # KONTRAK: 256x256. Jangan diubah tanpa kesepakatan tim.
    load_size: int = 286       # Ukuran resize sebelum random crop (augmentasi jitter).
    in_channels: int = 3       # Label map RGB.
    out_channels: int = 3      # Foto RGB.
    num_workers: int = 2       # Colab biasanya memberi 2 vCPU.

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    gen_features: int = 64     # Jumlah filter di layer pertama Generator.
    disc_features: int = 64    # Jumlah filter di layer pertama Discriminator.

    # ------------------------------------------------------------------
    # Training (nilai awal mengikuti paper Pix2Pix)
    # ------------------------------------------------------------------
    num_epochs: int = 200
    batch_size: int = 1
    lr_gen: float = 2e-4       # Learning rate Generator.
    lr_disc: float = 2e-4      # Learning rate Discriminator (dipisah supaya bisa di-tuning sendiri).
    beta1: float = 0.5         # Momentum Adam. 0.5 lebih stabil untuk GAN dibanding default 0.9.
    beta2: float = 0.999
    lambda_l1: float = 100.0   # Bobot L1 loss terhadap loss adversarial.
    seed: int = 42

    # ------------------------------------------------------------------
    # Output, checkpoint, dan logging
    # ------------------------------------------------------------------
    output_dir: str = "outputs"   # Di Colab, arahkan ke folder Google Drive.
    run_name: str = "baseline"    # Nama eksperimen; tiap eksperimen beda folder.
    save_every: int = 10          # Simpan checkpoint tiap N epoch.
    sample_every: int = 5         # Simpan contoh gambar validasi tiap N epoch.
    log_every: int = 50           # Catat loss tiap N iterasi.
    resume: bool = True           # Lanjutkan dari checkpoint terakhir kalau ada.

    def __post_init__(self) -> None:
        """Cek nilai config langsung saat dibuat, supaya salah setting ketahuan di awal."""
        if self.image_size != 256:
            raise ValueError(
                f"image_size harus 256 sesuai KONTRAK.md (dapat {self.image_size}). "
                "U-Net Pix2Pix butuh 8 kali downsampling."
            )
        if self.load_size < self.image_size:
            raise ValueError(
                f"load_size ({self.load_size}) tidak boleh lebih kecil dari "
                f"image_size ({self.image_size}), karena random crop butuh ruang."
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size minimal 1 (dapat {self.batch_size}).")
        if self.lr_gen <= 0 or self.lr_disc <= 0:
            raise ValueError("Learning rate harus lebih besar dari 0.")
        if self.lambda_l1 < 0:
            raise ValueError(f"lambda_l1 tidak boleh negatif (dapat {self.lambda_l1}).")

    # ------------------------------------------------------------------
    # Path turunan
    # ------------------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def sample_dir(self) -> Path:
        return self.run_dir / "samples"

    @property
    def log_dir(self) -> Path:
        return self.run_dir / "logs"

    # ------------------------------------------------------------------
    # Utilitas
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        """Pakai GPU kalau tersedia, kalau tidak jatuh ke CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def betas(self) -> tuple:
        """Format yang dibutuhkan torch.optim.Adam."""
        return (self.beta1, self.beta2)

    def make_dirs(self) -> None:
        """Buat folder checkpoint, sampel, dan log untuk run ini."""
        for folder in (self.checkpoint_dir, self.sample_dir, self.log_dir):
            folder.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Ubah config jadi dict, untuk disimpan di checkpoint dan log eksperimen."""
        return asdict(self)
