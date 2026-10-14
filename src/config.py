"""Konfigurasi training Pix2Pix (KONTRAK v1.2, PC lokal RTX 5060 Ti).

Semua hyperparameter dikumpulkan di satu dataclass. Nilai di sini adalah
default; untuk eksperimen tuning, override lewat argumen terminal di train.py
(contoh: py -m src.train --run_name lr1e-4 --lr 1e-4) supaya file ini tidak
perlu diedit tiap eksperimen.
"""
from dataclasses import dataclass, asdict
from pathlib import Path

import torch


@dataclass
class TrainConfig:
    # ---- Identitas run & path ----
    run_name: str = "baseline"
    data_root: str = "data/facades"
    output_root: str = "outputs"

    # ---- Data ----
    image_size: int = 256
    load_size: int = 286          # resize sebelum random crop (KONTRAK bagian 4)
    batch_size: int = 1           # nilai paper Pix2Pix untuk Facades

    # ---- DataLoader (lihat penjelasan di bawah) ----
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    # ---- Model ----
    in_channels: int = 3
    out_channels: int = 3
    features: int = 64

    # ---- Optimizer ----
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999

    # ---- Jadwal epoch: lr konstan dulu, lalu turun linear ke 0 ----
    epochs: int = 200
    decay_start_epoch: int = 100  # epoch 1..100 lr konstan, 101..200 turun linear

    # ---- Bobot loss: G_total = lambda_adv * G_adv + lambda_l1 * G_L1 ----
    lambda_adv: float = 1.0
    lambda_l1: float = 100.0

    # ---- Penyimpanan ----
    save_every: int = 10          # checkpoint bernomor tiap N epoch (last.pt tetap tiap epoch)
    sample_every: int = 1         # simpan grid sampel tiap N epoch
    num_samples: int = 4          # jumlah gambar val di grid sampel

    # ---- Reproducibility ----
    seed: int = 42

    # ---- Mode tes ----
    dummy: bool = False           # pakai data sintetis, bukan dataset.py
    max_steps_per_epoch: int = 0  # 0 = penuh; >0 = potong epoch (buat smoke test)

    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.run_name

    def loader_kwargs(self) -> dict:
        """Argumen DataLoader yang aman di semua kondisi.

        - pin_memory cuma berguna kalau ada GPU.
        - persistent_workers hanya boleh True kalau num_workers > 0
          (PyTorch akan error kalau tidak).
        """
        use_cuda = torch.cuda.is_available()
        return {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory and use_cuda,
            "persistent_workers": self.persistent_workers and self.num_workers > 0,
        }

    def to_dict(self) -> dict:
        return asdict(self)
