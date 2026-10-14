"""Fungsi bantu training: seed, checkpoint, log CSV, sampel gambar, grafik loss."""
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    """Kunci semua sumber random (Python, NumPy, PyTorch CPU & GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_rng_state() -> dict:
    """Ambil posisi semua generator random, disimpan di checkpoint."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    """Kembalikan posisi generator random setelah resume."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------------------
# Gradien
# ---------------------------------------------------------------------------
def set_requires_grad(model: nn.Module, flag: bool) -> None:
    """Nyalakan/matikan perhitungan gradien untuk semua parameter model.

    Dipakai saat update G: D dibekukan supaya backward tidak membuang waktu
    menghitung gradien D yang tidak akan dipakai.
    """
    for p in model.parameters():
        p.requires_grad_(flag)


# ---------------------------------------------------------------------------
# Gambar
# ---------------------------------------------------------------------------
def denorm(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] (rumus KONTRAK bagian 4)."""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def save_sample_grid(inp: torch.Tensor, fake: torch.Tensor, target: torch.Tensor,
                     path: Path) -> None:
    """Simpan grid: tiap baris = [label map | hasil G | foto asli]."""
    rows = []
    for i in range(inp.size(0)):
        rows.extend([inp[i], fake[i], target[i]])
    grid = denorm(torch.stack(rows).cpu())
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path, nrow=3, padding=2)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def save_checkpoint(state: dict, path: Path) -> None:
    """Simpan checkpoint secara atomik.

    Tulis ke file .tmp dulu, baru di-rename. Kalau listrik padam di tengah
    penyimpanan, yang rusak cuma .tmp; checkpoint lama tetap utuh.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    # weights_only=False karena checkpoint berisi RNG state & dict config,
    # bukan cuma tensor. Aman karena file ini kita buat sendiri.
    return torch.load(path, map_location=device, weights_only=False)


# ---------------------------------------------------------------------------
# Log loss (CSV)
# ---------------------------------------------------------------------------
class CSVLogger:
    """Satu baris per epoch. Mode append supaya log berlanjut saat resume."""

    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def truncate_after(self, epoch: int) -> None:
        """Buang baris epoch > `epoch`.

        Kasusnya: log epoch 15 sudah tertulis, tapi checkpoint terakhir di
        epoch 14 (PC mati sebelum checkpoint tersimpan). Saat resume, epoch 15
        dijalankan ulang, jadi baris lamanya harus dibuang agar tidak dobel.
        """
        if not self.path.exists():
            return
        with open(self.path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if int(r["epoch"]) <= epoch]
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def log(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)

    def read(self) -> list[dict]:
        with open(self.path, newline="") as f:
            return list(csv.DictReader(f))


def plot_losses(csv_path: Path, out_path: Path) -> None:
    """Gambar grafik loss dari CSV (dibaca ulang, jadi tetap utuh setelah resume)."""
    import matplotlib
    matplotlib.use("Agg")  # tanpa jendela GUI
    import matplotlib.pyplot as plt

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    ep = [int(r["epoch"]) for r in rows]

    def col(name):
        return [float(r[name]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(ep, col("D_total"), label="D_total")
    axes[0].plot(ep, col("G_adv"), label="G_adv")
    axes[0].set_title("Loss adversarial")
    axes[1].plot(ep, col("G_L1"), label="G_L1 (train)")
    axes[1].plot(ep, col("val_L1"), label="L1 (val)")
    axes[1].set_title("L1: train vs val")
    axes[2].plot(ep, col("lr"), label="lr")
    axes[2].set_title("Learning rate")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
