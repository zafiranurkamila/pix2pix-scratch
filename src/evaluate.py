"""Evaluasi satu run di split test: hitung metrik dan simpan semua gambar hasil G.

Contoh pemakaian (dari root repo):
    py -m src.evaluate --run_name baseline                      # last.pt, split test
    py -m src.evaluate --run_name baseline --checkpoint best.pt
    py -m src.evaluate --run_name baseline --g_mode train       # cek mode BatchNorm

Output di outputs/<run_name>/eval/<split>_<checkpoint>_<g_mode>/:
    metrics.json     rata-rata semua metrik + config run (dibaca compare_runs.py)
    per_image.csv    metrik per gambar (buat cari contoh terbaik/terburuk)
    images/          input_XXX.png, pred_XXX.png, target_XXX.png
"""
import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.config import TrainConfig
from src.metrics import (l1_per_image, psnr_per_image, sharpness_per_image,
                         ssim_per_image)
from src.models.generator import UNetGenerator
from src.utils import denorm, load_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluasi Pix2Pix di split test")
    p.add_argument("--run_name", required=True)
    p.add_argument("--output_root", default="outputs")
    p.add_argument("--checkpoint", default="last.pt",
                   help="nama file di checkpoints/ (last.pt, best.pt, epoch_100.pt, ...)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--g_mode", default="eval", choices=["eval", "train"],
                   help="eval = BatchNorm pakai statistik rata-rata training, dropout mati; "
                        "train = BatchNorm pakai statistik batch ini, dropout aktif "
                        "(cara implementasi asli Pix2Pix)")
    p.add_argument("--batch_size", type=int, default=1,
                   help="biarkan 1 supaya hasil g_mode=train tidak bergantung pada "
                        "gambar lain di batch yang sama")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--data_root", default=None,
                   help="override lokasi dataset (default: sama dengan saat training)")
    return p.parse_args()


def config_from_checkpoint(ckpt: dict) -> TrainConfig:
    """Bangun ulang TrainConfig dari dict di checkpoint.

    Field yang tidak dikenal diabaikan, jadi checkpoint lama tetap bisa dibaca
    walaupun config.py nanti bertambah field.
    """
    known = {f.name for f in fields(TrainConfig)}
    return TrainConfig(**{k: v for k, v in ckpt["config"].items() if k in known})


def build_dataset(cfg: TrainConfig, split: str):
    if cfg.dummy:
        from src.dummy_dataset import DummyPix2PixDataset as DatasetCls
    else:
        from src.dataset import Pix2PixDataset as DatasetCls
    return DatasetCls(root=cfg.data_root, split=split,
                      image_size=cfg.image_size, load_size=cfg.load_size)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    run_dir = Path(args.output_root) / args.run_name
    ckpt_path = run_dir / "checkpoints" / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path)
    cfg = config_from_checkpoint(ckpt)
    if args.data_root:
        cfg.data_root = args.data_root
    device = cfg.device
    seed_everything(cfg.seed)   # dropout di g_mode=train jadi deterministik

    G = UNetGenerator(cfg.in_channels, cfg.out_channels, cfg.features).to(device)
    G.load_state_dict(ckpt["G"])
    # Catatan: di mode train, BatchNorm ikut memperbarui running stats-nya.
    # Tidak masalah di sini karena model ini tidak disimpan lagi.
    G.train() if args.g_mode == "train" else G.eval()

    dataset = build_dataset(cfg, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")

    tag = f"{args.split}_{Path(args.checkpoint).stem}_{args.g_mode}"
    out_dir = run_dir / "eval" / tag
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] run={args.run_name} ckpt={args.checkpoint} (epoch {ckpt['epoch']}) "
          f"split={args.split} n={len(dataset)} g_mode={args.g_mode} device={device}")

    rows, idx = [], 0
    for batch in loader:
        real_A = batch["input"].to(device)
        real_B = batch["target"].to(device)
        fake_B = G(real_A)

        # L1 dihitung di skala [-1, 1], sama dengan G_L1/val_L1 di log training,
        # supaya angkanya bisa dibandingkan langsung.
        l1 = l1_per_image(fake_B, real_B)
        # Metrik lain di skala gambar biasa [0, 1].
        fake01, real01 = denorm(fake_B), denorm(real_B)
        psnr = psnr_per_image(fake01, real01)
        ssim = ssim_per_image(fake01, real01)
        sharp_fake = sharpness_per_image(fake01)
        sharp_real = sharpness_per_image(real01)

        for i in range(real_A.size(0)):
            save_image(denorm(real_A[i]).cpu(), img_dir / f"input_{idx:03d}.png")
            save_image(fake01[i].cpu(), img_dir / f"pred_{idx:03d}.png")
            save_image(real01[i].cpu(), img_dir / f"target_{idx:03d}.png")
            rows.append({"index": idx, "L1": l1[i].item(), "PSNR": psnr[i].item(),
                         "SSIM": ssim[i].item(), "sharp_fake": sharp_fake[i].item(),
                         "sharp_real": sharp_real[i].item()})
            idx += 1

    def mean(key):
        return sum(r[key] for r in rows) / len(rows)

    summary = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "epoch": ckpt["epoch"],
        "split": args.split,
        "g_mode": args.g_mode,
        "n_images": len(rows),
        "L1": mean("L1"),
        "PSNR": mean("PSNR"),
        "SSIM": mean("SSIM"),
        "sharp_fake": mean("sharp_fake"),
        "sharp_real": mean("sharp_real"),
        # rasio dari rata-rata (bukan rata-rata rasio) supaya tidak didominasi
        # satu-dua gambar yang foto aslinya sangat halus
        "sharp_ratio": mean("sharp_fake") / mean("sharp_real"),
        "config": cfg.to_dict(),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # print sengaja hanya ASCII: di Windows, output yang dialihkan ke file
    # (seperti di run_experiments.bat) memakai encoding cp1252.
    print(f"[eval] L1={summary['L1']:.4f}  PSNR={summary['PSNR']:.2f} dB  "
          f"SSIM={summary['SSIM']:.4f}  sharp_ratio={summary['sharp_ratio']:.3f}")
    print(f"[eval] hasil di {out_dir}")


if __name__ == "__main__":
    main()
