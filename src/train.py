"""Training loop Pix2Pix.

Contoh pemakaian (dari root repo):
    py -m src.train                                   # run baseline
    py -m src.train --dummy --epochs 3                # tes pakai data sintetis
    py -m src.train --run_name lr1e-4 --lr 1e-4       # eksperimen tuning
    py -m src.train --run_name baseline --epochs 200  # otomatis resume kalau last.pt ada
"""
import argparse
import time
from dataclasses import fields
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from src.config import TrainConfig
from src.losses import Pix2PixLoss
from src.models.discriminator import PatchGANDiscriminator, init_weights
from src.models.generator import UNetGenerator
from src.utils import (CSVLogger, get_rng_state, load_checkpoint, plot_losses,
                       save_checkpoint, save_sample_grid, seed_everything,
                       set_requires_grad, set_rng_state)

LOG_FIELDS = ["epoch", "D_real", "D_fake", "D_total", "G_adv", "G_L1",
              "G_total", "val_L1", "lr", "epoch_time_s"]


# ===========================================================================
# 1. Persiapan: config, data, model, optimizer
# ===========================================================================
def parse_args() -> TrainConfig:
    """Setiap field di TrainConfig otomatis jadi argumen --nama_field."""
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Training Pix2Pix")
    for f in fields(TrainConfig):
        default = getattr(cfg, f.name)
        if isinstance(default, bool):
            # --dummy / --no-dummy, --pin_memory / --no-pin_memory, dst.
            parser.add_argument(f"--{f.name}", default=default,
                                action=argparse.BooleanOptionalAction)
        else:
            parser.add_argument(f"--{f.name}", type=type(default), default=default)
    return TrainConfig(**vars(parser.parse_args()))


def build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    if cfg.dummy:
        from src.dummy_dataset import DummyPix2PixDataset as DatasetCls
    else:
        # Import di dalam fungsi: train.py tetap bisa jalan dengan --dummy
        # walaupun dataset.py belum ada di branch ini.
        from src.dataset import Pix2PixDataset as DatasetCls

    kw = {"root": cfg.data_root, "image_size": cfg.image_size, "load_size": cfg.load_size}
    train_ds = DatasetCls(split="train", **kw)
    val_ds = DatasetCls(split="val", **kw)

    loader_kw = cfg.loader_kwargs()
    # Urutan shuffle dipegang generator milik sampler sendiri (seed 42).
    # Kenapa tidak lewat DataLoader(shuffle=True, generator=g)? Karena
    # DataLoader juga mengambil angka acak dari generator itu saat iterator
    # pertama dibuat. Setelah resume (proses baru) pengambilan itu terjadi
    # lagi, sehingga urutan shuffle bergeser dari run yang tidak terputus.
    sampler = RandomSampler(train_ds, generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                              drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            **loader_kw)
    return train_loader, val_loader


def linear_decay(cfg: TrainConfig):
    """Faktor pengali lr: 1.0 sampai decay_start_epoch, lalu turun linear ke ~0."""
    n_decay = max(1, cfg.epochs - cfg.decay_start_epoch)

    def factor(epoch_idx: int) -> float:  # epoch_idx mulai dari 0
        done = epoch_idx + 1 - cfg.decay_start_epoch
        return 1.0 - max(0, done) / (n_decay + 1)
    return factor


# ===========================================================================
# 2. Satu iterasi training
# ===========================================================================
def train_step(batch: dict, G, D, criterion, opt_G, opt_D, device) -> dict:
    real_A = batch["input"].to(device, non_blocking=True)    # label map
    real_B = batch["target"].to(device, non_blocking=True)   # foto asli

    # --- (a) Generator membuat foto palsu ---
    fake_B = G(real_A)

    # --- (b) Update Discriminator ---
    set_requires_grad(D, True)
    opt_D.zero_grad(set_to_none=True)
    pred_real = D(real_A, real_B)
    pred_fake = D(real_A, fake_B.detach())   # detach: gradien berhenti di sini
    # ⚠️ SESUAIKAN nama method dengan Pix2PixLoss di losses.py kalau beda
    loss_D, logs_D = criterion.discriminator_loss(pred_real, pred_fake)
    loss_D.backward()
    opt_D.step()

    # --- (c) Update Generator ---
    set_requires_grad(D, False)              # D dibekukan, cuma jadi "juri"
    opt_G.zero_grad(set_to_none=True)
    pred_fake_for_G = D(real_A, fake_B)      # TANPA detach: gradien mengalir ke G
    loss_G, logs_G = criterion.generator_loss(pred_fake_for_G, fake_B, real_B)
    loss_G.backward()
    opt_G.step()

    return {**logs_D, **logs_G}


@torch.no_grad()
def evaluate_l1(G, val_loader, device) -> float:
    """Rata-rata L1 antara hasil G dan foto asli di split val.

    Loss GAN naik-turun dan sulit dibaca. L1 di val adalah angka yang paling
    jujur untuk membandingkan eksperimen tuning (makin kecil makin mirip).
    """
    G.eval()
    total, count = 0.0, 0
    for batch in val_loader:
        real_A = batch["input"].to(device)
        real_B = batch["target"].to(device)
        fake_B = G(real_A)
        total += torch.abs(fake_B - real_B).mean().item() * real_A.size(0)
        count += real_A.size(0)
    G.train()
    return total / max(1, count)


@torch.no_grad()
def save_samples(G, fixed_batch: dict, path: Path, device) -> None:
    G.eval()
    real_A = fixed_batch["input"].to(device)
    fake_B = G(real_A)
    save_sample_grid(real_A, fake_B, fixed_batch["target"], path)
    G.train()


# ===========================================================================
# 3. Loop utama
# ===========================================================================
def main() -> None:
    cfg = parse_args()
    device = cfg.device
    seed_everything(cfg.seed)
    if device.type == "cuda":
        # Ukuran input selalu 256x256, jadi cuDNN boleh memilih algoritma
        # konvolusi tercepat sekali di awal.
        torch.backends.cudnn.benchmark = True

    run_dir = cfg.run_dir
    ckpt_dir, sample_dir, log_dir = (run_dir / d for d in ("checkpoints", "samples", "logs"))
    for d in (ckpt_dir, sample_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[info] run={cfg.run_name} device={device} dummy={cfg.dummy}")
    if device.type == "cuda":
        print(f"[info] GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"[info] train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"| {len(train_loader)} iterasi/epoch")

    # Sampel tetap: N gambar val pertama, selalu sama tiap epoch,
    # supaya perkembangan hasil G bisa dibandingkan antar epoch.
    val_items = [val_loader.dataset[i] for i in range(min(cfg.num_samples, len(val_loader.dataset)))]
    fixed_batch = {k: torch.stack([it[k] for it in val_items]) for k in ("input", "target")}

    G = UNetGenerator(cfg.in_channels, cfg.out_channels, cfg.features).to(device)
    D = PatchGANDiscriminator(cfg.in_channels, cfg.features).to(device)
    init_weights(G)
    init_weights(D)
    criterion = Pix2PixLoss(lambda_adv=cfg.lambda_adv, lambda_l1=cfg.lambda_l1)

    betas = (cfg.beta1, cfg.beta2)
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=betas)
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, linear_decay(cfg))
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, linear_decay(cfg))

    logger = CSVLogger(log_dir / "losses.csv", LOG_FIELDS)

    # ---------------- Resume ----------------
    start_epoch = 1
    best_val = float("inf")
    last_path = ckpt_dir / "last.pt"
    if last_path.exists():
        ckpt = load_checkpoint(last_path, device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        sched_G.load_state_dict(ckpt["sched_G"])
        sched_D.load_state_dict(ckpt["sched_D"])
        set_rng_state(ckpt["rng"])
        train_loader.sampler.generator.set_state(ckpt["sampler_generator"])
        best_val = ckpt["best_val_L1"]
        start_epoch = ckpt["epoch"] + 1
        logger.truncate_after(ckpt["epoch"])
        print(f"[resume] lanjut dari epoch {start_epoch} (best val_L1={best_val:.4f})")
        if ckpt["config"] != cfg.to_dict():
            print("[warn] config sekarang beda dengan config di checkpoint. "
                  "Kalau ini eksperimen baru, pakai --run_name lain.")

    # ---------------- Loop epoch ----------------
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        G.train()
        D.train()
        sums, n = {}, 0
        for step, batch in enumerate(train_loader, start=1):
            logs = train_step(batch, G, D, criterion, opt_G, opt_D, device)
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            n += 1
            if cfg.max_steps_per_epoch and step >= cfg.max_steps_per_epoch:
                break

        avg = {k: v / n for k, v in sums.items()}
        lr_now = opt_G.param_groups[0]["lr"]
        sched_G.step()
        sched_D.step()

        val_l1 = evaluate_l1(G, val_loader, device)
        epoch_time = time.time() - t0
        logger.log({"epoch": epoch, **{k: round(avg[k], 5) for k in LOG_FIELDS if k in avg},
                    "val_L1": round(val_l1, 5), "lr": lr_now,
                    "epoch_time_s": round(epoch_time, 1)})

        if epoch % cfg.sample_every == 0:
            save_samples(G, fixed_batch, sample_dir / f"epoch_{epoch:03d}.png", device)

        # Isi checkpoint: semua yang dibutuhkan untuk melanjutkan persis
        state = {
            "epoch": epoch,
            "G": G.state_dict(), "D": D.state_dict(),
            "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
            "sched_G": sched_G.state_dict(), "sched_D": sched_D.state_dict(),
            "rng": get_rng_state(),
            "sampler_generator": train_loader.sampler.generator.get_state(),
            "best_val_L1": min(best_val, val_l1),
            "config": cfg.to_dict(),
        }
        if val_l1 < best_val:
            best_val = val_l1
            save_checkpoint(state, ckpt_dir / "best.pt")
        if epoch % cfg.save_every == 0:
            save_checkpoint(state, ckpt_dir / f"epoch_{epoch:03d}.pt")
        save_checkpoint(state, last_path)     # selalu terakhir, paling penting
        plot_losses(log_dir / "losses.csv", log_dir / "loss_curve.png")

        print(f"[epoch {epoch:3d}/{cfg.epochs}] "
              f"D={avg.get('D_total', 0):.4f} G_adv={avg.get('G_adv', 0):.4f} "
              f"G_L1={avg.get('G_L1', 0):.4f} val_L1={val_l1:.4f} "
              f"lr={lr_now:.2e} ({epoch_time:.1f}s)")

    print(f"[selesai] best val_L1={best_val:.4f} | output di {run_dir}")


# Wajib di Windows: worker DataLoader di-spawn dengan meng-import ulang file
# ini. Tanpa blok ini, tiap worker akan ikut menjalankan training (KONTRAK 8.10).
if __name__ == "__main__":
    main()
