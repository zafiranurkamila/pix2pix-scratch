"""Ambil contoh hasil terbaik dan terburuk (berdasarkan L1) dari hasil evaluate.py.

Jalankan SETELAH evaluate.py:
    py -m src.pick_examples --run_name bs4

Output: docs/hasil/contoh_terbaik_terburuk.png
    baris atas  = k gambar dengan L1 terkecil (paling mirip foto asli)
    baris bawah = k gambar dengan L1 terbesar
    kolom       = label map | hasil G | foto asli
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Contoh hasil terbaik dan terburuk")
    p.add_argument("--run_name", default="bs4")
    p.add_argument("--output_root", default="outputs")
    p.add_argument("--eval_tag", default="test_last_eval")
    p.add_argument("--k", type=int, default=3, help="jumlah contoh terbaik dan terburuk")
    p.add_argument("--out", default="docs/hasil/contoh_terbaik_terburuk.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.output_root) / args.run_name / "eval" / args.eval_tag
    with open(eval_dir / "per_image.csv", newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r["L1"]))
    picks = [("Terbaik", r) for r in rows[:args.k]] + [("Terburuk", r) for r in rows[-args.k:][::-1]]

    img_dir = eval_dir / "images"
    cols = [("Label map", "input"), (f"Hasil {args.run_name}", "pred"), ("Foto asli", "target")]
    fig, axes = plt.subplots(len(picks), 3, figsize=(6.6, 2.2 * len(picks) + 0.3), squeeze=False)
    for i, (kind, r) in enumerate(picks):
        idx = int(r["index"])
        for j, (title, prefix) in enumerate(cols):
            ax = axes[i][j]
            ax.imshow(Image.open(img_dir / f"{prefix}_{idx:03d}.png"))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=10)
        axes[i][0].set_ylabel(f"{kind}\ntest #{idx}\nL1 = {float(r['L1']):.3f}", fontsize=8.5,
                              color="#1a7f37" if kind == "Terbaik" else "#c62828")
        print(f"{kind:9s} test #{idx:3d}  L1={float(r['L1']):.4f}  PSNR={float(r['PSNR']):.2f}  "
              f"SSIM={float(r['SSIM']):.4f}")

    fig.tight_layout(pad=0.4)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[pick] disimpan ke {out}")


if __name__ == "__main__":
    main()
