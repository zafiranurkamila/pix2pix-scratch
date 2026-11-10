"""Demo Pix2Pix: ubah label map jadi foto fasad memakai Generator yang sudah dilatih.

Langkah 1 (di PC, sekali saja): ambil bobot Generator dari checkpoint run terbaik.
    py -m src.demo export --run_name bs4
    -> outputs/demo/G_bs4.pt (hanya bobot G, sekitar 220 MB)

Langkah 2 (di laptop presentasi, cukup CPU):
    py -m src.demo run --weights outputs/demo/G_bs4.pt --images data/facades/test --n 4
    py -m src.demo run --weights outputs/demo/G_bs4.pt --images data/facades/test/5.jpg

--images menerima folder atau file. File boleh berupa gambar gabungan dataset
(foto | label map, seperti di data/facades) atau label map saja.
Hasil ditampilkan di jendela dan disimpan ke outputs/demo/.
"""
import argparse
import random
import time
from pathlib import Path

import torch

from src.models.generator import UNetGenerator

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def export(args) -> None:
    ckpt_path = Path(args.output_root) / args.run_name / "checkpoints" / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    out = Path(args.out or f"outputs/demo/G_{args.run_name}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"G": ckpt["G"], "run_name": args.run_name, "epoch": ckpt["epoch"]}, out)
    print(f"[export] bobot G dari {ckpt_path} (epoch {ckpt['epoch']}) -> {out}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def load_pair(path: Path):
    """Kembalikan (label_map, foto_asli atau None) sebagai tensor [3,256,256] di [-1, 1]."""
    from PIL import Image
    import torchvision.transforms.functional as TF

    from src.dataset import LABEL_ON_LEFT

    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w >= 2 * h * 0.9:  # gambar gabungan dataset: dua gambar berdampingan
        left, right = img.crop((0, 0, w // 2, h)), img.crop((w // 2, 0, w, h))
        label, photo = (left, right) if LABEL_ON_LEFT else (right, left)
    else:                 # label map saja
        label, photo = img, None

    def to_t(im):
        im = TF.resize(im, [256, 256], interpolation=TF.InterpolationMode.BICUBIC)
        return TF.to_tensor(im) * 2.0 - 1.0

    return to_t(label), (to_t(photo) if photo is not None else None)


def collect(images: str, n: int, seed: int) -> list[Path]:
    p = Path(images)
    if p.is_file():
        return [p]
    files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTENSIONS)
    if not files:
        raise SystemExit(f"[error] tidak ada gambar di {p}")
    random.Random(seed).shuffle(files)
    return files[:n]


@torch.no_grad()
def run(args) -> None:
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.time()
    G = UNetGenerator(3, 3, 64)
    if args.weights.lower() != "none":   # "none" = bobot acak, hanya untuk tes program
        print(f"[1/3] Memuat Generator dari {args.weights} (bisa beberapa detik)...", flush=True)
        state = torch.load(args.weights, map_location="cpu", weights_only=False)
        G.load_state_dict(state["G"] if "G" in state else state)
        if isinstance(state, dict) and "epoch" in state:
            print(f"      bobot epoch {state['epoch']}", flush=True)
    G.eval()
    print(f"      model siap ({time.time() - t0:.1f} detik)", flush=True)

    files = collect(args.images, args.n, args.seed)
    print(f"[2/3] Memproses {len(files)} gambar...", flush=True)
    rows = []
    for k, f in enumerate(files, start=1):
        label, photo = load_pair(f)
        t1 = time.time()
        fake = G(label.unsqueeze(0))[0]
        print(f"      ({k}/{len(files)}) {f.name}: {time.time() - t1:.2f} detik", flush=True)
        rows.append((f.name, label, fake, photo))
    print("[3/3] Menyusun gambar hasil...", flush=True)

    to_img = lambda t: ((t.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
    has_photo = any(r[3] is not None for r in rows)
    cols = 3 if has_photo else 2
    fig, axes = plt.subplots(len(rows), cols, figsize=(3.2 * cols, 3.2 * len(rows) + 0.4), squeeze=False)
    titles = ["Label map (input)", "Hasil Generator", "Foto asli"]
    for i, (name, label, fake, photo) in enumerate(rows):
        for j, t in enumerate([label, fake, photo][:cols]):
            ax = axes[i][j]
            ax.axis("off")
            if t is not None:
                ax.imshow(to_img(t))
            if i == 0:
                ax.set_title(titles[j], fontsize=11)
        axes[i][0].text(-8, 128, name, rotation=90, va="center", ha="right", fontsize=8)
    fig.tight_layout()

    out_dir = Path("outputs/demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"demo_{time.strftime('%H%M%S')}.png"
    fig.savefig(out, dpi=120, facecolor="white")
    print(f"      disimpan ke {out}", flush=True)
    if not args.no_show:
        print("Selesai. Jendela hasil terbuka; tutup jendelanya untuk kembali ke terminal.", flush=True)
        plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description="Demo Pix2Pix")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="simpan bobot Generator saja (jalankan di PC)")
    e.add_argument("--run_name", default="bs4")
    e.add_argument("--checkpoint", default="last.pt")
    e.add_argument("--output_root", default="outputs")
    e.add_argument("--out", default=None)

    r = sub.add_parser("run", help="jalankan Generator pada gambar")
    r.add_argument("--weights", required=True, help="file hasil export, atau 'none' untuk tes")
    r.add_argument("--images", required=True, help="folder atau file gambar")
    r.add_argument("--n", type=int, default=4, help="jumlah gambar acak jika --images folder")
    r.add_argument("--seed", type=int, default=None, help="isi angka agar pilihan gambar sama tiap kali")
    r.add_argument("--no_show", action="store_true", help="hanya simpan, tanpa membuka jendela")

    args = p.parse_args()
    export(args) if args.cmd == "export" else run(args)


if __name__ == "__main__":
    main()
