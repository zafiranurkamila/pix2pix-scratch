"""
src/dataset.py — Data pipeline Pix2Pix (CMP Facades)
Pemilik: Fira (lihat KONTRAK.md bagian 3–6)

Setiap file mentah = satu gambar berisi dua gambar berdampingan (kiri | kanan).
Dataset ini memotongnya jadi dua, menerapkan preprocessing, lalu mengembalikan:
    {"input": label map [3,256,256], "target": foto asli [3,256,256]}
float32, range [-1, 1].
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# ⚠️ WAJIB DICEK MANUAL (KONTRAK.md bagian 3). Jalankan file ini sebagai skrip,
# buka gambar hasilnya, lalu sesuaikan nilai ini dan isi kolom di KONTRAK.md.
# True  -> label map ada di sisi KIRI, foto di KANAN
# False -> foto di KIRI, label map di KANAN
LABEL_ON_LEFT = False

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class Pix2PixDataset(Dataset):
    def __init__(self, root: str, split: str = "train",
                 image_size: int = 256, load_size: int = 286):
        self.split = split
        self.image_size = image_size
        self.load_size = load_size
        self.is_train = split == "train"

        folder = Path(root) / split
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder split tidak ditemukan: {folder}")

        # sorted() supaya urutan file selalu sama di semua laptop/OS
        self.files = sorted(p for p in folder.iterdir()
                            if p.suffix.lower() in IMG_EXTENSIONS)
        if not self.files:
            raise RuntimeError(f"Tidak ada file gambar di {folder}")

    def __len__(self) -> int:
        return len(self.files)

    # ---------- langkah-langkah kecil ----------

    @staticmethod
    def _split_pair(img: Image.Image):
        """Potong gambar gabungan jadi (label_map, foto) sesuai LABEL_ON_LEFT."""
        w, h = img.size
        half = w // 2
        left = img.crop((0, 0, half, h))
        right = img.crop((half, 0, w, h))
        return (left, right) if LABEL_ON_LEFT else (right, left)

    def _augment_train(self, inp: Image.Image, tgt: Image.Image):
        """Resize 286 -> random crop 256 -> random flip.
        Parameter acak diundi SEKALI lalu dipakai ke KEDUA gambar."""
        size = [self.load_size, self.load_size]
        inp = TF.resize(inp, size, interpolation=TF.InterpolationMode.BICUBIC)
        tgt = TF.resize(tgt, size, interpolation=TF.InterpolationMode.BICUBIC)

        # Undi posisi crop satu kali (pakai RNG torch -> ikut seed & aman di multi-worker)
        max_offset = self.load_size - self.image_size
        top = int(torch.randint(0, max_offset + 1, (1,)))
        left = int(torch.randint(0, max_offset + 1, (1,)))
        inp = TF.crop(inp, top, left, self.image_size, self.image_size)
        tgt = TF.crop(tgt, top, left, self.image_size, self.image_size)

        # Undi flip satu kali
        if torch.rand(1).item() < 0.5:
            inp = TF.hflip(inp)
            tgt = TF.hflip(tgt)
        return inp, tgt

    def _resize_eval(self, inp: Image.Image, tgt: Image.Image):
        """Val/test: resize langsung ke 256, tanpa augmentasi."""
        size = [self.image_size, self.image_size]
        inp = TF.resize(inp, size, interpolation=TF.InterpolationMode.BICUBIC)
        tgt = TF.resize(tgt, size, interpolation=TF.InterpolationMode.BICUBIC)
        return inp, tgt

    @staticmethod
    def _to_tensor(img: Image.Image) -> torch.Tensor:
        """PIL [0,255] -> tensor float32 [3,H,W] range [-1,1]."""
        x = TF.to_tensor(img)      # sudah membagi 255 -> [0, 1]
        return x * 2.0 - 1.0       # [0, 1] -> [-1, 1]

    # ---------- alur utama ----------

    def __getitem__(self, idx: int) -> dict:
        img = Image.open(self.files[idx]).convert("RGB")
        inp, tgt = self._split_pair(img)

        if self.is_train:
            inp, tgt = self._augment_train(inp, tgt)
        else:
            inp, tgt = self._resize_eval(inp, tgt)

        return {"input": self._to_tensor(inp), "target": self._to_tensor(tgt)}


def to_image_range(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1], untuk menyimpan/menampilkan gambar."""
    return (x + 1.0) / 2.0


# ===== Smoke test (KONTRAK.md bagian 10) — jalan di CPU =====
# Pemakaian:  python -m src.dataset --root data/facades [--workers 0]
if __name__ == "__main__":
    import argparse
    from torchvision.utils import save_image

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/facades")
    parser.add_argument("--out", default="dataset_check.png")
    parser.add_argument("--workers", type=int, default=2,
                        help="num_workers DataLoader; pakai 0 kalau error")
    args = parser.parse_args()

    torch.manual_seed(42)
    rows = []
    for split in ["train", "val", "test"]:
        ds = Pix2PixDataset(args.root, split=split)
        item = ds[0]
        x, y = item["input"], item["target"]

        assert x.shape == (3, 256, 256), f"[{split}] shape input salah: {x.shape}"
        assert y.shape == (3, 256, 256), f"[{split}] shape target salah: {y.shape}"
        assert x.dtype == torch.float32 and y.dtype == torch.float32
        for name, t in [("input", x), ("target", y)]:
            assert t.min() >= -1 and t.max() <= 1, f"[{split}] {name} keluar range"

        print(f"[{split}] {len(ds)} file | input {tuple(x.shape)} "
              f"[{x.min():.2f}, {x.max():.2f}] | target {tuple(y.shape)} "
              f"[{y.min():.2f}, {y.max():.2f}]")
        rows += [x, y]

    # Cek batch dari DataLoader (KONTRAK bagian 10). num_workers > 0 sekaligus
    # menguji apakah Dataset aman dipakai multi-proses (KONTRAK bagian 8 poin 10).
    from torch.utils.data import DataLoader
    loader = DataLoader(Pix2PixDataset(args.root, "train"), batch_size=2,
                        shuffle=True, num_workers=args.workers)
    batch = next(iter(loader))
    for name in ["input", "target"]:
        b = batch[name]
        assert b.ndim == 4 and b.shape[1:] == (3, 256, 256), f"batch {name}: {b.shape}"
        assert b.min() >= -1 and b.max() <= 1, f"batch {name} keluar range"
    print(f"[DataLoader] num_workers={args.workers} | input {tuple(batch['input'].shape)} "
          f"| target {tuple(batch['target'].shape)}")

    # Grid: tiap baris = satu split, kolom kiri = input, kolom kanan = target
    save_image(to_image_range(torch.stack(rows)), args.out, nrow=2)
    print(f"Semua cek lolos. Buka {args.out}: kolom KIRI harus label map warna-warni, "
          f"kolom KANAN harus foto. Kalau terbalik, ubah LABEL_ON_LEFT.")