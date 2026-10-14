"""Dataset sintetis untuk ngetes train.py sebelum dataset.py dari Fira masuk.

Signature dan format output-nya sama persis dengan Pix2PixDataset di KONTRAK:
    {"input": Tensor[3,256,256], "target": Tensor[3,256,256]}, float32, [-1, 1]

Bukan noise acak: "input" berupa kotak-kotak warna (mirip label map), dan
"target" adalah fungsi pasti dari input (warna dipetakan ke warna lain +
pola garis). Karena ada hubungan yang bisa dipelajari, G_L1 dan val_L1
HARUS turun kalau training loop-nya benar. Kalau tidak turun, ada bug.
"""
import torch
from torch.utils.data import Dataset

# 6 "kelas" label dan warna foto pasangannya (nilai sudah di [-1, 1])
_LABEL_COLORS = torch.tensor([
    [-1.0, -1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, 1.0],
])
_PHOTO_COLORS = torch.tensor([
    [0.6, 0.4, 0.2], [-0.2, -0.3, -0.4], [0.9, 0.9, 0.8],
    [-0.6, -0.1, 0.3], [0.2, 0.5, -0.5], [-0.8, -0.8, -0.7],
])


class DummyPix2PixDataset(Dataset):
    def __init__(self, root: str = "", split: str = "train",
                 image_size: int = 256, load_size: int = 286,
                 length: int | None = None):
        self.image_size = image_size
        self.length = length or {"train": 40, "val": 8, "test": 8}.get(split, 8)
        # offset seed per split supaya train dan val berisi gambar berbeda
        self.offset = {"train": 0, "val": 10_000, "test": 20_000}.get(split, 0)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        # Generator lokal: item yang sama selalu menghasilkan gambar yang sama,
        # tidak terpengaruh seed global atau worker DataLoader.
        g = torch.Generator().manual_seed(self.offset + idx)
        s = self.image_size
        label = torch.zeros(s, s, dtype=torch.long)
        for _ in range(8):  # tumpuk 8 kotak acak
            h, w = torch.randint(s // 8, s // 2, (2,), generator=g).tolist()
            y, x = torch.randint(0, s - h, (1,), generator=g).item(), \
                torch.randint(0, s - w, (1,), generator=g).item()
            label[y:y + h, x:x + w] = torch.randint(0, 6, (1,), generator=g).item()

        inp = _LABEL_COLORS[label].permute(2, 0, 1)        # [3, s, s]
        target = _PHOTO_COLORS[label].permute(2, 0, 1)     # [3, s, s]
        stripes = (torch.arange(s) % 16 < 8).float() * 0.2 - 0.1
        target = (target + stripes.view(1, s, 1)).clamp(-1.0, 1.0)

        return {"input": inp.float().contiguous(),
                "target": target.float().contiguous()}
