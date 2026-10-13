# KONTRAK PROYEK — Pix2Pix from Scratch

> Dokumen ini berisi kesepakatan teknis tim supaya pekerjaan tiga orang bisa digabung tanpa bentrok.
> **Aturan utama:** kalau ada yang mau mengubah isi kontrak ini, diskusikan dulu di grup, lalu update dokumen ini lewat Pull Request. Jangan ubah diam-diam di kode.

- **Versi:** 1.2
- **Tanggal disepakati:** 13/10/2026
- **Anggota:** Zafira (Model & Training), Fira (Dataset), Ce (Dokumentasi & Laporan)
- **Framework:** PyTorch
- **Lingkungan training:** PC lokal Zafira, GPU NVIDIA GeForce RTX 5060 Ti (detail di bagian 8)

---

## 1. Ringkasan Keputusan

| Hal | Keputusan |
|---|---|
| Model | Pix2Pix (Conditional GAN): Generator U-Net + Discriminator PatchGAN 70×70 |
| Dataset | CMP Facades |
| Arah translasi | **Label map → foto fasad** |
| Ukuran gambar | 256 × 256, RGB (3 channel) |
| Range nilai tensor | **[-1, 1]** |
| Format item dataset | `dict` dengan key `"input"` dan `"target"` |
| Output Discriminator | Logits mentah `[B, 1, 30, 30]`, **tanpa sigmoid** |
| Loss | `BCEWithLogitsLoss` (adversarial) + L1 × λ |
| Alur Git | Satu branch per bagian kerja, merge ke `main` lewat Pull Request |

---

## 2. Pembagian Peran & Kepemilikan File

| Peran | Nama | File yang dipegang | Branch |
|---|---|---|---|
| Prompt Engineer, Model & Training (U-Net, PatchGAN, loss, training loop, tuning) | Zafira | `src/models/`, `src/config.py`, `src/losses.py`, `src/train.py`, `src/utils.py`, `notebooks/` | `feat/model`, `feat/training` |
| Data Pipeline & Preprocessing | Fira | `src/dataset.py` | `feat/dataset` |
| Dokumentasi, Log Interaksi & Laporan | Ce | `docs/`, `README.md` | `docs/laporan` |

Pemilik file adalah orang yang paling paham isinya dan yang mereview perubahan di file tersebut.

**Catatan:** Zafira memakai **dua branch terpisah** (`feat/model` dan `feat/training`) walaupun dikerjakan satu orang. Tujuannya supaya Pull Request tetap kecil dan riwayat Git menunjukkan dengan jelas kapan model selesai dan kapan training dimulai.

**Review silang:** karena sebagian besar kode ada di Zafira, PR dari Zafira direview oleh Fira atau Ce, dan PR dari Fira direview oleh Zafira. Dengan begitu semua anggota ikut memahami kode untuk presentasi.

---

## 3. Dataset

- **Nama:** CMP Facades (versi dataset Pix2Pix).
- **Split:** `train/`, `val/`, `test/`.
- **Format mentah:** setiap file adalah satu gambar yang berisi **dua gambar berdampingan** (kiri | kanan): satu label map warna-warni, satu foto asli.
- ⚠️ **Wajib dicek manual:** buka beberapa gambar dan pastikan sisi mana yang label map dan mana yang foto. **Jangan berasumsi.** Hasil pengecekan ditulis di bagian ini:

  > Sisi kiri = ____________ , Sisi kanan = ____________ (diisi Fira)

- `Dataset` harus mengembalikan:
  - `"input"` → **label map** (kondisi)
  - `"target"` → **foto asli** (yang ingin dihasilkan)

---

## 4. Preprocessing & Augmentasi

### Split train
1. Pisahkan gambar menjadi dua (input dan target).
2. Resize keduanya ke **286 × 286**.
3. Random crop ke **256 × 256**.
4. Random horizontal flip (peluang 50%).
5. Ubah ke tensor dan normalisasi ke [-1, 1].

> ⚠️ **ATURAN KRITIS:** posisi crop dan keputusan flip **harus sama persis** untuk input dan target. Kalau input di-flip tapi target tidak, pasangannya rusak dan model belajar hal yang salah **tanpa memunculkan error apa pun**.

### Split val & test
1. Pisahkan gambar menjadi dua.
2. Resize langsung ke **256 × 256**.
3. Ubah ke tensor dan normalisasi ke [-1, 1].
4. **Tanpa augmentasi.**

### Rumus normalisasi
```python
# piksel [0, 255] -> tensor [-1, 1]
x = x / 255.0 * 2.0 - 1.0

# tensor [-1, 1] -> [0, 1], untuk menyimpan/menampilkan gambar
x = (x + 1.0) / 2.0
```

Alasan pakai [-1, 1]: Generator diakhiri `tanh` yang menghasilkan nilai di range [-1, 1]. Target harus berada di range yang sama supaya L1 loss membandingkan dua hal yang skalanya setara.

---

## 5. Format Tensor

| Titik | Shape | Dtype | Range |
|---|---|---|---|
| Item dari `Dataset` | `{"input": [3,256,256], "target": [3,256,256]}` | float32 | [-1, 1] |
| Batch dari `DataLoader` | `{"input": [B,3,256,256], "target": [B,3,256,256]}` | float32 | [-1, 1] |
| Output Generator | `[B, 3, 256, 256]` | float32 | [-1, 1] (tanh) |
| Input Discriminator (setelah concat di dalam D) | `[B, 6, 256, 256]` | float32 | [-1, 1] |
| Output Discriminator | `[B, 1, 30, 30]` | float32 | Logits mentah (bebas) |

> ⚠️ **Discriminator TIDAK BOLEH memakai sigmoid di layer terakhir.**
> Bagian training memakai `BCEWithLogitsLoss`, yang sudah menggabungkan sigmoid + BCE secara stabil secara numerik. Kalau D juga memakai sigmoid, terjadi sigmoid ganda: training tetap jalan tanpa error, tapi hasilnya rusak diam-diam.

Catatan: training loop **tidak meng-hardcode** ukuran 30×30. Label real/fake dibuat dengan `torch.ones_like(pred)` / `torch.zeros_like(pred)`.

---

## 6. Kontrak Kode (Signature)

Signature di bawah ini **wajib dipatuhi**. Isi implementasinya bebas, asal input/output-nya sesuai.

```python
# ===== src/dataset.py — Fira =====
class Pix2PixDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str = "train",
                 image_size: int = 256, load_size: int = 286): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict:
        # return {"input": Tensor[3,256,256], "target": Tensor[3,256,256]}
        # float32, range [-1, 1]
        ...


# ===== src/models/generator.py — Zafira =====
class UNetGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 features: int = 64): ...
    def forward(self, x: Tensor) -> Tensor:
        # [B,3,256,256] -> [B,3,256,256], layer terakhir tanh
        ...


# ===== src/models/discriminator.py — Zafira =====
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, features: int = 64): ...
        # in_channels = channel per gambar; conv pertama menerima in_channels * 2
    def forward(self, cond: Tensor, img: Tensor) -> Tensor:
        # concat [cond, img] di dim=1 -> [B,6,256,256]
        # return logits [B,1,30,30], TANPA sigmoid
        ...


def init_weights(model: nn.Module) -> None:
    # Bobot Conv/ConvTranspose ~ Normal(0, 0.02); BatchNorm weight ~ Normal(1, 0.02), bias = 0
    ...
```

**Kenapa D menerima dua argumen?** Ini yang membuat GAN-nya *conditional*. D tidak hanya menilai "apakah ini foto asli", tetapi "apakah foto ini pasangan yang masuk akal untuk label map ini". Concat dilakukan **di dalam D** supaya hanya ada satu tempat yang melakukan concat, dan training loop tinggal memanggil `D(cond, img)`.

**Kenapa ukuran harus 256?** U-Net Pix2Pix melakukan 8 kali downsampling (256 → 1). Ukuran lain membuat shape skip connection tidak cocok saat digabung.

---

## 7. Struktur Folder

```
pix2pix-scratch/
├── README.md               ← Ce
├── KONTRAK.md              ← dokumen ini (perubahan disepakati bersama)
├── requirements.txt        ← Zafira
├── .gitignore              ← Zafira
├── src/
│   ├── __init__.py
│   ├── config.py           ← Zafira
│   ├── dataset.py          ← Fira
│   ├── models/
│   │   ├── __init__.py
│   │   ├── generator.py    ← Zafira
│   │   └── discriminator.py← Zafira
│   ├── losses.py           ← Zafira
│   ├── train.py            ← Zafira
│   └── utils.py            ← Zafira
├── notebooks/              ← Zafira; opsional, hanya untuk eksplorasi & visualisasi hasil
├── data/                   ← tidak di-commit
│   └── facades/{train,val,test}/
├── outputs/                ← tidak di-commit (disimpan di disk PC, hasil penting di-backup ke Drive)
│   └── <run_name>/{checkpoints,samples,logs}/
└── docs/                   ← Ce
    ├── log_interaksi/      ← log LLM tiap anggota (dikompilasi Ce)
    └── laporan/
```

**Prinsip:** semua logika ditulis di file `.py`. Training dijalankan dari terminal (`python -m src.train`), dan notebook hanya untuk melihat atau memvisualisasikan hasil. Alasannya, notebook susah dibaca perubahannya di Git, sementara kontribusi tiap anggota dinilai dari commit. Sebelum commit notebook, **clear semua output** supaya file tidak membengkak.

---

## 8. Lingkungan Training (PC Lokal Zafira)

| Item | Spesifikasi |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti (arsitektur Blackwell, compute capability `sm_120`) |
| VRAM | 16 GB |
| Sistem operasi | Windows 11 |
| Python | 3.14.6 |
| PyTorch | **≥ 2.7, build CUDA 12.8 (`cu128`) atau lebih baru** |

### Aturan
1. **Semua run yang masuk laporan dijalankan di PC Zafira.** Dengan hardware yang sama, waktu training dan hasil antar eksperimen bisa dibandingkan dengan adil.
2. ⚠️ **PyTorch wajib build CUDA 12.8 ke atas.** RTX seri 50 memakai arsitektur baru (`sm_120`) yang belum didukung build PyTorch lama. Gejalanya menipu: `torch.cuda.is_available()` tetap `True`, tapi operasi pertama di GPU gagal dengan error `no kernel image is available for execution on the device`. Install PyTorch memakai perintah dari [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) dan pilih CUDA 12.8 atau lebih baru.
3. **Cek GPU sebelum training pertama:**
   ```python
   import torch
   print(torch.__version__, torch.version.cuda)          # versi CUDA harus >= 12.8
   print(torch.cuda.get_device_name(0))
   print(torch.cuda.get_device_capability(0))            # RTX 5060 Ti -> (12, 0)
   x = torch.randn(1024, 1024, device="cuda")
   print((x @ x).sum().item())                           # memastikan kernel GPU benar-benar jalan
   ```
4. **Pakai virtual environment khusus proyek** (venv atau conda). Versi paket yang dipakai dicatat di `requirements.txt`.
5. **Dataset disimpan di `data/facades/`** di dalam folder repo.
6. **Output disimpan di `outputs/<run_name>/`.** Training harus bisa di-*resume* dari checkpoint terakhir, untuk jaga-jaga kalau listrik padam, PC restart, atau training sengaja dihentikan lalu dilanjutkan.
7. **Backup ke Google Drive kelompok:** checkpoint terbaik, grafik loss, dan sampel gambar tiap run. Ce butuh akses ke file ini untuk laporan.
8. **Seed = 42** untuk semua eksperimen, supaya hasil tuning bisa dibandingkan dengan adil.
9. Setiap eksperimen tuning memakai `run_name` yang berbeda (contoh: `baseline`, `lr1e-4`, `bs4`), sehingga hasilnya tersimpan di folder terpisah.
10. **Kalau OS-nya Windows:** `DataLoader` dengan `num_workers > 0` butuh blok `if __name__ == "__main__":` di `train.py` (akan dipasang di sana). Kalau tetap error, set `num_workers=0`.
11. **Fira dan Ce tidak butuh GPU.** Smoke test dataset bisa dijalankan di CPU laptop masing-masing.

---

## 9. Alur Git

- Branch: `feat/model` dan `feat/training` (Zafira), `feat/dataset` (Fira), `docs/laporan` (Ce).
- Urutan merge yang disarankan: `feat/dataset` dan `feat/model` lebih dulu (bisa paralel), lalu `feat/training`, karena training butuh dataset dan model.
- Jangan push langsung ke `main`. Semua masuk lewat **Pull Request**, minimal dilihat satu anggota lain.
- Format pesan commit:
  ```
  feat(dataset): tambah augmentasi random crop dan flip
  fix(model): perbaiki shape skip connection di decoder
  docs(log): tambah log interaksi sesi debugging loss
  exp(train): eksperimen lr 1e-4 batch size 4
  ```
- Sebelum membuat PR, jalankan tes shape untuk file milik sendiri (lihat bagian 10).

---

## 10. Checklist Smoke Test

### Sebelum push (per orang)
- [ ] **Dataset:** ambil 1 item, cek shape `[3,256,256]` untuk input dan target, cek `min() >= -1` dan `max() <= 1`, simpan 1 contoh gambar dan cek pasangannya masih sejajar setelah augmentasi.
- [ ] **Generator:** input acak `[2,3,256,256]` → output `[2,3,256,256]`, nilai di [-1, 1].
- [ ] **Discriminator:** dua input acak `[2,3,256,256]` → output `[2,1,30,30]`.

### Setelah digabung (sebelum training beneran)
- [ ] Ambil 1 batch dari `DataLoader`, cek shape dan range.
- [ ] `fake = G(batch["input"])` → `[B,3,256,256]`.
- [ ] `D(batch["input"], fake)` → `[B,1,30,30]`.
- [ ] Hitung loss G dan D sekali, pastikan bukan NaN.
- [ ] Jalankan 1 epoch penuh, pastikan checkpoint dan sampel tersimpan di `outputs/<run_name>/`.
- [ ] Hentikan training lalu jalankan lagi, pastikan training lanjut dari epoch terakhir (resume berfungsi).

---

## 11. Log Interaksi LLM

- **Setiap anggota mencatat log sesi LLM-nya sendiri** memakai template dari panduan tugas, disimpan di `docs/log_interaksi/`.
- Nama file: `YYYY-MM-DD_<peran>_<topik>.md`, contoh: `2026-10-13_training_kontrak-interface.md`.
- Setiap log wajib menyertakan **tautan share chat** LLM.
- Pesan error dari terminal atau notebook disalin **utuh** ke log, beserta prompt perbaikan dan analisis solusinya.
- **Ce** mengompilasi semua log menjadi Log Interaksi akhir kelompok dan menyusun laporan akhir.
- Supaya Ce bisa menulis laporan tanpa menebak, Zafira dan Fira mengirimkan ke Ce: grafik loss, contoh hasil gambar per epoch, dan tabel hasil eksperimen tuning.

---

## 12. Riwayat Perubahan Kontrak

| Versi | Tanggal | Perubahan | Disetujui oleh |
|---|---|---|---|
| 1.0 | 13/10/2026 | Kesepakatan awal | Semua anggota |
| 1.1 | 13/10/2026 | Pembagian peran diperbarui: Zafira pegang model + loss + training, Fira pegang dataset, Ce pegang dokumentasi & laporan | Semua anggota |
| 1.2 | 13/10/2026 | Lingkungan training pindah dari Google Colab (T4) ke PC lokal Zafira (RTX 5060 Ti); notebook jadi opsional | Semua anggota |
