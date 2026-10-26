# LOG INTERAKSI LLM - KELOMPOK [Nama Kelompok]

- **Tanggal Sesi:** 15/10/2026 – 24/10/2026
- **Anggota / Peran:** Fira — Data Pipeline & Preprocessing
- **Target Modul:** `src/dataset.py` — class `Pix2PixDataset` (CMP Facades)
- **Tautan Sesi Chat:** https://claude.ai/share/9edb4980-7a34-4e2c-af0d-2d1962118bae

---

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Meminta Kode Awal)

- **Prompt Kelompok:**
  "Tolong bikin src/dataset.py sesuai KONTRAK.md bagian 3–6: class `Pix2PixDataset`, return dict `{"input", "target"}`, ukuran 256×256, range [-1, 1], augmentasi train (resize 286 → random crop 256 → flip) yang sama persis untuk input dan target. Jelaskan dulu alurnya sebelum kode."

- **Alasan Khusus:**
  - Prompt merujuk ke KONTRAK.md supaya output cocok dengan signature yang dipakai `train.py` (key dict, shape, range).
  - Syarat "augmentasi sama persis untuk input dan target" ditulis eksplisit karena bug ini **tidak memunculkan error**. Kalau crop/flip diundi terpisah, pasangan gambar rusak diam-diam.
  - Minta penjelasan alur dulu supaya saya paham kodenya untuk presentasi.

- **Output Model (ringkasan):**
  Alur `__getitem__`: buka gambar → `.convert("RGB")` → potong jadi dua (`_split_pair`) → train: resize 286, random crop 256, flip / val-test: resize 256 → `to_tensor` lalu `x*2-1` ke [-1, 1].

  Cuplikan krusial — parameter acak diundi **sekali**, lalu dipakai ke kedua gambar:
  ```python
  top  = int(torch.randint(0, max_offset + 1, (1,)))
  left = int(torch.randint(0, max_offset + 1, (1,)))
  inp = TF.crop(inp, top, left, self.image_size, self.image_size)
  tgt = TF.crop(tgt, top, left, self.image_size, self.image_size)
  if torch.rand(1).item() < 0.5:
      inp = TF.hflip(inp)
      tgt = TF.hflip(tgt)
  ```
  RNG memakai `torch` (bukan `random` Python) supaya ikut `torch.manual_seed(42)` dan aman saat `DataLoader` multi-worker.

- **Revisi (24/10/2026):** kontrak naik ke v1.2 (training pindah ke PC Windows Zafira). Smoke test ditambah cek batch `DataLoader` dengan opsi `--workers`, karena `num_workers > 0` memakai multi-proses (KONTRAK bagian 8 poin 10).

### 2. TAHAP PENANGANAN EROR / DEBUGGING

- **Gejala dari Terminal:** smoke test selesai tanpa output dan tanpa pesan error.
  ```
  (.venv) safira@MacBook-Air-safira pix2pix-scratch % python -m src.dataset --root data/facades
  (.venv) safira@MacBook-Air-safira pix2pix-scratch %
  ```
- **Prompt Perbaikan ke LLM:** menempelkan output terminal di atas apa adanya, lalu menanyakan kenapa tidak ada output.
- **Analisis & Solusi Kelompok:**
  - LLM menganalisis dulu sebelum memberi solusi: kalau masalahnya library atau path, Python pasti memunculkan error. Program yang selesai tanpa output berarti file-nya kosong, atau blok `if __name__ == "__main__":` (bagian yang mencetak hasil tes) tidak ikut.
  - Dicek dengan `wc -l src/dataset.py` → hasilnya `0`. File memang masih kosong, karena kodenya belum di-paste/di-save ke repo.
  - Solusi: paste ulang kode lengkap, simpan, cek lagi → `157 src/dataset.py`, lalu smoke test lolos.
  - Pelajaran: "tidak ada error" tidak sama dengan "berhasil". Smoke test harus mencetak output yang jelas supaya kegagalan diam seperti ini ketahuan.

---

## HASIL PENGECEKAN DATASET (untuk KONTRAK bagian 3)

- Sisi kiri = foto asli , Sisi kanan = label map
- Nilai `LABEL_ON_LEFT` yang dipakai: `False` (dicek visual lewat `dataset_check.png`)
- Jumlah file: train 400, val 100, test 106
- Output smoke test:
  ```
  [train] 400 file | input (3, 256, 256) [-1.00, 1.00] | target (3, 256, 256) [-1.00, 0.90]
  [val] 100 file | input (3, 256, 256) [-1.00, 1.00] | target (3, 256, 256) [-1.00, 1.00]
  [test] 106 file | input (3, 256, 256) [-1.00, 1.00] | target (3, 256, 256) [-1.00, 1.00]
  [DataLoader] num_workers=2 | input (2, 3, 256, 256) | target (2, 3, 256, 256)
  Semua cek lolos.
  ```
- Catatan: target train max `0.90` wajar. Batasnya hanya "tidak lewat ±1"; sampel yang di-crop itu tidak punya piksel putih murni.