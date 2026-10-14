# LAPORAN AKHIR PROYEK KELOMPOK
## Vibe Coding Generative AI from Scratch: Pix2Pix (Label Map → Foto Fasad)

**Mata Kuliah:** Generative AI
**Kelompok:** [isi nama kelompok]
**Anggota:**
- Zafira: Prompt Engineer, Model & Training
- Fira: Data Pipeline & Preprocessing Engineer
- Ce: Lead Documentarian & Error Handler

**Repositori:** https://github.com/zafiranurkamila/pix2pix-scratch
**Tanggal:** [isi]

> Catatan penulisan (hapus sebelum dikumpulkan):
> - Tulis dengan kata-kata sendiri. Jangan salin paragraf dari blog atau tutorial (aturan tugas: bebas plagiasi).
> - Penanda `[BAHAN: nama]` = bagian ini menunggu bahan dari orang tersebut.
> - Penanda `[TODO]` = Ce yang menulis.

---

## Abstrak
[TODO, tulis terakhir. 150–200 kata: masalah, metode (Pix2Pix: U-Net + PatchGAN), dataset (CMP Facades), hasil utama, dan kesimpulan singkat.]

---

## BAB 1. PENDAHULUAN

### 1.1 Latar Belakang
[TODO: apa itu image-to-image translation, kenapa penting (contoh: arsitektur, desain, sketsa ke foto), kenapa memilih Pix2Pix.]

### 1.2 Tujuan
[TODO: membangun Pix2Pix dari nol (from scratch) memakai metode vibe coding bersama LLM, melatihnya pada CMP Facades, dan mengevaluasi hasilnya.]

### 1.3 Ruang Lingkup dan Batasan
- Arah translasi: label map → foto fasad
- Ukuran gambar 256 × 256, RGB
- Framework: PyTorch
- Training di PC lokal (RTX 5060 Ti, 16 GB VRAM)
- [TODO: tambahkan batasan lain, misal jumlah epoch, tidak memakai pretrained weights]

### 1.4 Metode Vibe Coding dan Pembagian Tugas
[TODO: jelaskan singkat apa itu vibe coding dan bagaimana kelompok menerapkannya.]

| Anggota | Peran | File/Tanggung jawab | Branch |
|---|---|---|---|
| Zafira | Prompt Engineer, Model & Training | `src/models/`, `config.py`, `losses.py`, `train.py`, `utils.py` | `feat/model`, `feat/training` |
| Fira | Data Pipeline & Preprocessing | `src/dataset.py` | `feat/dataset` |
| Ce | Dokumentasi, Log & Laporan | `docs/`, `README.md` | `docs/laporan` |

---

## BAB 2. LANDASAN TEORI
*(Tulis ringkas dan dengan bahasa sendiri. Cukup 1–2 paragraf per subbab.)*

### 2.1 Generative Adversarial Network (GAN)
[TODO: generator vs discriminator, permainan minimax. Rujukan: Goodfellow dkk., 2014.]

### 2.2 Conditional GAN
[TODO: GAN yang diberi syarat/kondisi. Di Pix2Pix, kondisinya adalah gambar input.]

### 2.3 Pix2Pix
[TODO: translasi gambar dengan data berpasangan. Rujukan: Isola dkk., 2017.]

### 2.4 Generator U-Net dan Skip Connection
[TODO: encoder–decoder, 8 kali downsampling (256 → 1), skip connection menjaga detail. Rujukan: Ronneberger dkk., 2015.]
[BAHAN: Zafira, diagram arsitektur jika ada]

### 2.5 Discriminator PatchGAN
[TODO: menilai per patch (70×70), output grid 30×30, menerima pasangan (label map, gambar).]

### 2.6 Fungsi Loss
[TODO: loss adversarial (BCE with logits) + λ · L1. Jelaskan peran masing-masing: BCE membuat hasil tampak nyata, L1 menjaga kedekatan dengan target.]

---

## BAB 3. METODOLOGI

### 3.1 Dataset (CMP Facades)
[BAHAN: Fira]
- Sumber dan jumlah gambar per split (train / val / test): [isi]
- Format mentah (dua gambar berdampingan) dan hasil pengecekan manual sisi kiri/kanan: [isi]
- Contoh 3–4 pasangan gambar: [lampirkan]

### 3.2 Preprocessing dan Augmentasi
[BAHAN: Fira]
- Train: pisah gambar, resize 286, random crop 256, flip 50%, normalisasi [-1, 1]
- Val/test: resize 256, normalisasi [-1, 1], tanpa augmentasi
- Crop dan flip identik untuk input dan target (alasan: [TODO, tanya Fira])
- Hasil cek shape tensor dan range: [isi]

### 3.3 Arsitektur Model
[BAHAN: Zafira]
- Generator U-Net: [jumlah layer, jumlah parameter]
- Discriminator PatchGAN: [jumlah layer, jumlah parameter]
- Inisialisasi bobot Normal(0, 0.02)

### 3.4 Fungsi Loss dan Training
[BAHAN: Zafira]
- Urutan update (D dulu, lalu G), optimizer Adam, hyperparameter baseline: [lr, beta1, batch size, λ L1, jumlah epoch]
- Seed 42, mekanisme checkpoint dan resume

### 3.5 Lingkungan Eksperimen
| Item | Spesifikasi |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB) |
| OS | Windows 11 |
| Python | 3.14.6 |
| PyTorch | ≥ 2.7, CUDA 12.8 |

### 3.6 Alur Kerja Tim dan Git
[TODO: satu branch per bagian, Pull Request, review silang, format commit. Tempel ringkasan riwayat commit dan PR dari GitHub.]

---

## BAB 4. EKSPERIMEN DAN HASIL

### 4.1 Hasil Training Baseline
[BAHAN: Zafira, grafik loss G dan D per epoch]
[TODO: analisis, apakah D terlalu kuat atau terlalu lemah, apakah loss stabil.]

### 4.2 Eksperimen Tuning
[BAHAN: Zafira]

| run_name | Learning rate | Batch size | λ L1 | Epoch | Loss G akhir | Loss D akhir | SSIM | PSNR | Catatan |
|---|---|---|---|---|---|---|---|---|---|
| baseline | | | | | | | | | |
| | | | | | | | | | |

[TODO: analisis, pengaruh tiap hyperparameter terhadap hasil.]

### 4.3 Hasil Visual
[BAHAN: Fira (fungsi visualisasi), Zafira (checkpoint)]
- Perkembangan hasil per epoch (epoch awal, tengah, akhir): [lampirkan]
- Contoh hasil pada data test, format: input | output | target: [lampirkan]
- Contoh kasus yang berhasil dan yang gagal beserta analisisnya: [TODO]

### 4.4 Evaluasi Kuantitatif
[TODO: dikerjakan Ce jika sempat] SSIM dan PSNR pada test set, dihitung dari checkpoint terbaik.

---

## BAB 5. LOG INTERAKSI LLM DAN PENANGANAN ERROR

### 5.1 Ringkasan Sesi
[TODO: tabel dikompilasi Ce dari `docs/log_interaksi/`]

| No | Tanggal | Anggota | Modul | Link chat | File log |
|---|---|---|---|---|---|
| 1 | 13/10/2026 | Zafira | Kontrak & interface | [link] | |
| 2 | | Fira | Dataset | [link] | |
| 3 | | Ce | | [link] | |

### 5.2 Rekap Error dan Solusi
[TODO]

| No | Pesan error (ringkas) | Penyebab | Prompt perbaikan | Solusi | Pelajaran |
|---|---|---|---|---|---|
| 1 | | | | | |

### 5.3 Pola Prompting yang Efektif
[TODO: contoh variasi prompt (inisiasi, debugging, optimasi) dan apa yang membuat prompt efektif atau gagal. Bagian ini mendukung penilaian "Kualitas Vibe Coding".]

---

## BAB 6. KONTRIBUSI DAN KERJA SAMA TIM
[TODO]
- Kontribusi tiap anggota (sesuai job desk dan bukti commit)
- Cara integrasi: kontrak teknis (`KONTRAK.md`), review silang, smoke test
- Kendala kerja sama dan cara mengatasinya (contoh: perubahan kontrak v1.1 → v1.2)

---

## BAB 7. KESIMPULAN DAN SARAN
[TODO, tulis terakhir]
- Kesimpulan: apakah tujuan tercapai, temuan utama dari eksperimen
- Keterbatasan
- Saran pengembangan (contoh: dataset lebih besar, resolusi lebih tinggi, loss tambahan)

---

## DAFTAR PUSTAKA
- Goodfellow, I. dkk. (2014). *Generative Adversarial Nets.* NeurIPS.
- Isola, P., Zhu, J.-Y., Zhou, T., Efros, A. A. (2017). *Image-to-Image Translation with Conditional Adversarial Networks.* CVPR.
- Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
- Tyleček, R., Šára, R. (2013). *Spatial Pattern Templates for Recognition of Objects with Regular Structure.* (dataset CMP Facade)
- Dokumentasi PyTorch: https://pytorch.org/docs/

*(Cek ulang detail sitasi sebelum dikumpulkan.)*

---

## LAMPIRAN
- A. Tautan semua sesi chat LLM
- B. Log Interaksi LLM gabungan
- C. Riwayat commit dan Pull Request
- D. Kontrak teknis (`KONTRAK.md`)

---

## CHECKLIST BAHAN YANG HARUS DIKUMPULKAN CE

| Dari | Bahan | Status |
|---|---|---|
| Fira | Hasil cek sisi kiri/kanan dataset | ☐ |
| Fira | Jumlah gambar per split + contoh pasangan | ☐ |
| Fira | Contoh hasil fungsi visualisasi | ☐ |
| Fira | Link chat + file log | ☐ |
| Zafira | Grafik loss (baseline + tiap eksperimen) | ☐ |
| Zafira | Tabel hasil tuning | ☐ |
| Zafira | Sampel gambar per epoch | ☐ |
| Zafira | Akses Google Drive (checkpoint, hasil) | ☐ |
| Zafira | Link chat + file log | ☐ |
| Ce | Link chat + file log sendiri | ☐ |
