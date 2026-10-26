# LOG INTERAKSI LLM - KELOMPOK 9

- **Tanggal Sesi:** 16–26/10/2026
- **Anggota:** Zafira (Model & Training)
- **Target Modul:** Rencana tuning dan evaluasi: `src/evaluate.py`, `src/metrics.py`, `src/compare_runs.py`, `run_experiments.bat`, revisi `src/train.py` dan `src/config.py`
- **LLM:** Claude (claude.ai, proyek "Pix2Pix Gen AI"), lanjutan chat training loop
- **Tautan Sesi Chat:** https://claude.ai/share/76a63b11-5388-4948-939c-83bc384ed692

---

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Rencana Eksperimen Tuning)

- **Prompt (ringkasan):**
  > Training loop sudah lolos tes di GPU. Bagaimana rencana eksperimen tuning-nya: run apa saja yang perlu dijalankan, berapa epoch, dan bagaimana cara membandingkan hasilnya secara adil?

- **Alasan Khusus:** Tugasku termasuk tuning. Aku perlu rencana yang bisa dijelaskan di laporan (kenapa run ini dipilih), bukan sekadar mencoba angka acak.

- **Output Model (ringkasan):**
  - **5 run, masing-masing 200 epoch, hanya satu hyperparameter yang diubah dari baseline:**
    - `baseline`: setting paper
    - `l1_only` (`lambda_adv=0`): ablation, untuk melihat peran Discriminator
    - `lambda10` (`lambda_l1=10`): bobot L1 diperkecil
    - `lr1e-4`: learning rate setengahnya
    - `bs4`: batch size 4, lr tetap 2e-4
  - **Cara membandingkan:** evaluasi `last.pt` tiap run di split **test** dengan L1, PSNR, SSIM, ditambah **ketajaman (variance of Laplacian)**, karena L1/PSNR/SSIM cenderung memberi skor tinggi untuk hasil yang buram.
  - Usul cek tambahan: hasil G dalam mode `eval` vs `train` (BatchNorm/dropout), karena implementasi asli Pix2Pix memakai mode train saat testing.
  - Claude menawarkan membuat `evaluate.py`, script perbandingan, dan file `.bat` untuk menjalankan kelima run.

### 2. TAHAP INISIASI (Kode Evaluasi & Batch Runner)

- **Prompt:**
  > Iya, tolong buatkan evaluate.py, script tabel + gambar perbandingan + grafik gabungan, dan file .bat buat jalanin kelima run. Ukuran last.pt dan best.pt dari dummy run masing-masing 655 MB. Sisa disk C: 91 GB dari 952 GB. Sesuaikan save_every-nya, aku pikir 50 cukup.

- **Alasan Khusus:** Aku menyertakan ukuran checkpoint dan sisa disk supaya Claude bisa menghitung apakah 5 run × 200 epoch muat di disk. Dengan `save_every=10`, tiap run akan menyimpan 20 checkpoint × 655 MB ≈ 13 GB, jadi 5 run ≈ 65 GB, terlalu mepet.

- **Output Model (ringkasan):**
  - **`config.py`:** `save_every` 10 → 50.
  - **`train.py`:** `best.pt` dan `epoch_XXX.pt` sekarang hanya berisi bobot G dan D (tanpa state optimizer Adam), sekitar 1/3 ukuran `last.pt`. Resume tetap dari `last.pt` yang lengkap.
  - **`metrics.py`:** L1, PSNR, SSIM (jendela Gaussian 11×11), dan ketajaman, semuanya ditulis sendiri dengan PyTorch, tanpa library tambahan.
  - **`evaluate.py`:** memuat checkpoint, membangun ulang config dari isi checkpoint, menghitung metrik per gambar di split test, dan menyimpan `metrics.json`, `per_image.csv`, serta semua gambar hasil. Ada opsi `--g_mode train` untuk cek BatchNorm.
  - **`compare_runs.py`:** tabel Markdown siap tempel ke laporan (kolom "Yang diubah" terisi otomatis dari selisih config), grid gambar test yang sama dari semua run, dan grafik val_L1 + D_total semua run.
  - **`run_experiments.bat`:** menjalankan train → evaluate untuk kelima run, lalu menggabungkan hasilnya. Aman dijalankan ulang: run yang sudah selesai dilewati, run yang terputus dilanjutkan. Mode `dummy` untuk tes cepat.

- **Catatan integrasi:** saat menyalin dari chat, `config.py` sempat tertukar dengan isi `evaluate.py`. Ketahuan saat dibandingkan dengan file di repo, lalu disalin ulang.

### 3. TAHAP VERIFIKASI (Data Asli & Batch Dummy)

- **Tes pendek dengan dataset asli** (setelah `dataset.py` dari Fira di-merge):
  ```
  PS> py -m src.train --run_name smoke_real --epochs 2 --decay_start_epoch 1 --max_steps_per_epoch 20
  [info] run=smoke_real device=cuda dummy=False
  [info] GPU: NVIDIA GeForce RTX 5060 Ti
  [info] train=400 val=100 | 400 iterasi/epoch
  [epoch   1/2] D=0.8421 G_adv=0.9981 G_L1=0.3756 val_L1=0.3786 lr=2.00e-04 (27.4s)
  [epoch   2/2] D=0.6601 G_adv=0.8982 G_L1=0.3580 val_L1=0.3698 lr=1.00e-04 (1.2s)
  [selesai] best val_L1=0.3698 | output di outputs\smoke_real
  ```
  Dataset terbaca (400/100), semua loss turun, dan grid sampel menunjukkan label map di kiri dan foto di kanan, jadi `LABEL_ON_LEFT = False` dari Fira sudah benar.

- **Tes batch dengan data dummy:**
  ```
  PS> .\run_experiments.bat dummy
  [25/10/2026 23:29:23,57] Train    dummy_baseline
  [25/10/2026 23:30:17,50] Evaluate dummy_baseline
  ...
  [25/10/2026 23:34:22,00] Evaluate dummy_bs4
  [25/10/2026 23:34:33,69] Menggabungkan hasil...
  Hasil gabungan: outputs\dummy_comparison
  ```
  Kelima run, evaluasi, dan penggabungan berjalan tanpa error, sekitar 5 menit.

### 4. TAHAP PENANGANAN EROR / DEBUGGING

- **Masalah:** saat training asli berjalan, `outputs\_batch_logs\baseline_train.log` kosong, padahal training sudah berjalan lama.
  ```
  PS> Get-Content outputs\_batch_logs\baseline_train.log -Tail 5
  PS>
  ```

- **Penyebab:** saat output `print` dialihkan ke file, Python menahannya di buffer dan baru menulis ke file setelah buffer penuh atau program selesai. Training-nya sendiri berjalan normal; progres tetap terlihat di `outputs\baseline\logs\losses.csv` yang ditulis tiap epoch.

- **Solusi:** tambah `set PYTHONUNBUFFERED=1` di `run_experiments.bat`, supaya log ditulis langsung tiap epoch. Perbaikan ini dibuat lewat Claude Code (asisten coding di laptop), bukan di chat claude.ai.

- **Pelajaran:** tes dummy hanya memeriksa apakah program selesai tanpa error, bukan apakah log bisa dipantau selama berjalan. Masalah seperti ini baru terlihat pada run yang panjang.

- **Kejadian lain:** PC sempat mati saat batch berjalan. Status tiap run dicek dari epoch terakhir di `losses.csv` dan keberadaan `metrics.json`. Ternyata kelima run sudah selesai sebelum PC mati.

### 5. HASIL TUNING

| Run | Yang diubah | L1 ↓ | PSNR (dB) ↑ | SSIM ↑ | Ketajaman | Waktu (menit) |
|---|---|---|---|---|---|---|
| `baseline` | – | 0,3699 | 12,59 | 0,2251 | 0,513 | 39 |
| `l1_only` | λ_adv = 0 | 0,3576 | 12,95 | 0,3178 | 0,039 | 38 |
| `lambda10` | λ_L1 = 10 | 0,3768 | 12,43 | 0,2230 | 0,501 | 39 |
| `lr1e-4` | lr = 1e-4 | 0,3736 | 12,49 | 0,2281 | 0,477 | 38 |
| `bs4` | batch size = 4 | 0,3619 | 12,55 | 0,2470 | 0,569 | 16 |

![Perbandingan hasil semua run](../hasil/comparison_grid.png)

![Grafik gabungan](../hasil/curves_all_runs.png)

- **Analisis Kelompok:**
  - `l1_only` unggul di L1/PSNR/SSIM tapi sangat buram (ketajaman 0,039). Ini membuktikan bahwa Discriminator yang membuat hasil tajam, dan bahwa metrik kemiripan saja tidak cukup untuk menilai hasil.
  - `bs4` adalah run GAN terbaik: metrik terbaik, artefak paling sedikit, dan 2,4× lebih cepat.
  - `lambda10` paling banyak detail, tapi artefak warnanya juga paling banyak.
  - Training lebih cepat dari perkiraan awal: sekitar 39 menit per run (sekitar 12 detik per epoch), bukan 1,5–2 jam.
  - Keterbatasan: hanya 1 seed per run, dan selisih antar run GAN kecil.
  - Analisis lengkap ada di `docs/laporan/draf_hasil_analisis.md`.

---

**Status:** semua eksperimen tuning selesai. Hasil disimpan di `docs/hasil/`.

**Commit terkait (branch `feat/training`, PR #5):**
- `merge main ke feat/training (model + dataset)`
- `feat(training): best.pt dan epoch_XXX.pt hanya simpan bobot, save_every 50`
- `feat(eval): tambah evaluate, metrik L1/PSNR/SSIM/ketajaman, compare_runs, dan run_experiments.bat`
- `fix(bat): log batch ditulis langsung tiap epoch (PYTHONUNBUFFERED)`
- `docs(hasil): tabel, grid perbandingan, dan grafik 5 run tuning`
- `docs(laporan): draf bagian model, training, hasil, dan analisis tuning`
