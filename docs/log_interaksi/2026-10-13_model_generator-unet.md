# LOG INTERAKSI LLM - KELOMPOK 9

- **Tanggal Sesi:** 13/10/2026
- **Anggota:** Zafira (Model & Training)
- **Target Modul:** Generator U-Net — `DownBlock` (encoder) & `UpBlock` (decoder)
- **LLM:** Claude (claude.ai, proyek "Pix2Pix Gen AI")
- **Tautan Sesi Chat:** https://claude.ai/share/a98da975-f146-4377-9c2b-ccfb73e7d4fc

---

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Memahami Arsitektur)

- **Prompt:**
  > Sesuai KONTRAK.md, aku mulai bagian model: Generator U-Net di `src/models/generator.py`. Sebelum nulis kode, jelaskan dulu arsitektur U-Net Pix2Pix: susunan encoder dan decoder-nya, kenapa butuh skip connection, dan di layer mana dipakai dropout. Habis itu baru kita bikin kodenya bagian per bagian.

- **Alasan Khusus:** Biar paham arsitekturnya dulu sebelum ngoding, jadi kalau error tahu letak salahnya.

- **Output Model (ringkasan):**
  - Encoder 8 blok: Conv 4×4, stride 2, padding 1 → BatchNorm → LeakyReLU(0.2). Ukuran 256 → 1.
  - e1 dan bottleneck tanpa BatchNorm.
  - Decoder: ConvTranspose 4×4 → BatchNorm → (Dropout 0.5 di u1–u3) → ReLU, lalu concat dengan skip dari encoder.
  - Layer final: ConvTranspose ke 3 channel + tanh.
  - Skip connection menjaga detail posisi yang hilang di bottleneck.
  - Total sekitar 54 juta parameter.

### 2. TAHAP PENDALAMAN (Pertanyaan Lanjutan)

- **Prompt:**
  > Aku belum ngerti kenapa input channel u2 itu 1024, bukan 512. Bisa jelasin pakai contoh tensor shape tiap langkah?
  >
  > Kalau batch size nanti aku naikkan jadi 4, apakah bottleneck boleh pakai BatchNorm lagi?

- **Alasan Khusus:** Bingung kenapa channel jadi 1024, dan mau tahu dampaknya kalau batch size dinaikkan.

- **Output Model (ringkasan):**
  - `torch.cat(..., dim=1)` menumpuk channel: output u1 (512) + skip e7 (512) = 1024.
  - Batch size 4 secara teknis tidak error, tapi bottleneck tetap tanpa BatchNorm supaya statistik stabil dan eksperimen batch size adil. Tambahan pengaman: `drop_last=True` di DataLoader training.

- **Verifikasi mandiri (dijalankan di PC):**
  ```
  torch.Size([2, 1024, 2, 2]) torch.Size([4, 512, 2, 2])
  ```

### 3. TAHAP PEMBUATAN KODE — `DownBlock`

- **Prompt:**
  > Udah aku coba di PC, hasilnya sesuai. Lanjut ke DownBlock.

- **Output Model:** class `DownBlock(in_channels, out_channels, use_norm=True)` di `src/models/generator.py`.
  - `bias=not use_norm`: bias conv dimatikan kalau ada BatchNorm.
  - Tanpa `inplace=True` di LeakyReLU, karena output dipakai dua kali (blok berikutnya + skip).

### 4. TAHAP PENANGANAN EROR / DEBUGGING

- **Pesan Eror dari Notebook:**
  ```
  ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 512, 1, 1])
  ```

- **Konteks:** error ini **disengaja** untuk membuktikan keputusan arsitektur. Bottleneck dicoba pakai BatchNorm dengan batch size 1 dan input 2×2 → output 1×1.

- **Hasil tes lengkap:**
  ```
  e1   -> torch.Size([2, 64, 128, 128])
  e2   -> torch.Size([2, 128, 64, 64])
  bad  -> ValueError (seperti di atas)
  good -> torch.Size([1, 512, 1, 1])
  ```

- **Analisis & Solusi Kelompok:** BatchNorm butuh lebih dari 1 nilai per channel. Di bottleneck ukurannya 1×1, jadi kalau batch 1 pasti error. Karena itu bottleneck dibuat tanpa BatchNorm.

### 5. TAHAP PEMBUATAN KODE — `UpBlock`

- **Tanggal:** 14/10/2026

- **Prompt:**
  > Udah aku tes DownBlock di PC, semua hasilnya sesuai. Lanjut ke UpBlock.

- **Alasan Khusus:** Setelah encoder jadi, langkah berikutnya decoder. Aku minta blok decoder dibuat terpisah dulu supaya bisa dites sendiri sebelum dirakit jadi Generator utuh, jadi kalau nanti ada error shape gampang dilacak.

- **Output Model (ringkasan):** class `UpBlock(in_channels, out_channels, use_dropout=False)` ditambahkan di `src/models/generator.py`.
  - `ConvTranspose2d` kernel 4, stride 2, padding 1 → ukuran jadi 2× lipat: `(H − 1)·2 − 2 + 4 = 2H`.
  - BatchNorm selalu nyala dan `bias=False`. Aman untuk batch size 1, karena ukuran terkecil di decoder 2×2 (masih ada 4 nilai per channel).
  - Urutan: ConvTranspose → BatchNorm → (Dropout 0.5) → ReLU, sesuai paper.
  - `forward(x, skip)` melakukan concat **di dalam blok** (`torch.cat([x, skip], dim=1)`), supaya cuma ada satu tempat yang melakukan concat.

- **Verifikasi mandiri (dijalankan di PC):**
  ```
  out1 (u1: bottleneck + e7) -> torch.Size([2, 1024, 2, 2])
  out2 (u2: out1 + e6)       -> torch.Size([2, 1024, 4, 4])
  mode train, input sama dipanggil 2x -> torch.equal = False
  mode eval,  input sama dipanggil 2x -> torch.equal = True
  ```

- **Analisis Kelompok:** Shape-nya cocok dengan perhitungan di Tahap 2: output tiap blok decoder selalu 2× channel karena ditempel skip, makanya input u2 = 1024. Tes train/eval membuktikan dropout bekerja sebagai sumber keacakan pengganti noise `z`: di mode train input yang sama menghasilkan output berbeda, sedangkan di mode eval output-nya selalu sama. Ini jadi catatan untuk nanti saat menyimpan sampel gambar di `train.py`, harus diputuskan pakai `G.eval()` atau `G.train()`, dan pilihannya ditulis di laporan.

---

**Commit terkait:**
- `feat(model): tambah DownBlock encoder U-Net` (branch `feat/model`)
- `feat(model): tambah UpBlock decoder U-Net` (branch `feat/model`)
