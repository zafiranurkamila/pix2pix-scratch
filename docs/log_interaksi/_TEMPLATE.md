# TEMPLATE LOG INTERAKSI LLM

> Salin file ini untuk setiap sesi, lalu isi.
> **Nama file:** `YYYY-MM-DD_<peran>_<topik>.md`
> Contoh: `2026-10-14_dataset_augmentasi-crop-flip.md`
> **Simpan di:** `docs/log_interaksi/`
> Satu sesi = satu file. Setiap kali minta kode baru atau memecahkan bug, isi log-nya.

---

# LOG INTERAKSI LLM: KELOMPOK [Nama Kelompok]

- **Anggota:** [Nama, peran]
- **Tanggal Sesi:** [DD/MM/YYYY]
- **Target Modul:** [contoh: Pix2PixDataset / Debugging loss / Struktur U-Net]
- **Tautan Sesi Chat:** [share link chat]

---

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Meminta Kode Awal)
- **Prompt Kelompok:**
  > [tulis prompt spesifik yang dikirim]
- **Alasan Khusus:** [kenapa memakai pendekatan atau instruksi ini?]
- **Output Model:** [ringkasan skrip atau cuplikan kode krusial]

### 2. TAHAP PENANGANAN ERROR / DEBUGGING
*(Ulangi blok ini untuk setiap error. Hapus jika tidak ada error.)*
- **Pesan Error dari Notebook/Terminal:**
  ```
  [salin pesan error secara utuh]
  ```
- **Prompt Perbaikan ke LLM:**
  > [tulis bagaimana menanyakan solusi error ke AI]
- **Analisis & Solusi Kelompok:** [bagaimana AI membantu memperbaiki error, dan apa pelajaran yang diambil?]

### 3. TAHAP OPTIMASI / REFACTOR
*(Opsional)*
- **Prompt:**
  > [...]
- **Perubahan yang dihasilkan:** [...]
- **Hasil setelah perubahan:** [...]

---

## CATATAN PRIBADI
- **Yang sudah dipahami dari kode ini:** [jelaskan dengan kata sendiri, berguna untuk persiapan presentasi]
- **Yang masih belum jelas:** [...]
- **Commit terkait:** [hash atau pesan commit]
