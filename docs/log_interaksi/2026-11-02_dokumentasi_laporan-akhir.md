# LOG INTERAKSI LLM: KELOMPOK 9

- **Anggota:** Ce, Lead Documentarian & Error Handler
- **Tanggal Sesi:** 13/10/2026 – 04/11/2026 (satu percakapan yang berlanjut beberapa hari)
- **Target Modul:** Dokumentasi: kerangka laporan, template log, kompilasi hasil dan log ke laporan akhir, serta alur Git untuk dokumentasi
- **Tautan Sesi Chat:**(https://claude.ai/share/6fb00385-30af-43ea-a6c2-41b35085de79)

---

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Meminta Bantuan Awal)
- **Prompt Kelompok:**
  > Saya melampirkan panduan tugas kelompok (PDF) dan pembagian job desk Pix2Pix, lalu minta dibantu mengerjakan tugas pertama saya (merge PR kontrak, setup Git, bikin branch `docs/laporan`, menyiapkan kerangka laporan), membantu memahami apa yang dibuat kelompok, dan memahami job desk saya sebagai Lead Documentarian.
- **Alasan Khusus:** Sebagai pemegang dokumentasi dan laporan, saya perlu memahami proyek (arsitektur, kontrak teknis) lebih dulu supaya bisa mendokumentasikannya dengan benar, dan perlu kerangka laporan sejak awal supaya bahan dari Zafira dan Fira bisa dikumpulkan bertahap.
- **Output Model:** LLM membaca `KONTRAK.md` dan PR kontrak di GitHub, lalu menjelaskan Pix2Pix (U-Net, PatchGAN, loss BCE + L1) dan keputusan kontrak. LLM juga membuat file `laporan_kerangka.md` (kerangka Bab 1–7 lengkap dengan checklist bahan) dan `log_template.md`.

### 2. TAHAP PENGGUNAAN GIT UNTUK DOKUMENTASI
- **Prompt:**
  > Minta penjelasan lebih detail, step by step, cara membuat branch dan memasukkan file ke folder repo (file sudah saya download).
  > Menempelkan output terminal setelah `git push`, lalu bertanya "link PR nya tuh yang mana".
  > Bertanya cara mereview PR #5 dan di mana tombol "Review changes" berada.
  > Mengirim dua screenshot "Files changed" PR #5 untuk dicek apakah isinya aman di-approve.
- **Hasil:** Branch `docs/laporan` dibuat, dua file ditambahkan lewat dua commit terpisah, dipush, dan Pull Request dibuat. PR #5 (training) saya review, approve, dan merge setelah mengecek bahwa tidak ada file checkpoint (`.pt`) atau folder `outputs/` di dalamnya. Saya juga belajar bahwa tautan PR baru muncul setelah tombol "Create pull request" diklik.

### 3. TAHAP PENYUSUNAN LAPORAN AKHIR
- **Prompt:**
  > Minta cara mengerjakan laporan dan bertanya apakah bisa memakai Google Docs, lalu memilih membuat versi dokumen dulu sebelum dimasukkan ke repo.
  > Meminta LLM langsung membuatkan laporan dalam format .docx dari kerangka dan draf hasil analisis Zafira (`draf_hasil_analisis.md`). Foto/gambar saya tambahkan sendiri, LLM cukup memberi keterangan lokasi file-nya.
- **Perubahan yang dihasilkan:** LLM membuat `laporan_akhir.docx` (sekitar 22 halaman) dengan Bab 3.3–3.5 dan Bab 4 diisi dari draf Zafira, Bab 1, 2, 7, dan abstrak sebagai draf awal, kotak penanda untuk tiap gambar beserta lokasinya di repo, dan teks kuning untuk bagian yang menunggu data dari anggota lain.
- **Hasil setelah perubahan:** Saya melengkapi bagian yang kosong, menyisipkan gambar dari `docs/hasil/`, menyesuaikan teks dengan bahasa sendiri, lalu mengekspor `laporan_akhir.pdf`. PDF akhir (27 halaman) tidak lagi memiliki sisa penanda.

### 4. TAHAP KOMPILASI LOG DAN PENGUMPULAN BAHAN
- **Prompt:**
  > Meminta panduan step by step mengisi Bab 5.2 (rekap error) dan 5.3 (pola prompting) karena bingung mencarinya di mana.
  > Mengirim screenshot bagian "Tahap Penanganan Error" dari log tuning, karena hanya itu yang ada.
  > Bertanya dari mana nomor PR didapat, dan apa saja yang perlu diminta ke Fira dan Zafira.
- **Hasil:** Dari log ditemukan satu kendala, yaitu file log training kosong karena output `print` tertahan di buffer. Baris tabel 5.2 diisi dari kejadian ini. Daftar bahan yang diminta ke Fira dan Zafira dikirim ke grup.

---

## TAHAP PENANGANAN ERROR / DEBUGGING

Tidak ada pesan error yang muncul selama sesi ini: semua perintah Git (`git checkout`, `git add`, `git commit`, `git push`) berjalan berhasil. Pertanyaan yang muncul bersifat prosedural (letak tombol, letak file, arti istilah), bukan error program, sehingga tidak dicatat sebagai debugging.

Satu hal yang saya pelajari dari output `git status`: file `.DS_Store` (file sampah bawaan macOS) muncul sebagai *untracked*. Itu bukan error, tetapi saya memilih menambahkan file satu per satu dengan `git add <nama file>` dan tidak memakai `git add .`, supaya file tersebut tidak ikut ter-commit.

---

## CATATAN PRIBADI
-- **Yang sudah dipahami:**
  - **Kenapa branch dipisah dan PR direview.** Setiap anggota bekerja di branch sendiri supaya pekerjaan tidak saling menimpa dan tidak merusak `main`. Pull Request membuat perubahan dilihat anggota lain sebelum digabung, jadi kesalahan bisa ketahuan lebih awal dan semua anggota ikut paham kode yang masuk. Dengan commit kecil dan sering, riwayat Git juga menunjukkan siapa mengerjakan apa dan kapan.
  - **Kenapa Discriminator tidak memakai sigmoid.** Loss yang dipakai adalah `BCEWithLogitsLoss`, yang sudah menyertakan sigmoid di dalamnya. Kalau Discriminator juga memakai sigmoid, sigmoid dipakai dua kali. Training tetap jalan tanpa error, tetapi hasilnya jadi rusak secara diam-diam.
  - **Kenapa crop dan flip harus sama untuk input dan target.** Label map dan foto adalah pasangan yang posisinya harus sejajar (jendela di label map harus ada di tempat yang sama di foto). Kalau input di-flip tetapi target tidak, pasangannya tidak lagi cocok dan model belajar hal yang salah, tanpa ada pesan error.
  - **Kenapa `l1_only` skornya terbaik tetapi hasilnya buram.** L1, PSNR, dan SSIM mengukur seberapa mirip hasil dengan foto asli. Tanpa loss adversarial, generator mencari "rata-rata" dari semua kemungkinan foto, yang aman untuk skor tetapi hasilnya buram (ketajaman hanya 0,039). Jadi skor kemiripan saja tidak cukup untuk menilai model generatif, harus dibaca bersama ketajaman dan hasil visual.
  - **Kenapa `bs4` paling bagus di antara model GAN.** Dengan batch size 4, gradien dan statistik BatchNorm dihitung dari 4 gambar sekaligus sehingga lebih stabil dibanding dari 1 gambar. Hasilnya L1, SSIM, dan ketajaman terbaik di antara run yang memakai GAN, dan training 2,4 kali lebih cepat (16 menit dibanding sekitar 39 menit). Kesimpulan ini masih indikatif karena tiap konfigurasi hanya dijalankan dengan satu seed.
- **Yang masih belum jelas:** tidak ada
- **Bagian yang dibantu LLM dan yang saya kerjakan sendiri:** LLM membuat draf kerangka laporan dan file .docx awal. Saya yang mengambil bahan dari repo, melengkapi bagian yang kosong, mencocokkan angka dengan tabel hasil, menyisipkan gambar, mengekspor PDF, dan mengelola Git (branch, commit, push, Pull Request).
- **Commit terkait:**
  - `docs(laporan): tambah kerangka laporan akhir`
  - `docs(log): tambah template log interaksi LLM`
  - `docs(laporan): tambah laporan akhir versi docx`
  - `docs(laporan): tambah laporan akhir versi pdf`
  - `docs(log): tambah log interaksi Ce sesi laporan akhir`
