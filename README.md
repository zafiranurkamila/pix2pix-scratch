# pix2pix-scratch

Pix2Pix (Conditional GAN) yang dibangun dari nol dengan PyTorch, untuk mengubah **label map → foto fasad bangunan** pada dataset CMP Facades. Proyek kelompok mata kuliah Generative AI, Proyek 1 Vibe Coding (CAK4KBB3 2026-1).

## Hasil singkat

Lima konfigurasi dilatih masing-masing 200 epoch (seed 42) dan dievaluasi pada 106 gambar test.

| Run | Yang diubah | L1 ↓ | PSNR ↑ | SSIM ↑ | Ketajaman | Waktu |
|---|---|---|---|---|---|---|
| baseline | – | 0,3699 | 12,59 | 0,2251 | 0,513 | 39 mnt |
| l1_only | λ_adv = 0 | 0,3576 | 12,95 | 0,3178 | 0,039 | 38 mnt |
| lambda10 | λ_L1 = 10 | 0,3768 | 12,43 | 0,2230 | 0,501 | 39 mnt |
| lr1e-4 | lr = 1e-4 | 0,3736 | 12,49 | 0,2281 | 0,477 | 38 mnt |
| **bs4** | batch size = 4 | 0,3619 | 12,55 | 0,2470 | **0,569** | **16 mnt** |

- **`bs4` adalah konfigurasi terbaik** di antara model GAN dan 2,4× lebih cepat dilatih.
- **`l1_only` unggul di L1/PSNR/SSIM tetapi hasilnya sangat buram** (ketajaman 0,039). Ini menunjukkan peran loss adversarial, dan bahwa metrik kemiripan saja tidak cukup untuk menilai model generatif.
- Tiap konfigurasi hanya dijalankan dengan satu seed, jadi kesimpulan tuning bersifat indikatif.

Gambar hasil ada di `docs/hasil/` (perbandingan semua run, grafik loss, sampel per epoch).

## Model

- **Generator:** U-Net (encoder 8 blok, decoder 7 blok, skip connection, Tanh), 54.414.531 parameter.
- **Discriminator:** PatchGAN 70×70 (input 6 channel, output logits `[B, 1, 30, 30]`, tanpa sigmoid), 2.768.705 parameter.
- **Loss:** `BCEWithLogitsLoss` (adversarial) + λ_L1 × L1, dengan λ_L1 = 100.
- **Data:** gambar 256 × 256, nilai tensor [-1, 1]. Augmentasi (resize 286, random crop, flip) hanya pada data latih, dan identik untuk input dan target.

## Struktur folder

```
pix2pix-scratch/
├── README.md
├── KONTRAK.md              kesepakatan teknis tim
├── run_experiments.bat     jalankan 5 run: train → evaluate → gabungkan
├── src/
│   ├── dataset.py          Fira
│   ├── models/             generator dan discriminator (Zafira)
│   ├── losses.py, train.py, config.py, utils.py
│   ├── metrics.py, evaluate.py, compare_runs.py, pick_examples.py
│   └── dummy_dataset.py    data sintetis untuk tes cepat
├── data/                   dataset (tidak di-commit)
├── outputs/                hasil training (tidak di-commit)
└── docs/
    ├── laporan/            laporan akhir (.docx dan .pdf)
    ├── log_interaksi/      log interaksi LLM tiap anggota
    └── hasil/              tabel, grafik, dan gambar hasil
```

## Cara menjalankan

## Cara menjalankan

Kebutuhan: Python 3.10 ke atas, PyTorch, dan torchvision. Untuk GPU NVIDIA seri RTX 50, pakai PyTorch ≥ 2.7 dengan CUDA 12.8 (lihat `KONTRAK.md` bagian 8). Perintah instal PyTorch bisa diambil dari [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

```
git clone https://github.com/zafiranurkamila/pix2pix-scratch.git
cd pix2pix-scratch
pip install numpy matplotlib pillow
```

1. **Siapkan dataset.** Unduh [facades.tar.gz](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz), lalu ekstrak ke `data/facades/{train,val,test}/`.
2. **Tes cepat tanpa dataset asli:**
   ```
   py -m src.train --dummy --run_name smoke --epochs 2 --max_steps_per_epoch 5
   ```
3. **Training satu run** (hyperparameter bisa diubah lewat terminal, misalnya `--lr` dan `--batch_size`):
   ```
   py -m src.train --run_name baseline
   ```
   Training bisa dilanjutkan otomatis dari `last.pt` jika terhenti.
4. **Evaluasi dan perbandingan:**
   ```
   py -m src.evaluate --run_name baseline
   py -m src.compare_runs --runs baseline l1_only lambda10 lr1e-4 bs4
   ```
   Jalankan `evaluate` untuk setiap run dulu sebelum `compare_runs`. Hasilnya tersimpan di `outputs/comparison/`.
5. **Semua eksperimen sekaligus (Windows):**
   ```
   .\run_experiments.bat
   ```
## Tim

| Anggota | Peran | Bagian |
|---|---|---|
| Zafira | Prompt Engineer & Training Engineer | Model, loss, training loop, evaluasi, tuning |
| Fira | Data Pipeline & Preprocessing Engineer | `src/dataset.py`, preprocessing, augmentasi, visualisasi |
| Ce | Lead Documentarian & Error Handler | Laporan, log interaksi LLM, pengelolaan repo dan PR |

Setiap anggota meng-commit bagiannya sendiri dari akun GitHub masing-masing. Alur kerja: satu branch per bagian, digabung ke `main` lewat Pull Request. Detail kesepakatan ada di [`KONTRAK.md`](KONTRAK.md).

## Dokumentasi

- Laporan akhir: `docs/laporan/laporan_akhir.pdf`
- Log interaksi LLM: `docs/log_interaksi/` (daftar sesi di `DAFTAR_SESI.md`)

## Referensi

- Isola dkk. (2017). *Image-to-Image Translation with Conditional Adversarial Networks.* CVPR.
- Ronneberger dkk. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
- Tyleček & Šára (2013). *Spatial Pattern Templates for Recognition of Objects with Regular Structure.* (dataset CMP Facade)
