Evaluasi di split **test** (106 gambar), checkpoint `last.pt`, G mode `eval`.

| Run | Yang diubah | Epoch | L1 ↓ | PSNR (dB) ↑ | SSIM ↑ | Ketajaman (rasio ke foto asli, ideal ≈ 1) | val_L1 akhir | Waktu (menit) | Catatan |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | (baseline) | 200 | 0.3699 | 12.59 | 0.2251 | 0.513 | 0.3628 | 39 |  |
| `l1_only` | lambda_adv=0.0 | 200 | 0.3576 | 12.95 | 0.3178 | 0.039 | 0.3471 | 38 |  |
| `lambda10` | lambda_l1=10.0 | 200 | 0.3768 | 12.43 | 0.2230 | 0.501 | 0.3751 | 39 |  |
| `lr1e-4` | lr=0.0001 | 200 | 0.3736 | 12.49 | 0.2281 | 0.477 | 0.3689 | 38 |  |
| `bs4` | batch_size=4 | 200 | 0.3619 | 12.55 | 0.2470 | 0.569 | 0.3563 | 16 |  |

L1, PSNR, dan SSIM mengukur kemiripan dengan foto asli; model yang hasilnya blur sering tetap unggul di sini. Kolom ketajaman melengkapinya: rasio jauh di bawah 1 berarti blur, rasio di atas 1 biasanya berarti noise/artefak.
