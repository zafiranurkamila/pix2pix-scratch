"""Gabungkan hasil beberapa run jadi bahan laporan.

Jalankan SETELAH evaluate.py selesai untuk semua run:
    py -m src.compare_runs --runs baseline l1_only lambda10 lr1e-4 bs4

Run pertama di --runs dianggap baseline: kolom "Yang diubah" diisi otomatis
dengan perbedaan config tiap run terhadap run pertama.

Output di outputs/comparison/ (bisa diganti dengan --out):
    results_table.md    tabel siap tempel ke laporan (Markdown)
    results_table.csv   angka lengkap tanpa pembulatan
    comparison_grid.png gambar test yang sama dari semua run, berdampingan
    curves_all_runs.png val_L1 dan D_total semua run dalam satu grafik
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image

# Hyperparameter yang dibandingkan di kolom "Yang diubah"
TUNABLE = ["lr", "batch_size", "lambda_adv", "lambda_l1", "epochs",
           "decay_start_epoch", "seed"]

# Palet kategorikal tervalidasi (aman untuk buta warna), urutan tetap:
# run ke-1 selalu biru, ke-2 oranye, dst. Lebih dari 8 run -> pecah jadi dua grafik.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#2b2b2a"
MUTED = "#6b6a64"
GRID = "#e4e3de"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bandingkan hasil beberapa run")
    p.add_argument("--runs", nargs="+", required=True,
                   help="nama run; yang pertama dianggap baseline")
    p.add_argument("--output_root", default="outputs")
    p.add_argument("--eval_tag", default="test_last_eval",
                   help="subfolder eval/ yang dipakai (default: split test, last.pt, mode eval)")
    p.add_argument("--indices", nargs="*", type=int, default=None,
                   help="indeks gambar test untuk grid (default: 5 gambar tersebar merata)")
    p.add_argument("--out", default=None, help="folder output (default: outputs/comparison)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Membaca hasil
# ---------------------------------------------------------------------------
def load_run(output_root: Path, run: str, eval_tag: str) -> dict | None:
    run_dir = output_root / run
    metrics_path = run_dir / "eval" / eval_tag / "metrics.json"
    log_path = run_dir / "logs" / "losses.csv"
    if not metrics_path.exists() or not log_path.exists():
        print(f"[warn] {run}: metrics.json atau losses.csv tidak ada, dilewati. "
              f"Sudah jalankan evaluate.py untuk run ini?")
        return None
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    with open(log_path, newline="") as f:
        log = list(csv.DictReader(f))
    return {
        "name": run,
        "metrics": metrics,
        "log": log,
        "img_dir": metrics_path.parent / "images",
        "train_minutes": sum(float(r["epoch_time_s"]) for r in log) / 60.0,
    }


def describe_changes(cfg: dict, base_cfg: dict) -> str:
    diffs = [f"{k}={cfg[k]}" for k in TUNABLE
             if k in cfg and k in base_cfg and cfg[k] != base_cfg[k]]
    return ", ".join(diffs) if diffs else "(baseline)"


# ---------------------------------------------------------------------------
# 1. Tabel
# ---------------------------------------------------------------------------
def write_tables(runs: list[dict], out_dir: Path) -> None:
    base_cfg = runs[0]["metrics"]["config"]
    rows = []
    for r in runs:
        m, last = r["metrics"], r["log"][-1]
        rows.append({
            "run": r["name"],
            "yang_diubah": describe_changes(m["config"], base_cfg),
            "epoch": m["epoch"],
            "test_L1": m["L1"],
            "test_PSNR": m["PSNR"],
            "test_SSIM": m["SSIM"],
            "sharp_ratio": m["sharp_ratio"],
            "val_L1_akhir": float(last["val_L1"]),
            "D_total_akhir": float(last["D_total"]),
            "waktu_menit": r["train_minutes"],
        })

    with open(out_dir / "results_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    m0 = runs[0]["metrics"]
    lines = [
        f"Evaluasi di split **{m0['split']}** ({m0['n_images']} gambar), "
        f"checkpoint `{m0['checkpoint']}`, G mode `{m0['g_mode']}`.",
        "",
        "| Run | Yang diubah | Epoch | L1 ↓ | PSNR (dB) ↑ | SSIM ↑ "
        "| Ketajaman (rasio ke foto asli, ideal ≈ 1) | val_L1 akhir | Waktu (menit) | Catatan |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['run']}` | {row['yang_diubah']} | {row['epoch']} "
            f"| {row['test_L1']:.4f} | {row['test_PSNR']:.2f} | {row['test_SSIM']:.4f} "
            f"| {row['sharp_ratio']:.3f} | {row['val_L1_akhir']:.4f} "
            f"| {row['waktu_menit']:.0f} |  |")
    lines += [
        "",
        "L1, PSNR, dan SSIM mengukur kemiripan dengan foto asli; model yang hasilnya "
        "blur sering tetap unggul di sini. Kolom ketajaman melengkapinya: rasio jauh "
        "di bawah 1 berarti blur, rasio di atas 1 biasanya berarti noise/artefak.",
    ]
    with open(out_dir / "results_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 2. Grid perbandingan gambar
# ---------------------------------------------------------------------------
def pick_indices(n_images: int, k: int = 5) -> list[int]:
    if n_images <= k:
        return list(range(n_images))
    step = (n_images - 1) / (k - 1)
    return [round(i * step) for i in range(k)]


def save_comparison_grid(runs: list[dict], indices: list[int], path: Path) -> None:
    """Baris = satu gambar test. Kolom = label map | hasil tiap run | foto asli."""
    first_dir = runs[0]["img_dir"]
    columns = [("Label map", lambda i: first_dir / f"input_{i:03d}.png")]
    for r in runs:
        columns.append((r["name"], lambda i, d=r["img_dir"]: d / f"pred_{i:03d}.png"))
    columns.append(("Foto asli", lambda i: first_dir / f"target_{i:03d}.png"))

    n_rows, n_cols = len(indices), len(columns)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.1 * n_cols, 2.1 * n_rows + 0.4),
                             squeeze=False)
    for r_i, idx in enumerate(indices):
        for c_i, (title, path_fn) in enumerate(columns):
            ax = axes[r_i][c_i]
            ax.imshow(Image.open(path_fn(idx)))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r_i == 0:
                ax.set_title(title, fontsize=10, color=TEXT)
            if c_i == 0:
                ax.set_ylabel(f"test #{idx}", fontsize=9, color=MUTED)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Grafik gabungan
# ---------------------------------------------------------------------------
def save_curves(runs: list[dict], path: Path) -> None:
    """Dua panel terpisah (masing-masing satu sumbu y), satu garis per run."""
    panels = [("val_L1", "L1 di split val (makin kecil makin mirip)"),
              ("D_total", "Loss Discriminator (D_total)")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    direct_label = len(runs) <= 4   # >4 garis: label ujung garis bertabrakan, cukup legend

    for ax, (key, title) in zip(axes, panels):
        for i, r in enumerate(runs):
            color = SERIES_COLORS[i % len(SERIES_COLORS)]
            ep = [int(row["epoch"]) for row in r["log"]]
            val = [float(row[key]) for row in r["log"]]
            ax.plot(ep, val, color=color, linewidth=1.8, label=r["name"])
            if direct_label:
                ax.annotate(r["name"], (ep[-1], val[-1]), xytext=(6, 0),
                            textcoords="offset points", va="center",
                            fontsize=9, color=TEXT)
        ax.set_title(title, fontsize=11, color=TEXT, loc="left")
        ax.set_xlabel("epoch", color=MUTED)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # epoch selalu bulat
        if direct_label:
            ax.margins(x=0.12)   # ruang untuk label di ujung garis

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(runs), 8),
               frameon=False, fontsize=9, labelcolor=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    out_dir = Path(args.out) if args.out else output_root / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [r for r in (load_run(output_root, name, args.eval_tag) for name in args.runs) if r]
    if not runs:
        raise SystemExit("[error] tidak ada run yang bisa dibaca.")
    if runs[0]["name"] != args.runs[0]:
        print(f"[warn] baseline '{args.runs[0]}' tidak terbaca; "
              f"kolom 'Yang diubah' dibandingkan ke '{runs[0]['name']}'.")

    n_images = runs[0]["metrics"]["n_images"]
    indices = args.indices if args.indices else pick_indices(n_images)

    write_tables(runs, out_dir)
    save_comparison_grid(runs, indices, out_dir / "comparison_grid.png")
    save_curves(runs, out_dir / "curves_all_runs.png")
    print(f"[compare] {len(runs)} run digabung -> {out_dir}")


if __name__ == "__main__":
    main()
