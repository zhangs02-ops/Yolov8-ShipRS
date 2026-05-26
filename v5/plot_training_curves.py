#!/usr/bin/env python3
"""
从 results.csv 补出训练曲线图。
用法: python plot_training_curves.py
"""
import csv, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.unicode_minus": False,
    "figure.dpi": 150,
})

RESULTS_DIR = Path("/home/zhangs02/yolo_result/v5/ablation_cumulative_100/ablation_cumulative_p2_first")
OUTPUT_DIR  = RESULTS_DIR / "training_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

EXPERIMENTS = [
    ("01_p2",      "P2 (baseline)",      "#1f77b4"),
    ("02_spdconv", "+SPDConv",           "#ff7f0e"),
    ("03_ema",     "+EMA",               "#2ca02c"),
    ("04_cdgm",    "+CDGM",              "#d62728"),
    ("05_asg",     "+ASG (deprecated)",  "#9467bd"),
    ("06_dyhead",  "+DyHeadDetect",      "#8c564b"),
]

COLORS = {name: color for name, _, color in EXPERIMENTS}
STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]

def load_csv(path):
    data = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({k: float(v) for k, v in row.items() if k != "time"})
    return data

# ── 加载所有实验数据 ──────────────────────────────────────────────
all_data = {}
for name, _, _ in EXPERIMENTS:
    csv_path = RESULTS_DIR / name / "results.csv"
    if csv_path.exists():
        all_data[name] = load_csv(csv_path)
        print(f"  {name}: {len(all_data[name])} epochs")
    else:
        print(f"  {name}: CSV not found, skip")

# ── 1. 单图: 验证集 mAP50/mAP50-95 对比 ──────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for name, label, color in EXPERIMENTS:
    d = all_data.get(name)
    if not d:
        continue
    epochs = [x["epoch"] for x in d]
    m50 = [x["metrics/mAP50(B)"] for x in d]
    m95 = [x["metrics/mAP50-95(B)"] for x in d]
    ax1.plot(epochs, m50, color=color, lw=1.8, label=label)
    ax2.plot(epochs, m95, color=color, lw=1.8, label=label)

ax1.set_title("mAP50 on Validation Set")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("mAP50"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
ax2.set_title("mAP50-95 on Validation Set")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("mAP50-95"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_mAP_curves.png")
plt.close()
print("  Saved: 01_mAP_curves.png")

# ── 2. 单图: 验证集 Loss (box) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for name, label, color in EXPERIMENTS:
    d = all_data.get(name)
    if not d:
        continue
    epochs = [x["epoch"] for x in d]
    loss = [x["val/box_loss"] for x in d]
    ax.plot(epochs, loss, color=color, lw=1.8, label=label)
ax.set_title("Validation Box Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Box Loss"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_val_box_loss.png")
plt.close()

# ── 3. 单图: Precision / Recall ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for name, label, color in EXPERIMENTS:
    d = all_data.get(name)
    if not d:
        continue
    epochs = [x["epoch"] for x in d]
    ax1.plot(epochs, [x["metrics/precision(B)"] for x in d], color=color, lw=1.8, label=label)
    ax2.plot(epochs, [x["metrics/recall(B)"] for x in d], color=color, lw=1.8, label=label)
ax1.set_title("Precision"); ax1.set_xlabel("Epoch"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
ax2.set_title("Recall"); ax2.set_xlabel("Epoch"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_precision_recall.png")
plt.close()

# ── 4. 0~50 epoch 局部放大对比 ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for name, label, color in EXPERIMENTS:
    d = all_data.get(name)
    if not d:
        continue
    epochs = [x["epoch"] for x in d if x["epoch"] <= 50]
    m95 = [x["metrics/mAP50-95(B)"] for x in d if x["epoch"] <= 50]
    ax.plot(epochs, m95, color=color, lw=1.8, label=label)
ax.set_title("mAP50-95 (First 50 Epochs)")
ax.set_xlabel("Epoch"); ax.set_ylabel("mAP50-95"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_mAP50-95_first50.png")
plt.close()

# ── 5. 最终指标横向柱状图 ────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
names = []
m50_vals = []
m95_vals = []
for name, label, color in EXPERIMENTS:
    d = all_data.get(name)
    if not d:
        continue
    names.append(label)
    m50_vals.append(d[-1]["metrics/mAP50(B)"])
    m95_vals.append(d[-1]["metrics/mAP50-95(B)"])

x = range(len(names))
bars1 = ax1.bar(x, m50_vals, color=[c for _, _, c in EXPERIMENTS if all_data.get(_)], width=0.6)
ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=8, rotation=20)
ax1.set_ylabel("mAP50"); ax1.set_title("Final mAP50 Comparison"); ax1.grid(alpha=0.3, axis="y")
for bar, v in zip(bars1, m50_vals):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f"{v:.4f}", ha="center", fontsize=7)

bars2 = ax2.bar(x, m95_vals, color=[c for _, _, c in EXPERIMENTS if all_data.get(_)], width=0.6)
ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=8, rotation=20)
ax2.set_ylabel("mAP50-95"); ax2.set_title("Final mAP50-95 Comparison"); ax2.grid(alpha=0.3, axis="y")
for bar, v in zip(bars2, m95_vals):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002, f"{v:.4f}", ha="center", fontsize=7)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_final_bar_comparison.png")
plt.close()

# ── 6. 04_cdgm 单实验详细训练曲线 ───────────────────────────────
for target, title_label in [("04_cdgm", "04_cdgm (best)" ), ("06_dyhead", "06_dyhead (final)")]:
    d = all_data.get(target)
    if not d:
        continue
    epochs = [x["epoch"] for x in d]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # train losses
    ax = axes[0,0]; ax.plot(epochs, [x["train/box_loss"] for x in d], color="#1f77b4", lw=1.5)
    ax.set_title("Train Box Loss"); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    ax = axes[0,1]; ax.plot(epochs, [x["train/cls_loss"] for x in d], color="#ff7f0e", lw=1.5)
    ax.set_title("Train Cls Loss"); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    ax = axes[0,2]; ax.plot(epochs, [x["train/dfl_loss"] for x in d], color="#2ca02c", lw=1.5)
    ax.set_title("Train DFL Loss"); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)

    # val metrics
    ax = axes[1,0]; ax.plot(epochs, [x["metrics/mAP50(B)"] for x in d], color="#d62728", lw=1.5, label="mAP50")
    ax.plot(epochs, [x["metrics/mAP50-95(B)"] for x in d], color="#9467bd", lw=1.5, label="mAP50-95")
    ax.set_title("mAP"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1,1]; ax.plot(epochs, [x["metrics/precision(B)"] for x in d], color="#8c564b", lw=1.5, label="P")
    ax.plot(epochs, [x["metrics/recall(B)"] for x in d], color="#e377c2", lw=1.5, label="R")
    ax.set_title("Precision & Recall"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1,2]; ax.plot(epochs, [x["val/box_loss"] for x in d], color="#7f7f7f", lw=1.5)
    ax.set_title("Val Box Loss"); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)

    fig.suptitle(f"Training Curves - {title_label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"06_curves_{target}.png")
    plt.close()

# ── 7. 参数 vs 速度 vs 精度 散点图 ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
name_map = {n: l for n, l, _ in EXPERIMENTS}

for name, label, color in EXPERIMENTS:
    d = all_data.get(name)
    if not d:
        continue
    # 这里用 eval_results.json 里的 FPS 数据
    import json
    with open(RESULTS_DIR / "eval_results.json") as f:
        ev = json.load(f)
    params = ev[name]["params"]
    fps = ev[name]["fps"]
    m95 = d[-1]["metrics/mAP50-95(B)"]
    sz = (m95 - 0.5) * 500  # bubble size based on mAP50-95
    ax.scatter(params, fps, s=sz*2, c=color, alpha=0.7, edgecolors="k", linewidths=0.5, zorder=5)
    ax.annotate(label.split("(")[0].strip(), (params, fps), fontsize=8, ha="center", va="bottom")

ax.set_xlabel("Parameters (M)"); ax.set_ylabel("FPS")
ax.set_title("Accuracy vs Speed vs Parameters"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_accuracy_speed_tradeoff.png")
plt.close()

# ── 8. 两两对比: 04_cdgm vs 06_dyhead ────────────────────────────
d1, d2 = all_data.get("04_cdgm"), all_data.get("06_dyhead")
if d1 and d2:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = [x["epoch"] for x in d1]
    ax1.plot(epochs, [x["metrics/mAP50(B)"] for x in d1], "#d62728", lw=2, label="04_cdgm")
    ax1.plot(epochs, [x["metrics/mAP50(B)"] for x in d2], "#8c564b", lw=2, label="06_dyhead")
    ax1.set_title("mAP50: 04_cdgm vs 06_dyhead")
    ax1.set_xlabel("Epoch"); ax1.grid(alpha=0.3); ax1.legend()

    ax2.plot(epochs, [x["metrics/mAP50-95(B)"] for x in d1], "#d62728", lw=2, label="04_cdgm")
    ax2.plot(epochs, [x["metrics/mAP50-95(B)"] for x in d2], "#8c564b", lw=2, label="06_dyhead")
    ax2.set_title("mAP50-95: 04_cdgm vs 06_dyhead")
    ax2.set_xlabel("Epoch"); ax2.grid(alpha=0.3); ax2.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_cdgm_vs_dyhead.png")
    plt.close()

print(f"\nDone! All plots saved to: {OUTPUT_DIR}")
print("Files:")
for p in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  {p.name}  ({p.stat().st_size/1024:.0f} KB)")
