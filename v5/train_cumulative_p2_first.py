#!/usr/bin/env python3
"""
累积消融: P2-first — baseline → complete_v2 (逐模块累加)
=========================================================
Exp0: baseline                   标准 YOLOv8n
Exp1: +P2                        四尺度检测
Exp2: +SPDConv                   Backbone SPDConv
Exp3: +EMA                       Backbone P3 注意力
Exp4: +CDGM                      FPN 细节引导 (引入即×3)
Exp5: +ASGv2                     检测头门控 (引入即×4) = complete_v2
"""

import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTHONPATH"] = f"{PROJECT_ROOT}:{os.environ.get('PYTHONPATH', '')}"

try:
    from ultralytics.nn.modules import C2f_EMA, CDGM, ASG, SPDConv
    print("[OK] 自定义模块加载成功")
except Exception as e:
    print(f"[ERROR] 自定义模块加载失败: {e}")
    sys.exit(1)

from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore", message="grid_sampler_2d_backward_cuda")
warnings.filterwarnings("ignore", message="upsample_bilinear2d_backward_out_cuda")
warnings.filterwarnings("ignore", message="adaptive_avg_pool2d_backward_cuda")

SCRIPT_DIR = Path(__file__).resolve().parent

EXPERIMENTS = [
    ("00_baseline", "baseline.yaml",                  "标准 YOLOv8n",              200),
    ("01_p2",       "cumulative_p2_first_01_p2.yaml",  "+P2 (四尺度)",              100),
    ("02_spdconv",  "cumulative_p2_first_02_spdconv.yaml","+SPDConv",               200),
    ("03_ema",      "cumulative_p2_first_03_ema.yaml", "+EMA",                      200),
    ("04_cdgm",     "cumulative_p2_first_04_cdgm.yaml", "+CDGM×3",                  200),
    ("05_asg",      "cumulative_p2_first_05_asg.yaml",  "+ASGv2×4=complete_v2",     200),
    ("06_dyhead",   "cumulative_p2_first_06_dyhead.yaml","+DyHeadDetect (取代ASG)",  200),
    ("07_lite",     "cumulative_p2_first_07_lite.yaml", "+Lite(RepVGG+PConv)",       200),  # 待验证
]

TRAIN_ARGS = {
    "data": "data/seaship.yaml",
    "imgsz": 640, "batch": 64,
    "fraction": 0.3, "patience": 30, "device": 0,
    "workers": 8, "seed": 0, "plots": False,
    # 固定 SGD 优化器，避免 auto 根据 epochs 数自动切到 AdamW
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
}

def run(name, yaml_path, desc, epochs):
    print(f"\n{'='*60}\n实验: {name}\n描述: {desc}\nEpochs: {epochs}\nYAML: {yaml_path}\n{'='*60}\n")
    try:
        results = YOLO(str(yaml_path)).train(**TRAIN_ARGS, epochs=epochs, project="runs/ablation_cumulative_p2_first", name=name)
        m = {
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
            "params": float(getattr(results, "params", 0) or 0),
        }
        print(f"\n[RESULT] {name}: mAP50={m['mAP50']:.4f}, mAP50-95={m['mAP50-95']:.4f}")
        return m
    except Exception as e:
        print(f"\n[ERROR] {name} 失败: {e}")
        import traceback; traceback.print_exc()
        return None

def main():
    all_results = {}
    for name, yaml_name, desc, epochs in EXPERIMENTS:
        all_results[name] = run(name, SCRIPT_DIR / yaml_name, desc, epochs)

    base = all_results.get("00_baseline")
    print("\n" + "="*85)
    print("累积消融结果 (P2-first)")
    print("="*85)
    print(f"{'实验':<24} {'Epochs':>7} {'mAP50':>8} {'Δ步进':>8} {'Δ累计':>8} {'mAP50-95':>10} {'P':>7} {'R':>7} {'Params':>9}")
    print("-"*85)
    prev = base["mAP50"] if base else 0
    for name, _, desc, epochs in EXPERIMENTS:
        m = all_results.get(name)
        if m:
            ds = m["mAP50"] - prev
            dt = m["mAP50"] - base["mAP50"] if name != "00_baseline" else 0
            print(f"{name:<24} {epochs:>7} {m['mAP50']:>8.4f} {ds:>+8.4f} {dt:>+8.4f} {m['mAP50-95']:>10.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['params']:>9.0f}")
            prev = m["mAP50"]
        else:
            print(f"{name:<24} {'FAILED':>8}")
    print("="*85)

if __name__ == "__main__":
    main()
