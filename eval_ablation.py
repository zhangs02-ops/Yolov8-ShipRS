#!/usr/bin/env python3
"""
Ablation study comprehensive evaluation:
  - mAP50, mAP50-95
  - AP per size (S, M, L)
  - Precision, Recall
  - Params, GFLOPs
  - FPS (inference speed)
"""
import sys, os, time, json
from pathlib import Path

# 修复 nvrtc 库路径（PyTorch cu130 需要）
_cu13_lib = "/home/zhangs02/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib"
if os.path.isdir(_cu13_lib):
    os.environ.setdefault("LD_LIBRARY_PATH", "")
    if _cu13_lib not in os.environ["LD_LIBRARY_PATH"]:
        os.environ["LD_LIBRARY_PATH"] = f"{_cu13_lib}:{os.environ['LD_LIBRARY_PATH']}"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["PYTHONPATH"] = f"{Path(__file__).resolve().parent.parent}:{os.environ.get('PYTHONPATH', '')}"

from ultralytics import YOLO
import torch
import numpy as np

# ─── Config ──────────────────────────────────────────────────────────────
RESULTS_DIR = Path("/home/zhangs02/yolo_result/v5/ablation_cumulative_100/ablation_cumulative_p2_first")
DATA_YAML    = "/home/zhangs02/ultralytics-8.4.21/v5/v1/seaship.yaml"

EXPERIMENTS = [
    ("01_p2",       RESULTS_DIR / "01_p2"       / "weights" / "best.pt"),
    ("02_spdconv",  RESULTS_DIR / "02_spdconv"  / "weights" / "best.pt"),
    ("03_ema",      RESULTS_DIR / "03_ema"      / "weights" / "best.pt"),
    ("04_cdgm",     RESULTS_DIR / "04_cdgm"     / "weights" / "best.pt"),
    ("05_asg",      RESULTS_DIR / "05_asg"      / "weights" / "best.pt"),
    ("06_dyhead",   RESULTS_DIR / "06_dyhead"   / "weights" / "best.pt"),
]

# ─── Helpers ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def measure_fps(yolo_model, img_size=640, batch=1, warmup=50, iters=500):
    """测 FPS (纯推理，通过 YOLO wrapper 以包含预处理 + NMS)"""
    dummy = torch.randn(batch, 3, img_size, img_size)

    # warmup
    for _ in range(warmup):
        _ = yolo_model.predict(dummy, verbose=False)

    # timed inference
    start = time.perf_counter()
    for _ in range(iters):
        _ = yolo_model.predict(dummy, verbose=False)
    total = time.perf_counter() - start

    avg_ms = total / iters * 1000
    fps = 1000 / avg_ms * batch
    return fps, avg_ms


def extract_metrics(val_results):
    """从 val() 返回的结果中提取指标"""
    rd = val_results.results_dict
    return {
        "mAP50":      rd.get("metrics/mAP50(B)",      0),
        "mAP50-95":   rd.get("metrics/mAP50-95(B)",   0),
        "precision":  rd.get("metrics/precision(B)",   0),
        "recall":     rd.get("metrics/recall(B)",      0),
    }


def count_params(model):
    """计算参数量"""
    return sum(p.numel() for p in model.parameters())


@torch.inference_mode()
def estimate_gflops(model, img_size=640):
    """粗略估计 GFLOPs（单次前向传播）"""
    try:
        from thop import profile
    except ImportError:
        return 0

    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    return flops / 1e9


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"{'='*120}")
    print(f"{'Experiment':<12} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8} "
          f"{'Params(M)':>10} {'GFLOPs':>8} {'FPS':>8} {'ms/img':>8}")
    print(f"{'-'*100}")

    all_results = {}

    for name, ckpt_path in EXPERIMENTS:
        if not ckpt_path.exists():
            print(f"{name:<12}  {'WEIGHT NOT FOUND':>8}")
            continue

        print(f"\n>>> Evaluating {name} ...", flush=True)

        # 1. 加载模型
        model = YOLO(str(ckpt_path))

        # 2. 验证 (不用 half，避免 nvrtc 兼容问题)
        val_results = model.val(
            data=DATA_YAML,
            batch=32,
            imgsz=640,
            device=device,
            plots=False,
            save_json=False,
            verbose=False,
        )

        # 3. 提取指标
        metrics = extract_metrics(val_results)

        # 4. 参数量
        params_m = count_params(model.model) / 1e6

        # 5. FPS (YOLO wrapper predict 会自动处理 device)
        fps, ms = measure_fps(model, batch=1)

        # 6. GFLOPs
        gflops = estimate_gflops(model.model)

        # 7. 打印 summary 辅助信息
        try:
            summary = val_results.summary()
            if summary:
                s = summary[0]
                print(f"  [per-size] AP50_S:{s.get('ap50_small', '-'):>6} AP50_M:{s.get('ap50_medium', '-'):>6} AP50_L:{s.get('ap50_large', '-'):>6}")
        except:
            pass

        all_results[name] = {**metrics, "params": params_m, "gflops": gflops, "fps": fps, "ms": ms}

        print(f"{name:<12} {metrics['mAP50']:>8.4f} {metrics['mAP50-95']:>10.4f} "
              f"{metrics['precision']:>8.4f} {metrics['recall']:>8.4f} "
              f"{params_m:>10.3f} {gflops:>8.2f} {fps:>8.1f} {ms:>8.2f}")

    # ─── Summary Table ────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print(f"SUMMARY - Ablation Study @ 640×640 on seaship val set")
    print(f"{'='*100}")
    print(f"{'Experiment':<12} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8} "
          f"{'Params(M)':>10} {'GFLOPs':>8} {'FPS':>8} {'ms/img':>8}")
    print(f"{'-'*100}")
    for name, _ in EXPERIMENTS:
        r = all_results.get(name)
        if r is None:
            print(f"{name:<12}  {'N/A':>8}")
            continue
        print(f"{name:<12} {r['mAP50']:>8.4f} {r['mAP50-95']:>10.4f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} "
              f"{r['params']:>10.3f} {r['gflops']:>8.2f} {r['fps']:>8.1f} {r['ms']:>8.2f}")

    # ─── Delta Table ──────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"DELTA vs 01_p2 (P2 baseline)")
    print(f"{'='*100}")
    print(f"{'Experiment':<12} {'ΔmAP50':>8} {'ΔmAP50-95':>10} {'ΔP':>8} {'ΔR':>8} "
          f"{'ΔParams':>10} {'ΔGFLOPs':>8}")
    print(f"{'-'*100}")
    base = all_results.get("01_p2", {})
    for name, _ in EXPERIMENTS:
        r = all_results.get(name)
        if r is None or name == "01_p2":
            print(f"{name:<12}  {'-':>8}  {'-':>10}  {'-':>8}  {'-':>8}  {'-':>10}  {'-':>8}")
            continue
        print(f"{name:<12} {r['mAP50']-base['mAP50']:>+8.4f} {r['mAP50-95']-base['mAP50-95']:>+10.4f} "
              f"{r['precision']-base['precision']:>+8.4f} {r['recall']-base['recall']:>+8.4f} "
              f"{r['params']-base['params']:>+10.3f} {r['gflops']-base['gflops']:>+8.2f}")

    # 保存结果
    out_path = RESULTS_DIR / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
