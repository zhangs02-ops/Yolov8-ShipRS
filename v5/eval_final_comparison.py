#!/usr/bin/env python3
"""
Comprehensive evaluation: 08_lite (lightweight) vs Full version vs other models.
Measures: mAP50, mAP50-95, Precision, Recall, Params, GFLOPs, FPS, ms/img, Weight size.
"""
import sys, os, time, json, subprocess
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────
DATA_YAML = "/home/zhangs02/ultralytics-8.4.21/v5/v1/seaship.yaml"
OUTPUT_DIR = Path("/home/zhangs02/yolo_result/v5/eval_comparison")

# Models to evaluate:
# (name, weight_path, description)
EXPERIMENTS = [
    # === Primary 200-epoch comparison ===
    ("08_lite_gsconv",
     "/home/zhangs02/yolo_result/v5/final_dif_model/08_lite/weights/best.pt",
     "08_lite (GSConv+PConv) [200ep]"),
    ("06_dyhead_full",
     "/home/zhangs02/yolo_result/v5/final (2)/final/weights/best.pt",
     "Full-DyHead [200ep]"),
    ("yolov10n",
     "/home/zhangs02/yolo_result/v5/final_dif_model/yolov10n/yolov10n/weights/best.pt",
     "YOLOv10n [200ep]"),
    ("MVD-YOLOv8",
     "/home/zhangs02/yolo_result/v5/final_dif_model/MVD-YOLOv8/MVD-YOLOv8/weights/best.pt",
     "MVD-YOLOv8 [200ep]"),
    # === 100-epoch ablation references ===
    ("01_p2_100ep",
     "/home/zhangs02/yolo_result/v5/ablation_cumulative_100/ablation_cumulative_p2_first/01_p2/weights/best.pt",
     "P2-baseline [100ep]"),
    ("02_spdconv_100ep",
     "/home/zhangs02/yolo_result/v5/ablation_cumulative_100/ablation_cumulative_p2_first/02_spdconv/weights/best.pt",
     "+SPDConv [100ep]"),
    ("03_ema_100ep",
     "/home/zhangs02/yolo_result/v5/ablation_cumulative_100/ablation_cumulative_p2_first/03_ema/weights/best.pt",
     "+EMA [100ep]"),
    ("04_cdgm_100ep",
     "/home/zhangs02/yolo_result/v5/ablation_cumulative_100/ablation_cumulative_p2_first/04_cdgm/weights/best.pt",
     "+CDGM [100ep]"),
    ("05_asg_100ep",
     "/home/zhangs02/yolo_result/v5/ablation_cumulative_100/ablation_cumulative_p2_first/05_asg/weights/best.pt",
     "+ASG [100ep]"),
    ("06_dyhead_100ep",
     "/home/zhangs02/yolo_result/v5/ablation_cumulative_100/ablation_cumulative_p2_first/06_dyhead/weights/best.pt",
     "+DyHead [100ep]"),
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_single_eval(name, weight_path, desc):
    """Run evaluation for a single model in a subprocess with proper PYTHONPATH."""
    script = f'''
import sys, os, time, json
from pathlib import Path

# Fix nvrtc path
_cu13_lib = "/home/zhangs02/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib"
if os.path.isdir(_cu13_lib):
    os.environ.setdefault("LD_LIBRARY_PATH", "")
    if _cu13_lib not in os.environ["LD_LIBRARY_PATH"]:
        os.environ["LD_LIBRARY_PATH"] = f"{{_cu13_lib}}:{{os.environ['LD_LIBRARY_PATH']}}"

from ultralytics import YOLO, __version__
import torch
import numpy as np

DATA_YAML = "{DATA_YAML}"
output_file = "{OUTPUT_DIR / name}.json"

print(f"Device: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}}")
print(f"Ultralytics: {{__version__}}, PyTorch: {{torch.__version__}}")

# 1. Load model
print("Loading model...", flush=True)
model = YOLO("{weight_path}")
weight_mb = os.path.getsize("{weight_path}") / 1e6

# 2. Validate
print("Running validation...", flush=True)
val_results = model.val(
    data=DATA_YAML,
    batch=32,
    imgsz=640,
    device=0,
    plots=False,
    save_json=False,
    verbose=False,
)

rd = val_results.results_dict
metrics = {{
    "mAP50":    rd.get("metrics/mAP50(B)",     0),
    "mAP50-95": rd.get("metrics/mAP50-95(B)",  0),
    "precision": rd.get("metrics/precision(B)", 0),
    "recall":   rd.get("metrics/recall(B)",     0),
}}

# 3. Parameters
params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
print(f"  Params: {{params_m:.3f}}M", flush=True)

# 4. FPS
print("  Measuring FPS...", flush=True)
@torch.inference_mode()
def measure_fps(yolo_model, img_size=640, batch=1, warmup=50, iters=500):
    dummy = torch.randn(batch, 3, img_size, img_size).cuda()
    for _ in range(warmup):
        _ = yolo_model.predict(dummy, verbose=False)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = yolo_model.predict(dummy, verbose=False)
    torch.cuda.synchronize()
    total = time.perf_counter() - start
    avg_ms = total / iters * 1000
    fps = 1000 / avg_ms * batch
    return fps, avg_ms

try:
    fps, ms = measure_fps(model, batch=1)
    print(f"  FPS: {{fps:.1f}}, ms: {{ms:.2f}}", flush=True)
except Exception as e:
    print(f"  FPS failed: {{e}}", flush=True)
    fps, ms = 0, 0

# 5. GFLOPs
print("  Estimating GFLOPs...", flush=True)
try:
    from thop import profile
    device = next(model.model.parameters()).device
    dummy = torch.randn(1, 3, 640, 640, device=device)
    flops, _ = profile(model.model, inputs=(dummy,), verbose=False)
    gflops = flops / 1e9
except Exception as e:
    print(f"  GFLOPs failed: {{e}}", flush=True)
    gflops = 0

results = {{
    **metrics,
    "params": params_m,
    "gflops": gflops,
    "fps": fps,
    "ms": ms,
    "weight_mb": weight_mb,
}}

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {{output_file}}")
print(json.dumps(results, indent=2))
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path(__file__).resolve().parent.parent}:{env.get('PYTHONPATH', '')}"

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min per model
    )

    print(result.stdout)
    if result.stderr:
        # Filter tqdm progress bars
        stderr_lines = [l for l in result.stderr.split('\\n') if '%' not in l and 'it/s' not in l and 's/it' not in l]
        if stderr_lines:
            print("STDERR:", '\\n'.join(stderr_lines[:20]), file=sys.stderr)
        else:
            print("(progress bars only in stderr)", file=sys.stderr)

    return result.returncode


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    print(f"{'='*130}")
    print(f"COMPREHENSIVE EVALUATION - Final Comparison")
    print(f"{'='*130}")
    print(f"Models: {len(EXPERIMENTS)}")
    print()

    all_results = {}

    for name, weight_path, desc in EXPERIMENTS:
        weight_path = Path(weight_path)
        if not weight_path.exists():
            print(f"  SKIP {name}: weight not found at {weight_path}")
            continue

        # Check for cached result
        cache_file = OUTPUT_DIR / f"{name}.json"
        if cache_file.exists():
            print(f"  Loading cached result for {name}...", flush=True)
            with open(cache_file) as f:
                all_results[name] = json.load(f)
            print(f"  {desc}: OK (cached)")
            continue

        print(f"\\n{'='*80}")
        print(f">>> Evaluating: {desc} ({name})")
        print(f"{'='*80}", flush=True)

        ret = run_single_eval(name, weight_path, desc)
        if ret != 0:
            print(f"  FAILED (exit code {ret})")
            continue

        # Load result
        cache_file = OUTPUT_DIR / f"{name}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                all_results[name] = json.load(f)
            r = all_results[name]
            print(f"  {desc}")
            print(f"  mAP50={r['mAP50']:.4f}  mAP50-95={r['mAP50-95']:.4f}  "
                  f"P={r['precision']:.4f}  R={r['recall']:.4f}")
            print(f"  Params={r['params']:.3f}M  GFLOPs={r['gflops']:.2f}  "
                  f"FPS={r['fps']:.1f}  ms={r['ms']:.2f}  Weight={r['weight_mb']:.1f}MB")

        # Clear GPU cache between models
        import gc; gc.collect()
        import torch; torch.cuda.empty_cache()

    # ─── Summary Table ────────────────────────────────────────────────
    print(f"\\n\\n{'='*130}")
    print(f"COMPREHENSIVE COMPARISON @ 640×640 on seaship val set")
    print(f"{'='*130}")
    print(f"{'Model':<30} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8} "
          f"{'Params(M)':>10} {'GFLOPs':>8} {'FPS':>8} {'ms/img':>8} {'Weight':>8}")
    print(f"{'-'*130}")

    for name, weight_path, desc in EXPERIMENTS:
        r = all_results.get(name)
        if r is None:
            print(f"{desc:<30}  {'SKIP':>8}")
            continue
        print(f"{desc:<30} {r['mAP50']:>8.4f} {r['mAP50-95']:>10.4f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} "
              f"{r['params']:>10.3f} {r['gflops']:>8.2f} {r['fps']:>8.1f} {r['ms']:>8.2f} {r['weight_mb']:>7.1f}MB")

    # ─── Key Comparison: Lite vs Full (200 epoch) ─────────────────────
    lite = all_results.get("08_lite_gsconv")
    full = all_results.get("06_dyhead_full")
    if lite and full:
        print(f"\\n{'='*100}")
        print(f"KEY COMPARISON: Lite (08_lite) vs Full (06_dyhead) @ 200 epochs")
        print(f"{'='*100}")
        print(f"{'Metric':<20} {'Lite':>14} {'Full':>14} {'Delta':>12} {'Change':>10}")
        print(f"{'-'*72}")
        for k in ["mAP50", "mAP50-95", "precision", "recall",
                   "params", "gflops", "fps", "ms", "weight_mb"]:
            lv = lite[k]
            fv = full[k]
            if isinstance(lv, (int, float)) and isinstance(fv, (int, float)) and fv != 0:
                delta = lv - fv
                pct = delta / fv * 100
                print(f"{k:<20} {lv:>14.4f} {fv:>14.4f} {delta:>+12.4f} {pct:>+9.2f}%")
            else:
                print(f"{k:<20} {str(lv):>14} {str(fv):>14}")

    # ─── Save master results ──────────────────────────────────────────
    master = {}
    for name, weight_path, desc in EXPERIMENTS:
        if name in all_results:
            master[name] = {**all_results[name], "desc": desc}

    out_path = OUTPUT_DIR / "all_results.json"
    with open(out_path, "w") as f:
        json.dump(master, f, indent=2, default=str)
    print(f"\\nMaster results saved to {out_path}")


if __name__ == "__main__":
    main()
