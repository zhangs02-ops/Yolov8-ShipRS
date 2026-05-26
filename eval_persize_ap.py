#!/usr/bin/env python3
"""
Per-size AP evaluation (S/M/L) using COCO API.
Steps:
  1. Convert YOLO val labels → COCO JSON (ground truth)
  2. Run model.predict() on val set → convert to COCO predictions JSON
  3. pycocotools → per-size AP (small/medium/large)
"""
import sys, os, json, math, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# fix nvrtc
_cu13_lib = "/home/zhangs02/miniconda3/lib/python3.13/site-packages/nvidia/cu13/lib"
if os.path.isdir(_cu13_lib):
    os.environ.setdefault("LD_LIBRARY_PATH", "")
    if _cu13_lib not in os.environ["LD_LIBRARY_PATH"]:
        os.environ["LD_LIBRARY_PATH"] = f"{_cu13_lib}:{os.environ['LD_LIBRARY_PATH']}"

from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from PIL import Image
import torch

VAL_IMG_DIR  = "/home/zhangs02/Airbus Ship Detection Yolo format/val/images"
VAL_LAB_DIR  = "/home/zhangs02/Airbus Ship Detection Yolo format/val/labels"
RESULTS_DIR  = Path("/home/zhangs02/yolo_result/v5/ablation_cumulative_100/ablation_cumulative_p2_first")
DATA_YAML    = "/home/zhangs02/ultralytics-8.4.21/v5/v1/seaship.yaml"

EXPERIMENTS = [
    ("01_p2",       RESULTS_DIR / "01_p2"       / "weights" / "best.pt"),
    ("02_spdconv",  RESULTS_DIR / "02_spdconv"  / "weights" / "best.pt"),
    ("03_ema",      RESULTS_DIR / "03_ema"      / "weights" / "best.pt"),
    ("04_cdgm",     RESULTS_DIR / "04_cdgm"     / "weights" / "best.pt"),
    ("05_asg",      RESULTS_DIR / "05_asg"      / "weights" / "best.pt"),
    ("06_dyhead",   RESULTS_DIR / "06_dyhead"   / "weights" / "best.pt"),
]

OUTPUT_DIR = RESULTS_DIR / "coco_eval"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── Step 1: Build COCO Ground Truth JSON ───────────────────────────────
def build_coco_gt():
    gt_path = OUTPUT_DIR / "gt.json"
    if gt_path.exists():
        with open(gt_path) as f:
            return json.load(f)

    print("Building COCO ground truth JSON...")
    images = []
    annotations = []
    ann_id = 0
    img_names = sorted(os.listdir(VAL_IMG_DIR))

    for img_id, img_name in enumerate(img_names):
        img_path = os.path.join(VAL_IMG_DIR, img_name)
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except:
            w, h = 768, 768

        images.append({
            "id": img_id,
            "file_name": img_name,
            "width": w,
            "height": h,
        })

        label_path = os.path.join(VAL_LAB_DIR, os.path.splitext(img_name)[0] + ".txt")
        if not os.path.exists(label_path):
            continue

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, bw, bh = map(float, parts[:5])
                x = (cx - bw / 2) * w
                y = (cy - bh / 2) * h
                bw_abs = bw * w
                bh_abs = bh * h
                area = bw_abs * bh_abs

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x, y, bw_abs, bh_abs],
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [],
                })
                ann_id += 1

        if (img_id + 1) % 2000 == 0:
            print(f"  processed {img_id+1}/{len(img_names)} images")

    gt = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "ship", "supercategory": "object"}],
    }

    with open(gt_path, "w") as f:
        json.dump(gt, f)
    print(f"GT saved: {len(images)} images, {len(annotations)} objects")
    return gt


# ─── Step 2: Run inference & build COCO predictions JSON ────────────────
def build_pred_json(name, model):
    pred_path = OUTPUT_DIR / f"{name}_predictions.json"
    if pred_path.exists():
        with open(pred_path) as f:
            return json.load(f)

    print(f"\n>>> Running inference for {name} ...")
    img_names = sorted(os.listdir(VAL_IMG_DIR))
    BATCH = 32
    all_preds = []

    for i in range(0, len(img_names), BATCH):
        batch_names = img_names[i:i+BATCH]
        batch_paths = [os.path.join(VAL_IMG_DIR, n) for n in batch_names]
        preds = model(batch_paths, verbose=False, imgsz=640, max_det=300)
        torch.cuda.empty_cache()

        for img_offset, pred in enumerate(preds):
            img_id = i + img_offset
            if pred.boxes is None:
                continue
            boxes = pred.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, cls_id = box
                w = float(x2 - x1)
                h = float(y2 - y1)
                all_preds.append({
                    "image_id": img_id,
                    "category_id": int(cls_id) + 1,
                    "bbox": [float(x1), float(y1), w, h],
                    "score": float(conf),
                    "area": w * h,
                    "segmentation": [],
                })

        if (i // BATCH + 1) % 10 == 0:
            print(f"  {min(i+BATCH, len(img_names))}/{len(img_names)}, {len(all_preds)} dets")
            # incremental save
            with open(pred_path, "w") as f:
                json.dump(all_preds, f)

    print(f"  predictions saved: {len(all_preds)} detections")
    return all_preds


# ─── Step 3: COCO Eval ──────────────────────────────────────────────────
def eval_coco(gt_json_path, pred_path):
    """Run COCO eval and return per-size metrics."""
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(str(pred_path))

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    if stats is None or len(stats) < 12:
        print(f"  WARNING: only got {len(stats) if stats else 0} stats")
        stats = [0.0] * 12

    results = {
        "AP50-95_all":    float(stats[0]),
        "AP50_all":       float(stats[1]),
        "AP50-95_small":  float(stats[3]),
        "AP50-95_medium": float(stats[4]),
        "AP50-95_large":  float(stats[5]),
    }

    return results


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    # Step 1: GT
    gt = build_coco_gt()
    gt_json = OUTPUT_DIR / "gt.json"
    print(f"GT: {len(gt['images'])} images, {len(gt['annotations'])} objects")

    all_results = {}

    for name, ckpt_path in EXPERIMENTS:
        if not ckpt_path.exists():
            print(f"SKIP {name}: weight not found")
            continue

        # Step 2: Predict
        try:
            model = YOLO(str(ckpt_path))
            preds = build_pred_json(name, model)
        except Exception as e:
            print(f"FAIL {name}: {e}")
            import traceback; traceback.print_exc()
            continue

        # Step 3: COCO eval
        pred_path = OUTPUT_DIR / f"{name}_predictions.json"
        persize = eval_coco(str(gt_json), pred_path)
        all_results[name] = persize

        print(f"  {name}: "
              f"AP50={persize['AP50_all']:.4f} "
              f"AP50-95={persize['AP50-95_all']:.4f} "
              f"S={persize['AP50-95_small']:.4f} "
              f"M={persize['AP50-95_medium']:.4f} "
              f"L={persize['AP50-95_large']:.4f}")

        # cleanup model from GPU
        del model
        torch.cuda.empty_cache()

    # ─── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"Per-Size AP (COCO) @ 640×640 on seaship val")
    print(f"{'='*110}")
    print(f"{'Experiment':<12} {'AP50':>8} {'AP50-95':>10} "
          f"{'AP50-95_S':>10} {'AP50-95_M':>10} {'AP50-95_L':>10}")
    print(f"{'-'*110}")

    for name, _ in EXPERIMENTS:
        r = all_results.get(name)
        if r is None:
            print(f"{name:<12}  {'N/A':>8}")
            continue
        print(f"{name:<12} {r['AP50_all']:>8.4f} {r['AP50-95_all']:>10.4f} "
              f"{r['AP50-95_small']:>10.4f} {r['AP50-95_medium']:>10.4f} {r['AP50-95_large']:>10.4f}")

    # delta table
    print(f"\n{'='*110}")
    print(f"Delta vs 01_p2")
    print(f"{'='*110}")
    base = all_results.get("01_p2", {})
    if base:
        print(f"{'Experiment':<12} {'ΔAP50':>8} {'ΔAP50-95':>10} "
              f"{'ΔAP50-95_S':>10} {'ΔAP50-95_M':>10} {'ΔAP50-95_L':>10}")
        print(f"{'-'*110}")
        for name, _ in EXPERIMENTS:
            r = all_results.get(name)
            if r is None or name == "01_p2":
                print(f"{name:<12}  {'-':>8}  {'-':>10}  {'-':>10}  {'-':>10}  {'-':>10}")
                continue
            print(f"{name:<12} {r['AP50_all']-base['AP50_all']:>+8.4f} {r['AP50-95_all']-base['AP50-95_all']:>+10.4f} "
                  f"{r['AP50-95_small']-base['AP50-95_small']:>+10.4f} {r['AP50-95_medium']-base['AP50-95_medium']:>+10.4f} "
                  f"{r['AP50-95_large']-base['AP50-95_large']:>+10.4f}")

    # save
    out_path = RESULTS_DIR / "eval_persize.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
