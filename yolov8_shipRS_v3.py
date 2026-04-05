"""ShipRS-v3 训练脚本 — 原创改进版 (DASC + BSA + SOAU)"""
from ultralytics import YOLO

if __name__ == "__main__":
    results = YOLO("yolov8-ShipRS-v3.yaml").train(
        data="seaship.yaml",
        epochs=500,
        fraction=0.3,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        batch=64,
        imgsz=640,
        patience=30,
        seed=0,
        device=0,
        workers=8,
    )
