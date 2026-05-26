"""ShipRS-v3 + NWD 训练脚本."""

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
        nwd=True,
        nwd_c=2.0,
    )
