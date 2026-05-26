"""ShipRS-v2-EMA 消融训练脚本（仅 EMA 注意力，无渐进式融合）."""

from ultralytics import YOLO

model = YOLO("yolov8-EMA.yaml")

model.train(
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
