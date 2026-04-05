"""ShipRS-v2 训练脚本（完整模型：渐进式融合 + EMA + Shape-IoU）"""
from ultralytics import YOLO

model = YOLO("yolov8-ShipRS-v2.yaml")

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
    shape_iou=True,  # 使用 Shape-IoU 损失函数
)
