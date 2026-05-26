"""ShipRS-v2 训练脚本（完整模型 + NWD + Shape-IoU 组合损失）."""

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
    nwd=True,  # 使用 NWD 损失（检测阶段）
    nwd_c=2.0,
    shape_iou=True,  # Shape-IoU 与 NWD 互斥，这里会优先使用 NWD
    # 如需单独测试 Shape-IoU，请将 nwd 设为 False
)
