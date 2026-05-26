from ultralytics import YOLO

# Windows多进程必须的修复
if __name__ == "__main__":
    # 方法1：禁用多进程（最简单）
    results = YOLO("yolov8-CA.yaml").train(
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
