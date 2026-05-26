#!/usr/bin/env python3
"""
simple_yolo_trainer.py
极简智能YOLO训练助手 - 自动早停和性能监控.
"""

import os

import torch

from ultralytics import YOLO


class SimpleYOLOTrainer:
    def __init__(self, model_path="yolo26n.pt", data_config="seaship.yaml"):
        """极简训练器初始化.

        Args:
            model_path: 模型路径
            data_config: 数据配置
        """
        self.model_path = model_path
        self.data_config = data_config

        # 超简单配置
        self.config = {
            "data": data_config,
            "epochs": 300,  # 最大epochs
            "batch": 128,  # 根据你的GPU调整
            "patience": 0,  # 禁用内置早停
            "device": 0 if torch.cuda.is_available() else "cpu",
            "save": True,
            "project": "runs/detect",
            "name": "simple_train",
            "exist_ok": True,
        }

        # 智能早停参数
        self.early_stop = {
            "patience": 20,  # 20个epoch无改进就停止
            "min_epochs": 30,  # 至少训练30个epoch
            "min_delta": 0.001,  # 最小改进阈值
            "best_map": 0.0,  # 最佳mAP
            "no_improve": 0,  # 无改进计数
        }

        print("✅ 初始化完成")
        print(f"📊 模型: {model_path}")
        print(f"📁 数据: {data_config}")
        print(f"🎯 最大epochs: {self.config['epochs']}")
        print(f"🛑 早停耐心值: {self.early_stop['patience']}")

    def train(self):
        """开始训练."""
        print("\n🚀 开始训练...")

        # 加载模型
        model = YOLO(self.model_path)

        # 训练循环
        for epoch in range(1, self.config["epochs"] + 1):
            print(f"\n📈 Epoch {epoch}/{self.config['epochs']}")

            # 训练一个epoch
            model.train(**self.config, epochs=epoch, resume=True if epoch > 1 else False)

            # 获取当前mAP
            current_map = self._get_current_map()

            # 检查是否早停
            should_stop, reason = self._check_early_stop(current_map, epoch)

            if should_stop:
                print(f"\n🛑 训练停止: {reason}")
                break

            # 每10个epoch显示进度
            if epoch % 10 == 0:
                print(f"📊 当前mAP: {current_map:.4f}, 最佳mAP: {self.early_stop['best_map']:.4f}")

        print("\n✅ 训练完成!")
        return model

    def _get_current_map(self):
        """获取当前mAP值."""
        try:
            # 从最新结果文件读取
            results_dir = f"{self.config['project']}/{self.config['name']}"
            results_file = f"{results_dir}/results.csv"

            if os.path.exists(results_file):
                import pandas as pd

                df = pd.read_csv(results_file)
                if not df.empty:
                    # 获取最后一行的mAP50-95
                    last_row = df.iloc[-1]
                    if "metrics/mAP_0.5:0.95" in df.columns:
                        return float(last_row["metrics/mAP_0.5:0.95"])
                    elif "metrics/mAP50(B)" in df.columns:
                        return float(last_row["metrics/mAP50(B)"])

            # 如果读取失败，返回默认值
            return 0.0

        except Exception as e:
            print(f"⚠️ 读取mAP失败: {e}")
            return 0.0

    def _check_early_stop(self, current_map, epoch):
        """检查是否应该早停."""
        # 1. 检查是否达到最小epochs
        if epoch < self.early_stop["min_epochs"]:
            return False, f"继续训练（最小epochs: {self.early_stop['min_epochs']}）"

        # 2. 检查是否有改进
        improvement = current_map - self.early_stop["best_map"]

        if improvement > self.early_stop["min_delta"]:
            # 有改进，更新最佳值
            self.early_stop["best_map"] = current_map
            self.early_stop["no_improve"] = 0
            return False, f"有改进: +{improvement:.4f}"
        else:
            # 无改进，计数增加
            self.early_stop["no_improve"] += 1

            if self.early_stop["no_improve"] >= self.early_stop["patience"]:
                return True, f"{self.early_stop['patience']}个epoch无改进"
            else:
                return False, f"无改进: {self.early_stop['no_improve']}/{self.early_stop['patience']}"

        return False, "继续训练"


# ==================== 使用方法 ====================


# 方法1: 最简单的方式
def train_simple():
    """最简单的一键训练."""
    trainer = SimpleYOLOTrainer(model_path="yolo26n.pt", data_config="seaship.yaml")
    model = trainer.train()
    return model


# 方法2: 自定义参数
def train_custom():
    """自定义参数训练."""
    trainer = SimpleYOLOTrainer(model_path="yolo26n.pt", data_config="seaship.yaml")

    # 修改配置
    trainer.config["batch"] = 256  # 尝试更大的batch
    trainer.config["name"] = "my_training"

    # 修改早停参数
    trainer.early_stop["patience"] = 15  # 更短的耐心值
    trainer.early_stop["min_epochs"] = 20  # 更早开始检查

    # 开始训练
    model = trainer.train()
    return model


# 方法3: 直接调用（最简）
if __name__ == "__main__":
    # 直接运行这个文件即可开始训练
    print("=" * 50)
    print("极简YOLO训练助手")
    print("=" * 50)

    # 自动检测配置
    model_path = "yolo26n.pt"
    data_config = "seaship.yaml"

    # 检查文件是否存在
    import os

    if not os.path.exists(model_path):
        print(f"⚠️ 模型文件不存在: {model_path}")
        print("请确保yolo26n.pt在当前目录")
        exit(1)

    if not os.path.exists(data_config):
        print(f"⚠️ 数据配置文件不存在: {data_config}")
        print("请确保seaship.yaml在当前目录")
        exit(1)

    # 开始训练
    trainer = SimpleYOLOTrainer(model_path, data_config)
    model = trainer.train()

    print("\n🎉 训练完成!")
    print("最佳模型保存在: runs/detect/simple_train/weights/best.pt")
