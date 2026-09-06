# YOLOv8-ShipRS：面向遥感船舶小目标检测的 YOLOv8 改进

毕业设计项目。基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)（8.4.21），
针对**遥感船舶检测中的小目标、细长形态、海面背景干扰**等问题进行模型改进与累积消融实验验证。

## 改进内容

**累积消融主线（P2-first，配置见 `cumulative_p2_first_*.yaml`）**

| 实验 | 改进项 | 说明 |
|------|--------|------|
| Exp0 | baseline | 标准 YOLOv8n |
| Exp1 | +P2 | 增加高分辨率检测层，四尺度检测 |
| Exp2 | +SPDConv | Backbone 采用 SPD-Conv，利于小目标特征传递 |
| Exp3 | +EMA | Backbone P3 引入 EMA 注意力 |
| Exp4 | +CDGM | FPN 细节引导模块 |
| Exp5 | +ASG | 检测头门控（complete_v2） |
| Exp6-8 | +DyHead / +Lite / +GSCConv | 轻量化与检测头进一步增强 |

**原创模块（针对船舶特性自研，详见《修改说明文档v3.md》）**

- **DASC** 方向感知条形卷积：针对船舶 3:1~10:1 的细长形态
- **BSA** 背景抑制注意力：抑制海面波浪/云影干扰
- **SOAU** 小目标感知上采样：缓解 FPN 传递中小目标特征稀释
- **SADL** 尺度感知分布损失：针对小目标的回归损失设计

各版本模型结构定义见 `ultralytics/cfg/models/v8/yolov8-ShipRS*.yaml`。

## 环境安装

```bash
# Python >= 3.9，需要 PyTorch（按本机 CUDA 版本安装）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

## 数据集

使用 [Airbus Ship Detection](https://www.kaggle.com/competitions/airbus-ship-detection) 转换的
YOLO 格式数据集，路径在 `seaship.yaml` 中配置（单类别 `ship`），需按实际存放位置修改 `path` 字段。

## 使用方法

```bash
# 累积消融训练（baseline → complete_v2 逐模块累加）
python train_cumulative_p2_first.py

# 评估：消融汇总 / 最终对比 / 按尺寸 AP（参数量、GFLOPs、FPS）
python eval_ablation.py
python eval_final_comparison.py
python eval_persize_ap.py

# 训练曲线可视化
python plot_training_curves.py

# 生成性能对比 Word 报告
python generate_report.py
```

简单训练入口见 `auto_train.py` / `001.py`（含 Windows 多进程注意事项）。

## 项目结构

```
├── train_cumulative_p2_first.py   # 累积消融训练入口
├── cumulative_p2_first_*.yaml     # 消融实验训练配置（01-08）
├── eval_*.py                      # 评估脚本（消融/对比/按尺寸 AP）
├── plot_training_curves.py        # 训练曲线绘制
├── generate_report.py             # 生成 Word 性能报告
├── seaship.yaml                   # 数据集配置
├── auto_train.py / 001.py         # 通用训练脚本
├── ultralytics/                   # 修改后的 ultralytics 源码（含自研模块）
│   └── cfg/models/v8/             # ShipRS 系列模型结构定义
├── v5/                            # v5 阶段消融实验框架（配置+脚本副本）
├── plots/                         # 结构示意图与结果图
├── 修改说明文档*.md                # 各阶段改进设计说明
└── docs/                          # ultralytics 原始文档（参考）
```

## 说明文档

改进设计的完整演进过程见根目录三份《修改说明文档》（v1 → v2 → v3），
其中 v3 详细论述了原创模块的动机、结构与实验对比。

## 许可

本项目基于 AGPL-3.0 协议的 ultralytics 修改而来，遵循 [AGPL-3.0](LICENSE) 开源协议。
