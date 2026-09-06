# 001.py - 修复后的版本
import os
import torch
from ultralytics import YOLO
import multiprocessing

# Windows多进程必须的修复
if __name__ == '__main__':
    # 方法1：禁用多进程（最简单）
    results = YOLO('yolo26n.pt').train(
        data="seaship.yaml", 
        epochs=3,
        workers=0,  # 关键：设置workers=0禁用多进程
        device=0 if torch.cuda.is_available() else 'cpu'
    )