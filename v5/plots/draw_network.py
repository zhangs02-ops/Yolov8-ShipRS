"""
绘制 cumulative_p2_first_06_dyhead.yaml 的完整网络结构图

用法:
    cd ~/ultralytics-8.4.21/v5/v5/plots
    python draw_network.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def box(ax, x, y, w, h, text, color='#E8F4FD', edge='#2196F3', fontsize=8, bold=False):
    """画圆角矩形框"""
    b = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=color, edgecolor=edge, linewidth=1.5
    )
    ax.add_patch(b)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color='#333333', weight=weight)
    return b


def arrow(ax, x1, y1, x2, y2, color='#666666', style='-'):
    """画箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2, linestyle=style))


def line(ax, x1, y1, x2, y2, color='#666666', style='--'):
    """画直线"""
    ax.plot([x1, x2], [y1, y2], color=color, lw=1.2, linestyle=style)


# ======================================================================
fig, ax = plt.subplots(1, 1, figsize=(22, 14))
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis('off')
ax.set_facecolor('white')
ax.set_title('YOLOv8-P2-ShipRS Architecture\n(SPDConv + C2f_EMA + CDGM + DyHeadDetect)',
             fontsize=18, weight='bold', pad=15)

# 颜色定义
colors = {
    'input': '#FFF3E0',
    'spdconv': '#AED6F1',
    'c2f': '#D5F5E3',
    'c2f_ema': '#82E0AA',
    'sppf': '#FADBD8',
    'upsample': '#FCF3CF',
    'cdgm': '#F5B7B1',
    'concat': '#E5E8E8',
    'conv': '#D6EAF8',
    'detect': '#F1948A',
    'text': '#2C3E50',
}

# ======================================================================
# Backbone (水平排列, y=11)
# ======================================================================
ax.text(2.5, 13.5, 'Backbone', fontsize=14, weight='bold', color=colors['text'])

backbone_layers = [
    (2.0, 11.5, 'Input\n640×640', colors['input'], '#FF9800'),
    (3.5, 11.5, 'SPDConv\n64ch', colors['spdconv'], '#2196F3'),
    (5.0, 11.5, 'SPDConv\n128ch', colors['spdconv'], '#2196F3'),
    (6.5, 11.5, 'C2f ×3\n128ch', colors['c2f'], '#27AE60'),
    (8.0, 11.5, 'SPDConv\n256ch', colors['spdconv'], '#2196F3'),
    (9.5, 11.5, 'C2f_EMA\n×6  256ch', colors['c2f_ema'], '#27AE60'),
    (11.0, 11.5, 'SPDConv\n512ch', colors['spdconv'], '#2196F3'),
    (12.5, 11.5, 'C2f ×6\n512ch', colors['c2f'], '#27AE60'),
    (14.0, 11.5, 'SPDConv\n1024ch', colors['spdconv'], '#2196F3'),
    (15.5, 11.5, 'C2f ×3\n1024ch', colors['c2f'], '#27AE60'),
    (17.0, 11.5, 'SPPF\n1024ch', colors['sppf'], '#E74C3C'),
]

# 标注skip连接点 (P4, P3, P2)
skip_indices = {4: 'P4\n(80×80)', 6: 'P3\n(40×40)', 8: 'P2\n(20×20)'}
for i, (x, y, text, fc, ec) in enumerate(backbone_layers):
    box(ax, x, y, 1.2, 0.7, text, fc, ec, fontsize=7.5)
    if i < len(backbone_layers) - 1:
        arrow(ax, x + 0.6, y, backbone_layers[i+1][0] - 0.6, y)

# 标记skip输出位置
arrow(ax, 9.5, 11.15, 9.5, 10.0, color='#3498DB', style='--')
ax.text(9.7, 10.5, 'P4', fontsize=8, color='#3498DB', weight='bold')

arrow(ax, 12.5, 11.15, 12.5, 9.0, color='#3498DB', style='--')
ax.text(12.7, 10.0, 'P3', fontsize=8, color='#3498DB', weight='bold')

arrow(ax, 15.5, 11.15, 15.5, 7.5, color='#3498DB', style='--')
ax.text(15.7, 8.5, 'P2', fontsize=8, color='#3498DB', weight='bold')

# ======================================================================
# Head FPN (向上融合)
# ======================================================================
ax.text(2.5, 10.2, 'Head FPN (Detail-Guided Fusion)', fontsize=13, weight='bold', color=colors['text'])

# FPN P4
box(ax, 17.0, 9.5, 1.4, 0.6, 'Upsample\n2×', colors['upsample'], '#F39C12', fontsize=8)
arrow(ax, 17.0, 10.15, 17.0, 9.8)

box(ax, 14.5, 9.5, 1.8, 0.7, 'CDGM\n(P4 skip)', colors['cdgm'], '#E74C3C', fontsize=8, bold=True)
arrow(ax, 16.3, 9.5, 14.5 + 0.9, 9.5)
line(ax, 9.5, 10.0, 9.5, 9.5, color='#3498DB')
line(ax, 9.5, 9.5, 13.6, 9.5, color='#3498DB')
arrow(ax, 13.6, 9.5, 13.6, 9.5, color='#3498DB')
ax.text(11.5, 9.7, 'P4 skip', fontsize=7, color='#3498DB')

box(ax, 12.0, 9.5, 1.2, 0.6, 'C2f ×3\n512ch', colors['c2f'], '#27AE60', fontsize=8)
arrow(ax, 13.3, 9.5, 12.6, 9.5)

# FPN P3
box(ax, 12.0, 8.2, 1.4, 0.6, 'Upsample\n2×', colors['upsample'], '#F39C12', fontsize=8)
arrow(ax, 12.0, 9.2, 12.0, 8.5)

box(ax, 9.5, 8.2, 1.8, 0.7, 'CDGM\n(P3 skip)', colors['cdgm'], '#E74C3C', fontsize=8, bold=True)
arrow(ax, 11.3, 8.2, 9.5 + 0.9, 8.2)
line(ax, 12.5, 9.0, 12.5, 8.2, color='#3498DB')
line(ax, 12.5, 8.2, 10.4, 8.2, color='#3498DB')
ax.text(11.2, 8.4, 'P3 skip', fontsize=7, color='#3498DB')

box(ax, 7.5, 8.2, 1.2, 0.6, 'C2f ×3\n256ch', colors['c2f'], '#27AE60', fontsize=8)
arrow(ax, 8.6, 8.2, 8.1, 8.2)

# FPN P2
box(ax, 7.5, 6.9, 1.4, 0.6, 'Upsample\n2×', colors['upsample'], '#F39C12', fontsize=8)
arrow(ax, 7.5, 7.9, 7.5, 7.2)

box(ax, 5.0, 6.9, 1.8, 0.7, 'CDGM\n(P2 skip)', colors['cdgm'], '#E74C3C', fontsize=8, bold=True)
arrow(ax, 6.8, 6.9, 5.0 + 0.9, 6.9)
line(ax, 15.5, 7.5, 15.5, 6.9, color='#3498DB')
line(ax, 15.5, 6.9, 5.9, 6.9, color='#3498DB')
ax.text(10.0, 7.1, 'P2 skip', fontsize=7, color='#3498DB')

box(ax, 3.0, 6.9, 1.2, 0.6, 'C2f ×3\n128ch', colors['c2f'], '#27AE60', fontsize=8)
arrow(ax, 4.1, 6.9, 3.6, 6.9)

# ======================================================================
# Head PAN (向下融合)
# ======================================================================
ax.text(2.5, 6.0, 'Head PAN (Top-Down Fusion)', fontsize=13, weight='bold', color=colors['text'])

# PAN P3
box(ax, 3.0, 5.0, 1.4, 0.6, 'Conv\n128ch, 3×3, s=2', colors['conv'], '#2980B9', fontsize=7.5)
arrow(ax, 3.0, 6.6, 3.0, 5.3)

box(ax, 5.5, 5.0, 1.4, 0.6, 'Concat\n(P3)', colors['concat'], '#7F8C8D', fontsize=8)
arrow(ax, 3.7, 5.0, 4.8, 5.0)
line(ax, 7.5, 8.2, 7.5, 5.0, color='#666666', style='--')
line(ax, 7.5, 5.0, 6.2, 5.0, color='#666666', style='--')
ax.text(6.8, 6.5, 'P3\nshortcut', fontsize=7, color='#666666')

box(ax, 7.5, 5.0, 1.2, 0.6, 'C2f ×3\n256ch', colors['c2f'], '#27AE60', fontsize=8)
arrow(ax, 6.2, 5.0, 6.9, 5.0)

# PAN P4
box(ax, 7.5, 3.7, 1.4, 0.6, 'Conv\n256ch, 3×3, s=2', colors['conv'], '#2980B9', fontsize=7.5)
arrow(ax, 7.5, 4.7, 7.5, 4.0)

box(ax, 10.0, 3.7, 1.4, 0.6, 'Concat\n(P4)', colors['concat'], '#7F8C8D', fontsize=8)
arrow(ax, 8.2, 3.7, 9.3, 3.7)
line(ax, 12.0, 9.5, 12.0, 3.7, color='#666666', style='--')
line(ax, 12.0, 3.7, 10.7, 3.7, color='#666666', style='--')
ax.text(11.3, 6.0, 'P4\nshortcut', fontsize=7, color='#666666')

box(ax, 12.0, 3.7, 1.2, 0.6, 'C2f ×3\n512ch', colors['c2f'], '#27AE60', fontsize=8)
arrow(ax, 10.7, 3.7, 11.4, 3.7)

# PAN P5
box(ax, 12.0, 2.4, 1.4, 0.6, 'Conv\n512ch, 3×3, s=2', colors['conv'], '#2980B9', fontsize=7.5)
arrow(ax, 12.0, 3.4, 12.0, 2.7)

box(ax, 14.5, 2.4, 1.4, 0.6, 'Concat\n(P5)', colors['concat'], '#7F8C8D', fontsize=8)
arrow(ax, 12.7, 2.4, 13.8, 2.4)
line(ax, 17.0, 11.5, 17.0, 2.4, color='#666666', style='--')
line(ax, 17.0, 2.4, 15.2, 2.4, color='#666666', style='--')
ax.text(16.0, 6.0, 'SPPF\nshortcut', fontsize=7, color='#666666')

box(ax, 16.5, 2.4, 1.2, 0.6, 'C2f ×3\n1024ch', colors['c2f'], '#27AE60', fontsize=8)
arrow(ax, 15.2, 2.4, 15.9, 2.4)

# ======================================================================
# Detect
# ======================================================================
ax.text(2.5, 1.5, 'Detection Head', fontsize=13, weight='bold', color=colors['text'])

# 收集四个尺度的特征
arrow(ax, 3.0, 6.6, 3.0, 1.2, color='#E74C3C', style='--')
arrow(ax, 7.5, 4.7, 7.5, 1.2, color='#E74C3C', style='--')
arrow(ax, 12.0, 3.4, 12.0, 1.2, color='#E74C3C', style='--')
arrow(ax, 16.5, 3.4, 16.5, 1.2, color='#E74C3C', style='--')

box(ax, 10.0, 0.8, 3.5, 0.8, 'DyHeadDetect\n4-scale: P2/P3/P4/P5', colors['detect'], '#C0392B', fontsize=10, bold=True)
line(ax, 3.0, 1.2, 3.0, 0.8, color='#E74C3C')
line(ax, 7.5, 1.2, 7.5, 0.8, color='#E74C3C')
line(ax, 12.0, 1.2, 12.0, 0.8, color='#E74C3C')
line(ax, 16.5, 1.2, 16.5, 0.8, color='#E74C3C')
line(ax, 3.0, 0.8, 8.25, 0.8, color='#E74C3C')
line(ax, 11.75, 0.8, 16.5, 0.8, color='#E74C3C')

# ======================================================================
# 图例
# ======================================================================
legend_items = [
    (colors['spdconv'], 'SPDConv (×5)'),
    (colors['c2f_ema'], 'C2f_EMA'),
    (colors['cdgm'], 'CDGM (×3)'),
    (colors['detect'], 'DyHeadDetect'),
    (colors['concat'], 'Concat'),
    (colors['conv'], 'Conv (s=2)'),
]
for i, (color, label) in enumerate(legend_items):
    x = 19.0 + (i % 3) * 1.0
    y = 13.0 - (i // 3) * 0.6
    b = FancyBboxPatch((x - 0.35, y - 0.2), 0.7, 0.4,
                        boxstyle="round,pad=0.02,rounding_size=0.08",
                        facecolor=color, edgecolor='#333', linewidth=1)
    ax.add_patch(b)
    ax.text(x + 0.5, y, label, ha='left', va='center', fontsize=8, color='#333')

# 保存
plt.tight_layout()
fig.savefig('/home/zhangs02/ultralytics-8.4.21/v5/v5/plots/network_architecture.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("[保存] network_architecture.png")
