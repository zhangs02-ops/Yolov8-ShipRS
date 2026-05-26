"""
DyHead 网络结构可视化
生成两张图:
  1. dyhead_overview.png  — DyHeadDetect 整体架构（vs 标准 Detect）
  2. dyblock_detail.png   — DyBlock 内部详细结构

用法:
    cd ~/ultralytics-8.4.21/v5/v5/plots
    python draw_dyhead.py
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


def draw_box(ax, x, y, w, h, text, color='#E8F4FD', edge='#2196F3', fontsize=9, bold=False):
    """画一个带文字的圆角矩形框"""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=color, edgecolor=edge, linewidth=1.5
    )
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color='#333333', weight=weight, wrap=True)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color='#666666'):
    """画箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))


def draw_dashed_arrow(ax, x1, y1, x2, y2, color='#FF9800'):
    """画虚线箭头（残差连接）"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                linestyle='--'))


# ======================================================================
# 图1: DyHeadDetect 整体架构
# ======================================================================
fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
ax1.set_xlim(0, 12)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('DyHeadDetect vs Standard Detect Architecture', fontsize=16, weight='bold', pad=20)

# --- 左侧: Standard Detect ---
ax1.text(2.5, 9.5, 'Standard Detect', ha='center', va='center',
         fontsize=14, weight='bold', color='#666666')

# Neck 输出
neck_feats = ['P2\n(160×160)', 'P3\n(80×80)', 'P4\n(40×40)', 'P5\n(20×20)']
for i, feat in enumerate(neck_feats):
    x = 1.2 + i * 0.8
    draw_box(ax1, x, 8.2, 0.7, 0.6, feat, color='#FFF3E0', edge='#FF9800', fontsize=8)
    draw_arrow(ax1, x, 7.9, x, 7.3)

# 直接检测
ax1.text(2.5, 7.5, 'Direct Detection', ha='center', va='center',
         fontsize=10, color='#999999', style='italic')
for i in range(4):
    x = 1.2 + i * 0.8
    draw_box(ax1, x, 6.8, 0.7, 0.5, 'cv2+cv3', color='#FFEBEE', edge='#E57373', fontsize=7)
    draw_arrow(ax1, x, 6.55, x, 5.9)

# 输出
for i in range(4):
    x = 1.2 + i * 0.8
    draw_box(ax1, x, 5.6, 0.7, 0.4, 'Box+Cls', color='#E8F5E9', edge='#81C784', fontsize=7)

# --- 右侧: DyHeadDetect ---
ax1.text(8.5, 9.5, 'DyHeadDetect (Ours)', ha='center', va='center',
         fontsize=14, weight='bold', color='#1565C0')

# Neck 输出
for i, feat in enumerate(neck_feats):
    x = 7.2 + i * 0.8
    draw_box(ax1, x, 8.2, 0.7, 0.6, feat, color='#FFF3E0', edge='#FF9800', fontsize=8)
    draw_arrow(ax1, x, 7.9, x, 7.3)

# DyBlock
ax1.text(8.5, 7.5, 'DyBlock × 4', ha='center', va='center',
         fontsize=10, color='#1565C0', weight='bold')
for i in range(4):
    x = 7.2 + i * 0.8
    draw_box(ax1, x, 6.8, 0.7, 0.5, 'DyBlock',
             color='#E3F2FD', edge='#2196F3', fontsize=7, bold=True)
    draw_arrow(ax1, x, 6.55, x, 5.9)

# 检测
for i in range(4):
    x = 7.2 + i * 0.8
    draw_box(ax1, x, 5.6, 0.7, 0.4, 'cv2+cv3', color='#FFEBEE', edge='#E57373', fontsize=7)
    draw_arrow(ax1, x, 5.4, x, 4.8)

# 输出
for i in range(4):
    x = 7.2 + i * 0.8
    draw_box(ax1, x, 4.5, 0.7, 0.4, 'Box+Cls', color='#E8F5E9', edge='#81C784', fontsize=7)

# --- 中间对比箭头 ---
ax1.annotate('', xy=(5.5, 7.0), xytext=(4.5, 7.0),
             arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2.5))
ax1.text(5.0, 7.25, '+ DyBlock', ha='center', va='bottom',
         fontsize=11, weight='bold', color='#4CAF50')

# --- 底部说明 ---
info_box = FancyBboxPatch((0.5, 0.3), 11, 2.0,
                           boxstyle="round,pad=0.02,rounding_size=0.2",
                           facecolor='#FAFAFA', edgecolor='#CCCCCC', linewidth=1)
ax1.add_patch(info_box)

ax1.text(6.0, 2.0, 'DyHeadDetect 改进点', ha='center', va='center',
         fontsize=12, weight='bold', color='#333333')

improvements = [
    '① 多尺度空间感知: 3个并行空洞深度卷积 (dilation=1,2,3)，感受野 3×3 / 5×5 / 7×7',
    '② 通道感知: SE (Squeeze-Excite) 注意力，自适应增强关键通道',
    '③ 可学习残差缩放: scale=0.1 初始值，渐进式特征增强',
    '④ 轻量设计: 中间通道折半 (c//2)，每个 DyBlock 仅 ~85K 参数',
]
for i, text in enumerate(improvements):
    ax1.text(6.0, 1.55 - i*0.35, text, ha='center', va='center',
             fontsize=9.5, color='#555555')

plt.tight_layout()
fig1.savefig('/home/zhangs02/ultralytics-8.4.21/v5/v5/plots/dyhead_overview.png',
             dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("[保存] dyhead_overview.png")


# ======================================================================
# 图2: DyBlock 内部详细结构
# ======================================================================
fig2, ax2 = plt.subplots(1, 1, figsize=(11, 10))
ax2.set_xlim(0, 11)
ax2.set_ylim(0, 12)
ax2.axis('off')
ax2.set_title('DyBlock Internal Structure (Dynamic Attention Block)', fontsize=16, weight='bold', pad=20)

# 输入
draw_box(ax2, 5.5, 11.0, 2.0, 0.6, 'Input Feature\n(B, C, H, W)', color='#FFF3E0', edge='#FF9800', fontsize=10, bold=True)
draw_arrow(ax2, 5.5, 10.7, 5.5, 10.1)

# 分叉: 主路径 + 残差
ax2.text(5.5, 10.3, 'Split', ha='center', va='center', fontsize=9, color='#666666')

# 残差路径 (右侧虚线)
ax2.plot([7.5, 7.5, 5.5], [11.0, 1.8, 1.8], color='#FF9800', lw=1.5, linestyle='--', alpha=0.7)
ax2.text(7.7, 6.5, 'Residual\nPath', ha='left', va='center', fontsize=8, color='#FF9800')

# 主路径
# pw_reduce
draw_box(ax2, 5.5, 9.5, 2.0, 0.6, 'pw_reduce\n1×1 Conv  (C → C/2)', color='#E3F2FD', edge='#2196F3', fontsize=9)
draw_arrow(ax2, 5.5, 9.2, 5.5, 8.6)

# 三分支并行空洞卷积
ax2.text(5.5, 8.75, 'Parallel Multi-Scale DWConv', ha='center', va='center',
         fontsize=10, weight='bold', color='#1565C0')

branch_labels = ['dw1\n3×3  d=1\nRF=3×3', 'dw2\n3×3  d=2\nRF=5×5', 'dw3\n3×3  d=3\nRF=7×7']
branch_colors = ['#E8F5E9', '#E3F2FD', '#F3E5F5']
branch_edges = ['#66BB6A', '#42A5F5', '#AB47BC']
for i, (label, color, edge) in enumerate(zip(branch_labels, branch_colors, branch_edges)):
    x = 3.0 + i * 2.5
    draw_box(ax2, x, 7.8, 1.8, 0.7, label, color=color, edge=edge, fontsize=8)
    draw_arrow(ax2, 5.5, 8.3, x, 8.15, color='#999999')
    draw_arrow(ax2, x, 7.45, 5.5, 6.9, color='#999999')

# 相加
draw_box(ax2, 5.5, 6.6, 1.0, 0.4, '⊕', color='#FFFFFF', edge='#333333', fontsize=14, bold=True)
draw_arrow(ax2, 5.5, 6.4, 5.5, 5.9)

# pw_expand + BN + SiLU
draw_box(ax2, 5.5, 5.5, 2.2, 0.6, 'pw_expand + BN + SiLU\n1×1 Conv (C/2 → C)', color='#E3F2FD', edge='#2196F3', fontsize=9)
draw_arrow(ax2, 5.5, 5.2, 5.5, 4.7)

# SE 注意力
se_box = FancyBboxPatch((3.5, 3.8), 4.0, 1.5,
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         facecolor='#FCE4EC', edgecolor='#EC407A', linewidth=2)
ax2.add_patch(se_box)
ax2.text(5.5, 5.0, 'SE Channel Attention', ha='center', va='center',
         fontsize=11, weight='bold', color='#C2185B')

# SE 内部
draw_box(ax2, 4.0, 4.3, 1.4, 0.45, 'GlobalAvgPool\n(H×W → 1×1)', color='#FFFFFF', edge='#EC407A', fontsize=7.5)
draw_arrow(ax2, 4.7, 4.3, 5.2, 4.3, color='#EC407A')
draw_box(ax2, 5.8, 4.3, 1.4, 0.45, 'FC: C → C/4\nSiLU', color='#FFFFFF', edge='#EC407A', fontsize=7.5)
draw_arrow(ax2, 6.5, 4.3, 7.0, 4.3, color='#EC407A')
draw_box(ax2, 7.6, 4.3, 1.4, 0.45, 'FC: C/4 → C\nSigmoid', color='#FFFFFF', edge='#EC407A', fontsize=7.5)
ax2.text(5.5, 3.95, 'Channel-wise reweighting: x = x ⊗ SE(x)', ha='center', va='center',
         fontsize=8.5, color='#AD1457', style='italic')

draw_arrow(ax2, 5.5, 3.8, 5.5, 3.2)

# 输出前: 与残差相加
draw_box(ax2, 5.5, 2.7, 2.2, 0.5, 'x_out × scale  +  residual\n(scale: learnable, init=0.1)', color='#FFF8E1', edge='#FFA000', fontsize=9)
draw_arrow(ax2, 5.5, 2.45, 5.5, 1.9)

# 最终输出
draw_box(ax2, 5.5, 1.5, 2.0, 0.6, 'Enhanced Feature\n(B, C, H, W)', color='#E8F5E9', edge='#4CAF50', fontsize=10, bold=True)

# 连接残差虚线到最终相加点
ax2.plot([7.5, 7.5], [2.95, 2.7], color='#FF9800', lw=1.5, linestyle='--', alpha=0.7)
ax2.annotate('', xy=(6.6, 2.7), xytext=(7.5, 2.7),
             arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1.5, linestyle='--'))

# 右侧参数统计
stat_box = FancyBboxPatch((8.8, 6.5), 2.0, 4.0,
                           boxstyle="round,pad=0.02,rounding_size=0.15",
                           facecolor='#FAFAFA', edgecolor='#BBBBBB', linewidth=1)
ax2.add_patch(stat_box)
ax2.text(9.8, 10.2, 'Params (C=256)', ha='center', va='center',
         fontsize=10, weight='bold', color='#333333')
stats = [
    'pw_reduce:  32.8K',
    '3×DWConv:    3.5K',
    'pw_expand:  32.8K',
    'SE module:  16.4K',
    '----------------',
    'Total:      ~85K',
]
for i, s in enumerate(stats):
    color = '#333333' if 'Total' not in s else '#D32F2F'
    weight = 'bold' if 'Total' in s else 'normal'
    ax2.text(9.8, 9.7 - i*0.45, s, ha='center', va='center',
             fontsize=9, color=color, weight=weight, family='monospace')

# 底部公式
ax2.text(5.5, 0.6, r'$y = x + \sigma \cdot \left[ \text{SE}\left( \text{SiLU}\left(\text{BN}\left(\text{Conv}_{1\times1}\left(\sum_{d=1}^{3}\text{DWConv}_d(\text{Conv}_{1\times1}(x))\right)\right)\right)\right) \right]$',
         ha='center', va='center', fontsize=10, color='#555555')

plt.tight_layout()
fig2.savefig('/home/zhangs02/ultralytics-8.4.21/v5/v5/plots/dyblock_detail.png',
             dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("[保存] dyblock_detail.png")

print("\n两张图已生成:")
print("  1. plots/dyhead_overview.png  — 整体架构对比")
print("  2. plots/dyblock_detail.png   — DyBlock 内部细节")
