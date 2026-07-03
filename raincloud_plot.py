"""
Raincloud Plot (雨云图): 半小提琴 + 箱线图 + 抖动散点
复现参考图中 GraphViT / PhysGTO 风格的组合图

原理:
  左侧: 半小提琴图 (kde 密度)  → 展示分布形状
  中间: 箱线图               → 展示四分位数、中位数
  右侧: 抖动散点 (jitter)     → 展示每一个真实数据点
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 0. 全局字体与排版设置 (Arial & Custom parameters)
# ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'Arial',
    'font.weight': 'normal',
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'axes.unicode_minus': False,  # 正常显示负号
})

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

# ──────────────────────────────────────────────
# 1. 读取数据
# ──────────────────────────────────────────────
csv_dir = Path('csv_outside/csv_outside_25')
metric = 'mae'  # 可选 'mae', 'rmse', 'l2_loss'

model_configs = [
    # (csv文件名前缀, 显示名称)
    ('fno',              '1'),
    ('afno',             '2'),
    ('transolver',       '3'),
    ('flronet',          '4'),
    ('CATO-afno',        '5'),
    ('CATO-transolver',  '6'),
]

data = []
labels = []
for prefix, display_name in model_configs:
    csv_path = csv_dir / f'{prefix}_outside.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        data.append(df[metric].values)
        labels.append(display_name)
    else:
        print(f"Warning: {csv_path} not found, skipping.")

n_models = len(data)

# ──────────────────────────────────────────────
# 2. 颜色方案
# ──────────────────────────────────────────────
# 基线用冷色调，CATO用暖色调（与参考图风格一致）
colors = [
    '#8C8C8C',  # FNO      - 灰色
    '#5B9BD5',  # AFNO     - 蓝色
    '#70AD47',  # Transolver - 绿色
    '#9B59B6',  # FLRONet  - 紫色
    '#ED7D31',  # CATO-AFNO - 橙色
    '#E74C3C',  # CATO-Trans - 红色
][:n_models]

# ──────────────────────────────────────────────
# 3. 绘图
# ──────────────────────────────────────────────
# 宽度设为 180mm (180 / 25.4 英寸)，高度设为 3.8 英寸 (约 96mm) 保持优雅比例
fig, ax = plt.subplots(figsize=(80 / 25.4, 60 / 25.4))

# 关键参数
positions = np.arange(1, n_models + 1)
violin_width = 0.35       # 半小提琴的最大宽度
box_width = 0.15          # 箱线图宽度
jitter_width = 0.12       # 散点抖动的范围
jitter_offset = 0.18      # 散点向右偏移量

for i, (vals, pos, color) in enumerate(zip(data, positions, colors)):
    # ── 3a. 半小提琴 (左侧) ──
    kde = gaussian_kde(vals)
    y_range = np.linspace(vals.min(), vals.max(), 300)
    density = kde(y_range)
    # 归一化密度到 violin_width
    density_scaled = density / density.max() * violin_width

    # 画在左侧: x 从 pos 向左延伸
    ax.fill_betweenx(
        y_range,
        pos - density_scaled,   # 左边界
        pos,                    # 右边界 (中心线)
        alpha=0.7,
        color=color,
        edgecolor='none',
    )

    # ── 3b. 箱线图 (中间偏右) ──
    bp = ax.boxplot(
        vals,
        positions=[pos + 0.02],
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        zorder=3,
    )
    # 样式设置
    bp['boxes'][0].set_facecolor('white')
    bp['boxes'][0].set_edgecolor('black')
    bp['boxes'][0].set_linewidth(0.5)
    bp['medians'][0].set_color('black')
    bp['medians'][0].set_linewidth(0.7)
    for element in ['caps', 'whiskers']:
        for line in bp[element]:
            line.set_color('black')
            line.set_linewidth(0.5)

    # ── 3c. 抖动散点 (右侧) ──
    n_points = len(vals)
    jitter = np.random.default_rng(42).uniform(
        -jitter_width / 2, jitter_width / 2, n_points
    )
    ax.scatter(
        pos + jitter_offset + jitter,
        vals,
        s=2,
        alpha=0.6,
        color=color,
        edgecolors='none',
        zorder=2,
    )

# ──────────────────────────────────────────────
# 4. 坐标轴与标签
# ──────────────────────────────────────────────
ax.set_xticks(positions)
ax.set_xticklabels(labels)                  # 自动使用 xtick.labelsize: 8
ax.set_ylabel(metric.upper())               # 自动使用 axes.labelsize: 9
ax.set_title('Turbulent Flow (Extrapolation)', fontsize=8, fontweight='normal', pad=12)

# 移除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 去掉手动覆盖 tick_params 的字号，让其使用全局的 8pt
plt.tight_layout()

# 保存
output_path = Path('plots_paper') / f'raincloud_{metric}_outside.png'
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=1200, bbox_inches='tight', transparent=True)
print(f"Saved to {output_path}")
