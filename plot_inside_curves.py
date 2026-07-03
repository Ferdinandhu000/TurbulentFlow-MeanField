import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set Arial font parameters (scaled down for small 60x40mm figure)
plt.rcParams.update({
    'font.family': 'Arial',
    'font.weight': 'normal',
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'axes.unicode_minus': False,
})

def plot_combined_curves(metric='l2_loss'):
    """
    Plots the combined mean error curves with shaded standard deviation bands.
    Inside (frames 0-20) and Outside (frames 21-25) are joined on a single plot
    with a visual gap between the two segments.
    """
    project_dir = Path(__file__).resolve().parent
    
    # Define frame sequences
    inside_frames = list(range(21))       # 0 to 20
    outside_frames = list(range(21, 26))   # 21 to 25
    
    # Visual gap width on the X-axis between frame 20 and 21
    gap = 1.0
    
    model_configs = [
        ('fno',              'FNO',              '#8C8C8C'),  # 灰色
        ('afno',             'AFNO',             '#5B9BD5'),  # 蓝色
        ('transolver',       'Transolver',       '#70AD47'),  # 绿色
        ('flronet',          'FLRONet',          '#9B59B6'),  # 紫色
        ('CATO-afno',        'CATO-AFNO',        '#ED7D31'),  # 橙色
        ('CATO-transolver',  'CATO-Transolver',  '#E74C3C'),  # 红色
    ]

    # Width: 130mm, Height: 90mm
    fig, ax = plt.subplots(figsize=(80 / 25.4, 60 / 25.4))
    
    max_y_val = 0.0

    for prefix, display_name, color in model_configs:
        means = []
        stds = []
        x_indices = []

        # 1. Load inside data (0-20)
        for f in inside_frames:
            csv_path = project_dir / 'csv_inside' / f'csv_inside_{f}' / f'{prefix}_inside.csv'
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    vals = df[metric].values
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                    x_indices.append(f)
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")

        # 2. Load outside data (21-25)
        for f in outside_frames:
            csv_path = project_dir / 'csv_outside' / f'csv_outside_{f}' / f'{prefix}_outside.csv'
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    vals = df[metric].values
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                    # Shift outside X coordinates to create a visual gap
                    x_indices.append(f + gap)
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")

        if x_indices:
            x_indices = np.array(x_indices)
            means = np.array(means)
            stds = np.array(stds)
            
            current_max = np.max(means + stds)
            if current_max > max_y_val:
                max_y_val = current_max

            # Plot inside segment (0 to 20)
            inside_mask = x_indices <= 20
            ax.plot(x_indices[inside_mask], means[inside_mask], color=color, linewidth=0.5)
            ax.fill_between(
                x_indices[inside_mask], 
                np.clip(means[inside_mask] - stds[inside_mask], 0, None), 
                means[inside_mask] + stds[inside_mask], 
                facecolor=color, alpha=0.15, edgecolor='none'
            )
            
            # Plot outside segment (21 to 25)
            outside_mask = x_indices > 20
            ax.plot(x_indices[outside_mask], means[outside_mask], label=display_name, color=color, linewidth=0.5)
            ax.fill_between(
                x_indices[outside_mask], 
                np.clip(means[outside_mask] - stds[outside_mask], 0, None), 
                means[outside_mask] + stds[outside_mask], 
                facecolor=color, alpha=0.15, edgecolor='none'
            )

    # Annotate region text
    y_text_pos = max_y_val * 0.95 if max_y_val > 0 else 0.5
    ax.text(10.0, y_text_pos, '', ha='center', va='top', fontsize=8, color='#555555', fontweight='bold')
    ax.text(23.0 + gap, y_text_pos, '', ha='center', va='top', fontsize=8, color='#555555', fontweight='bold')

    # Styling and spines (framed with a rectangle)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(0.8)
    
    # Custom X-ticks: inside (0, 5, 10, 15, 20), outside (21, 22, 23, 24, 25)
    xticks_labels = [5, 10, 15, 20, 21, 25]
    xticks_positions = [5, 10, 15, 20, 21 + gap, 25 + gap]
    
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels([str(t) for t in xticks_labels])
    ax.set_xlim(0, 25.0 + gap)
    ax.set_xlabel('Timestep')
    y_labels = {
        'l2_loss': 'L2 loss',
        'mae': 'MAE',
        'rmse': 'RMSE'
    }
    ax.set_ylabel(y_labels.get(metric, metric.upper()))
    
    # Position legend (Removed legend as requested)
    # ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.02, 0.90))
    
    # Save output
    output_dir = project_dir / 'plots_paper'
    output_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = output_dir / f'turbulent_combined_{metric}_curves.png'
    
    plt.tight_layout()
    plt.savefig(out_img_path, dpi=1200, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"[*] Successfully saved combined plot with gap to {out_img_path}")

if __name__ == '__main__':
    plot_combined_curves(metric='l2_loss')
    plot_combined_curves(metric='mae')
    plot_combined_curves(metric='rmse')
