import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil

def main():
    project_dir = Path(__file__).resolve().parent
    artifacts_dir = Path(r"C:\Users\HJ000\.gemini\antigravity\brain\d559beff-90e2-4679-8d80-04bf042063a7")
    
    # Configure exact matplotlib publication styles (80mm x 60mm) with 0.4 linewidth
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.weight': 'normal',
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 6.5,
        'axes.linewidth': 0.4,
        'xtick.major.width': 0.4,
        'ytick.major.width': 0.4,
        'axes.unicode_minus': False,
    })
    
    # Load data
    json_path = project_dir / "turbulent_evaluation_results.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    sensor_counts = [96, 128, 160, 192]
    models = ["AFNO", "CATO-afno", "Transolver", "CATO-trans"]
    
    # Colors matching project standard exactly
    model_colors = {
        "AFNO":        "#5B9BD5",  # 蓝色
        "CATO-afno":   "#ED7D31",  # 橙色
        "Transolver":  "#70AD47",  # 绿色
        "CATO-trans":  "#E74C3C"   # 红色
    }
    
    model_labels = {
        "AFNO":        "AFNO",
        "CATO-afno":   "CATONet-AFNO",
        "Transolver":  "Transolver",
        "CATO-trans":  "CATONet-Transolver"
    }
    
    # Reorganize data: {model_name: {n_sensors: mae}}
    models_data = {m: {} for m in models}
    for item in data:
        name = item["name"]
        sensors = item["sensors"]
        if name in models_data and sensors in sensor_counts:
            models_data[name][sensors] = item["mae"]
            
    fig, ax = plt.subplots(figsize=(80 / 25.4, 60 / 25.4))
    
    x = np.arange(len(sensor_counts))
    w = 0.10
    gap = 0.10
    
    # Symmetrical grouping centers: AFNO series (1 and 2) vs Transolver series (3 and 4)
    # AFNO series touch exactly at x - w - gap/2
    # Transolver series touch exactly at x + w + gap/2
    pos1 = x - 1.5 * w - 0.5 * gap
    pos2 = x - 0.5 * w - 0.5 * gap
    pos3 = x + 0.5 * w + 0.5 * gap
    pos4 = x + 1.5 * w + 0.5 * gap
    
    # Plot grouped bars
    ax.bar(pos1, [models_data["AFNO"][s] for s in sensor_counts], w, 
           label=model_labels["AFNO"], color=model_colors["AFNO"], edgecolor='none')
           
    ax.bar(pos2, [models_data["CATO-afno"][s] for s in sensor_counts], w, 
           label=model_labels["CATO-afno"], color=model_colors["CATO-afno"], edgecolor='none')
           
    ax.bar(pos3, [models_data["Transolver"][s] for s in sensor_counts], w, 
           label=model_labels["Transolver"], color=model_colors["Transolver"], edgecolor='none')
           
    ax.bar(pos4, [models_data["CATO-trans"][s] for s in sensor_counts], w, 
           label=model_labels["CATO-trans"], color=model_colors["CATO-trans"], edgecolor='none')
        
    ax.set_xlabel("Number of Sensors")
    ax.set_ylabel("MAE")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sensor_counts])
    ax.set_ylim(0.30, 0.49)  # Expand slightly to fit legend cleanly
    ax.grid(True, linestyle="--", alpha=0.5, color="#cccccc", linewidth=0.5)
    
    # Frame border (spines set to 0.4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(0.4)
        
    # Legend in upper right, no border frame
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    
    output_dir = project_dir / "plots_paper"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "turbulent_sensor_mae_curves.png"
    plt.savefig(out_file, dpi=1200, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Saved Turbulent sensor MAE bar chart to {out_file}")
    
    # Copy to artifacts for preview
    if artifacts_dir.exists():
        shutil.copy(out_file, artifacts_dir / "turbulent_sensor_mae_curves.png")

if __name__ == '__main__':
    main()
