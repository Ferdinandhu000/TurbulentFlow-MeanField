import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure the project directory is on PYTHONPATH
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from common.training import CheckpointLoader
from model import FLRONetFNO, FLRONetAFNO, FLRONetUNet, FLRONetMLP, FNO3D, FLRONetTransolver, FNO, AFNO, Transolver, UNet

def plot_heatmap(attn_matrix, out_path):
    """
    Plots a scientific cross-attention heatmap without captions, transparent background, and keeping grid.
    """
    h, w = attn_matrix.shape
    figwidth = 5.0
    figheight = figwidth * (h / w)
    fig, ax = plt.subplots(figsize=(figwidth, figheight), dpi=300)
    
    # Use RdYlBu_r for a vibrant blue-to-orange-red colormap
    im = ax.imshow(
        attn_matrix, 
        cmap='RdYlBu_r', 
        aspect='equal', 
        origin='upper',
        vmin=None,
        vmax=None
    )
    
    # Set major ticks but clear major labels and tick marks
    ax.set_xticks(np.arange(w))
    ax.set_yticks(np.arange(h))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', size=0)
    
    # Set minor ticks exactly between pixels to draw grid lines
    ax.set_xticks(np.arange(w + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(h + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(axis='both', which='minor', size=0) # Hide minor tick marks
    
    # Set spines (borders) to black with thin linewidth to match the grid
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.5)
        
    # Make background transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    print(f"Saved transparent MeanField heatmap to: {out_path}")

def main():
    # Load config_plot.yaml
    config_path = os.path.join(PROJECT_DIR, "config_plot.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    checkpoint_path = os.path.join(PROJECT_DIR, config['evaluate']['from_checkpoint'])
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load model
    checkpoint_loader = CheckpointLoader(checkpoint_path=checkpoint_path)
    net = checkpoint_loader.load(scope=globals())
    net = net.cuda()
    net.eval()

    print(f"Loaded {net.__class__.__name__} successfully!")
    
    if net.mean_field_net is None:
        raise ValueError("The loaded model does not have a active mean_field_net.")

    # Get sensor timeframes from config
    sensor_times_list = config['dataset']['init_sensor_timeframes']
    target_times_list = config['evaluate']['init_fullstate_timeframes']
    
    print(f"Sensor timeframes (T_s = {len(sensor_times_list)}): {sensor_times_list}")
    print(f"Target timeframes (T_f = {len(target_times_list)}): {target_times_list}")

    # Convert to tensors
    sensor_times = torch.tensor([sensor_times_list], dtype=torch.float, device="cuda")
    target_times = torch.tensor([target_times_list], dtype=torch.float, device="cuda")

    with torch.no_grad():
        # Call private method _temporal_weights to get temporal weight matrix w of MeanFieldNet
        # shape: (B, T_f, T_s)
        w = net.mean_field_net._temporal_weights(sensor_times, target_times)
        w_np = w[0].cpu().numpy() # Shape: (T_f, T_s)

    # Output directory
    out_dir = os.path.join(PROJECT_DIR, "plots_paper")
    os.makedirs(out_dir, exist_ok=True)
    
    # Save raw numpy array for research backup
    npy_path = os.path.join(out_dir, "meanfield_weights.npy")
    np.save(npy_path, w_np)
    
    # Save heatmap plot
    plot_path = os.path.join(out_dir, "meanfield_heatmap.png")
    plot_heatmap(attn_matrix=w_np, out_path=plot_path)

if __name__ == "__main__":
    main()
