import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Ensure the project root is on PYTHONPATH
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from common.training import CheckpointLoader
from cfd.dataset import CFDDataset
from model import FLRONetFNO, FLRONetAFNO, FLRONetUNet, FLRONetMLP, FNO3D, FLRONetTransolver, FNO, AFNO, Transolver, UNet

def plot_frame(frame, vmin, vmax, out_path):
    """
    Saves a clean transparent flow field visualization.
    """
    figwidth = 20 / 25.4
    figheight = 5 / 25.4
    
    fig, ax = plt.subplots(1, 1, figsize=(figwidth, figheight))
    ax.imshow(
        frame,
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="bicubic",
        aspect="auto",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def main():
    config_path = os.path.join(PROJECT_DIR, "config_plot.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    checkpoint_path = os.path.join(PROJECT_DIR, "checkpoints_best/.checkpoints_06-T-mean-operator_CATO_trans_inside_config/flronettransolver86.pt")
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load model
    checkpoint_loader = CheckpointLoader(checkpoint_path=checkpoint_path)
    net = checkpoint_loader.load(scope=globals())
    net = net.cuda()
    net.eval()
    print(f"Loaded {net.__class__.__name__} successfully!")

    # Instantiate dataset
    init_sensor_timeframes = list(config['dataset']['init_sensor_timeframes'])
    resolution = tuple(config['dataset']['resolution'])
    n_sensors = int(config['dataset']['n_sensors'])
    sensor_generator = str(config['dataset']['sensor_generator'])
    embedding_generator = str(config['dataset']['embedding_generator'])
    sensor_position_path = str(config['evaluate'].get('sensor_position_path', 'sensor_position_pt/pos_seed1.pt'))
    
    dataset = CFDDataset(
        root=os.path.join(PROJECT_DIR, 'data/test'),
        init_sensor_timeframes=init_sensor_timeframes,
        future_prediction_range=config['dataset'].get('future_prediction_range'),
        n_fullstate_timeframes_per_chunk=len(config['evaluate']['init_fullstate_timeframes']),
        n_samplings_per_chunk=1,
        resolution=resolution,
        n_sensors=n_sensors,
        dropout_probabilities=[],
        noise_level=0.0,
        sensor_generator=sensor_generator,
        embedding_generator=embedding_generator,
        init_fullstate_timeframes=config['evaluate']['init_fullstate_timeframes'],
        seed=int(config['dataset'].get('seed', 1)),
        write_to_disk=True,
        sensor_position_path=sensor_position_path,
    )

    print(f"Dataset loaded. Total chunks: {len(dataset)}")

    # We want to collect outputs for target frames 100 to 120.
    # Chunk 100 covers target frames 100 to 119.
    # Chunk 101 covers target frames 101 to 120.
    chunks_to_run = [100]
    
    plots_root = os.path.join(PROJECT_DIR, "plots_fusing")
    os.makedirs(plots_root, exist_ok=True)

    branch_stats_printed = False

    for chunk_idx in chunks_to_run:
        # Load chunk data
        st_tensor, sv_tensor, ft_tensor, fv_tensor, case_name, sampling_id = dataset[chunk_idx]
        
        # Move to GPU and add batch dimension
        st_t = st_tensor.unsqueeze(0).cuda()
        sv_t = sv_tensor.unsqueeze(0).cuda()
        ft_t = ft_tensor.unsqueeze(0).cuda()
        
        with torch.no_grad():
            # 1. Temporal embeddings
            fullstate_emb = net.sinusoid_embedding(ft_t)
            sensor_emb = net.sinusoid_embedding(st_t)
            
            # 2. TrunkNet outputs
            trunk_outputs = net.trunk_net(fullstate_emb, sensor_emb)
            
            # 3. BranchNet outputs
            branch_outputs = []
            for i in range(net.n_stacked_networks):
                branch_net = net.branch_nets[i]
                if branch_net.__class__.__name__ in ("FNOBranchNet", "TransolverBranchNet"):
                    branch_output = branch_net(sensor_values=sv_t, out_resolution=resolution)
                else:
                    branch_output = branch_net(sensor_values=sv_t)
                branch_outputs.append(branch_output)
            
            # Print BranchNet statistics once
            if not branch_stats_printed:
                print("\n=== BranchNet Output Statistics ===")
                for i in range(net.n_stacked_networks):
                    val = branch_outputs[i].cpu().numpy()
                    print(f"Stack {i} BranchNet Output (shape: {val.shape}):")
                    print(f"  Max : {np.max(val):.6f}")
                    print(f"  Min : {np.min(val):.6f}")
                    print(f"  Mean: {np.mean(val):.6f}")
                    print(f"  Std : {np.std(val):.6f}")
                branch_stats_printed = True
            
            # 4. Cross-Attention Fusion for each stack
            fused_s0 = torch.einsum('nfs,nschw->nfchw', trunk_outputs[0], branch_outputs[0])
            fused_s1 = torch.einsum('nfs,nschw->nfchw', trunk_outputs[1], branch_outputs[1])
            fused_sum = fused_s0 + fused_s1
            
            # 5. Mean field bias mu
            if net.mean_field_net is not None:
                if net.use_mean_field == 'branch_output':
                    mean_field_input = sum(branch_outputs)
                else:
                    mean_field_input = sv_t
                mu = net.mean_field_net(
                    sensor_values=mean_field_input,
                    sensor_times=st_t,
                    fullstate_times=ft_t,
                )
                final_out = fused_sum + mu
            else:
                mu = None
                final_out = fused_sum

        # Extract values for the target frames we need
        target_frames_in_chunk = ft_tensor.tolist() # Length 20
        for offset, t_frame in enumerate(target_frames_in_chunk):
            if 100 <= t_frame <= 120:
                frame_dir = os.path.join(plots_root, str(int(t_frame)))
                os.makedirs(frame_dir, exist_ok=True)
                
                # Ground truth frame
                gt = fv_tensor[offset, 0].cpu().numpy()
                vmin = np.min(gt)
                vmax = np.max(gt)
                
                # Get stack frames
                s0_frame = fused_s0[0, offset, 0].cpu().numpy()
                s1_frame = fused_s1[0, offset, 0].cpu().numpy()
                sum_frame = fused_sum[0, offset, 0].cpu().numpy()
                final_frame = final_out[0, offset, 0].cpu().numpy()
                
                # Print stats for frame 100 as a reference
                if int(t_frame) == 100:
                    print("\n=== Intermediate Fused Flow Field Statistics (Frame 100) ===")
                    print(f"Stack 0 Fused: max={np.max(s0_frame):.6f}, min={np.min(s0_frame):.6f}, mean={np.mean(s0_frame):.6f}, std={np.std(s0_frame):.6f}")
                    print(f"Stack 1 Fused: max={np.max(s1_frame):.6f}, min={np.min(s1_frame):.6f}, mean={np.mean(s1_frame):.6f}, std={np.std(s1_frame):.6f}")
                    print(f"Sum (Stack0+1): max={np.max(sum_frame):.6f}, min={np.min(sum_frame):.6f}, mean={np.mean(sum_frame):.6f}, std={np.std(sum_frame):.6f}")
                    if mu is not None:
                        mu_frame = mu[0, offset, 0].cpu().numpy()
                        print(f"MeanField μ  : max={np.max(mu_frame):.6f}, min={np.min(mu_frame):.6f}, mean={np.mean(mu_frame):.6f}, std={np.std(mu_frame):.6f}")
                    print(f"Final Output  : max={np.max(final_frame):.6f}, min={np.min(final_frame):.6f}, mean={np.mean(final_frame):.6f}, std={np.std(final_frame):.6f}")
                    print(f"Ground Truth  : max={vmin:.6f}, min={vmax:.6f}, mean={np.mean(gt):.6f}, std={np.std(gt):.6f}")
                
                # Plot and save
                plot_frame(s0_frame, vmin, vmax, os.path.join(frame_dir, "fused_stack0.png"))
                plot_frame(s1_frame, vmin, vmax, os.path.join(frame_dir, "fused_stack1.png"))
                plot_frame(sum_frame, vmin, vmax, os.path.join(frame_dir, "fused_sum.png"))
                if mu is not None:
                    mu_frame = mu[0, offset, 0].cpu().numpy()
                    plot_frame(mu_frame, vmin, vmax, os.path.join(frame_dir, "mean_field.png"))
                plot_frame(final_frame, vmin, vmax, os.path.join(frame_dir, "final_prediction.png"))
                plot_frame(gt, vmin, vmax, os.path.join(frame_dir, "ground_truth.png"))

    print("\nVisualizations successfully saved in plots_fusing/ for frames 100 to 120!")

if __name__ == "__main__":
    main()
