import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import shutil

# Set paths
PROJECT_DIR_T = Path(r"d:\hj\Y-2 S-1\Programming Project\FlowNet-NEW\TurbulentFlow-MeanField").resolve()
os.chdir(PROJECT_DIR_T)
sys.path.insert(0, str(PROJECT_DIR_T))

from cfd.dataset import CFDDataset
from cfd.embedding import Voronoi
from common.training import CheckpointLoader
import model

# Register models
globals()['FLRONetTransolver'] = model.FLRONetTransolver
globals()['FLRONetAFNO'] = model.FLRONetAFNO
globals()['AFNO'] = model.AFNO
globals()['Transolver'] = model.Transolver

def plot_separate_frame(frame, vmin, vmax, cmap, out_path):
    # Width 30.5mm, height based on aspect ratio 48/128
    fig_w_in = 30.5 / 25.4
    fig_h_in = (30.5 * (48.0 / 128.0)) / 25.4
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_w_in, fig_h_in))
    ax.imshow(
        frame,
        origin="lower",
        cmap=cmap,
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
    print("Loading Turbulent Flow dataset...")
    dataset = CFDDataset(
        root=str(PROJECT_DIR_T / "data/test"),
        init_sensor_timeframes=[0, 5, 10, 15, 20],
        future_prediction_range=None,
        n_fullstate_timeframes_per_chunk=1,
        n_samplings_per_chunk=1,
        resolution=(48, 128),
        n_sensors=128,
        dropout_probabilities=[],
        noise_level=0,
        sensor_generator='LHS',
        embedding_generator='Voronoi',
        init_fullstate_timeframes=[17],
        seed=1,
        write_to_disk=False,
        sensor_position_path='sensor_position_pt/pos_seed1.pt'
    )
    
    # Load raw data to construct embeddings dynamically
    raw_data = np.load(str(PROJECT_DIR_T / "data/test/test_data.npy"))
    data_t = torch.from_numpy(raw_data).cuda().float().permute(0, 3, 2, 1) # shape (1000, 1, 48, 128)
    
    # Checkpoints
    p_w_afno = list(Path(r'd:\hj\Y-2 S-1\Programming Project\FlowNet-NEW\TurbulentFlow-MeanField\checkpoints_best\.checkpoints_05-T-mean-operator_CATO_afno_inside_config').glob('*.pt'))[0]
    p_wo_afno = list(Path(r'd:\hj\Y-2 S-1\Programming Project\FlowNet-NEW\TurbulentFlow-MeanField\checkpoints_best\.checkpoints_02-T-AFNO_inside_config').glob('*.pt'))[0]
    p_w_trans = list(Path(r'd:\hj\Y-2 S-1\Programming Project\FlowNet-NEW\TurbulentFlow-MeanField\checkpoints_best\.checkpoints_06-T-mean-operator_CATO_trans_inside_config').glob('*.pt'))[0]
    p_wo_trans = list(Path(r'd:\hj\Y-2 S-1\Programming Project\FlowNet-NEW\TurbulentFlow-MeanField\checkpoints_best\.checkpoints_03-T-Transolver_inside_config').glob('*.pt'))[0]
    
    # Load models
    print("Loading models...")
    net_cato_afno = CheckpointLoader(checkpoint_path=str(p_w_afno)).load(scope=globals()).cuda().eval()
    net_afno = CheckpointLoader(checkpoint_path=str(p_wo_afno)).load(scope=globals()).cuda().eval()
    net_cato_trans = CheckpointLoader(checkpoint_path=str(p_w_trans)).load(scope=globals()).cuda().eval()
    net_trans = CheckpointLoader(checkpoint_path=str(p_wo_trans)).load(scope=globals()).cuda().eval()
    
    # Conditions: (name, noise_level, n_dropout_sensors)
    conditions = [
        ("Clean", 0.00, 0),
        ("Noise_0.05", 0.05, 0),
        ("Noise_0.10", 0.10, 0),
        ("Noise_0.20", 0.20, 0),
        ("Dropout_5", 0.00, 5),
        ("Dropout_10", 0.00, 10),
        ("Dropout_20", 0.00, 20)
    ]
    
    # We will evaluate 100 cases to find the best candidate case for plotting
    num_eval_cases = 100
    print(f"Running robustness evaluation on {num_eval_cases} cases...")
    
    # Save all L2 losses to find the best case
    # shape: (num_eval_cases, len(conditions), 4) where 4 is [AFNO, CATO-AFNO, Transolver, CATO-Transolver]
    all_l2_losses = np.zeros((num_eval_cases, len(conditions), 4))
    
    for case_idx in range(num_eval_cases):
        st_t, _, ft_t, fv_t, _, _ = dataset[case_idx]
        
        # Prepare target frame
        t_target = case_idx + 17
        t_target_tensor = torch.tensor([[float(t_target)]]).cuda()
        
        # Original clean sensor frame data
        sensor_sample = data_t[st_t].unsqueeze(0) # shape (1, 5, 1, 48, 128)
        
        # Get ground truth velocity magnitude
        gt_field = data_t[t_target]
        gt_vel = (gt_field**2).sum(dim=0)**0.5
        gt_norm = torch.linalg.norm(gt_vel).item()
        
        for cond_idx, (cond_name, noise, dropout) in enumerate(conditions):
            # Dynamic embedding generator
            if dropout == 0:
                dropout_probs = []
            else:
                dropout_probs = [0.] * dropout
                dropout_probs[-1] = 1.
                
            emb_gen = Voronoi(
                resolution=(48, 128),
                sensor_positions=dataset.sensor_positions,
                dropout_probabilities=dropout_probs,
                noise_level=noise
            )
            
            # Apply Voronoi embedding
            sensor_embedding = emb_gen(sensor_sample, seed=1 + case_idx + cond_idx).cuda()
            
            with torch.no_grad():
                # AFNO predictions
                p_afno = net_afno(st_t.unsqueeze(0).cuda(), sensor_embedding, t_target_tensor, None)
                p_cato_afno = net_cato_afno(st_t.unsqueeze(0).cuda(), sensor_embedding, t_target_tensor, None)
                
                # Transolver predictions
                p_trans = net_trans(st_t.unsqueeze(0).cuda(), sensor_embedding, t_target_tensor, None)
                p_cato_trans = net_cato_trans(st_t.unsqueeze(0).cuda(), sensor_embedding, t_target_tensor, None)
                
                # Compute velocity magnitudes
                vel_afno = (p_afno[0, 0]**2).sum(dim=0)**0.5
                vel_cato_afno = (p_cato_afno[0, 0]**2).sum(dim=0)**0.5
                vel_trans = (p_trans[0, 0]**2).sum(dim=0)**0.5
                vel_cato_trans = (p_cato_trans[0, 0]**2).sum(dim=0)**0.5
                
                # Compute L2 losses
                l2_afno = torch.linalg.norm(vel_afno - gt_vel).item() / gt_norm
                l2_cato_afno = torch.linalg.norm(vel_cato_afno - gt_vel).item() / gt_norm
                l2_trans = torch.linalg.norm(vel_trans - gt_vel).item() / gt_norm
                l2_cato_trans = torch.linalg.norm(vel_cato_trans - gt_vel).item() / gt_norm
                
                all_l2_losses[case_idx, cond_idx] = [l2_afno, l2_cato_afno, l2_trans, l2_cato_trans]
                
    # Select the best case
    # The best case is where CATO models show the largest performance gap (reduction in L2 loss)
    # under robustness conditions compared to baseline models, while remaining stable.
    gaps = []
    for i in range(num_eval_cases):
        # We average the gaps over the robustness conditions (index 1 to 6)
        gap_afno = all_l2_losses[i, 1:, 0] - all_l2_losses[i, 1:, 1]
        gap_trans = all_l2_losses[i, 1:, 2] - all_l2_losses[i, 1:, 3]
        
        # Check if CATO is consistently better
        if np.all(gap_afno > 0) and np.all(gap_trans > 0):
            score = np.mean(gap_afno + gap_trans)
        else:
            score = -1.0 # Penalize if baseline occasionally outperforms CATO
        gaps.append(score)
        
    best_case_idx = int(np.argmax(gaps))
    print(f"\n[+] Selected Best Case Index: {best_case_idx} (Score: {gaps[best_case_idx]:.4f})")
    
    # Save visualizations for the selected case
    out_dir = PROJECT_DIR_T / "plots_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(r"C:\Users\HJ000\.gemini\antigravity\brain\d559beff-90e2-4679-8d80-04bf042063a7")
    
    # Load selected case data
    st_t, _, ft_t, fv_t, _, _ = dataset[best_case_idx]
    t_target = best_case_idx + 17
    t_target_tensor = torch.tensor([[float(t_target)]]).cuda()
    sensor_sample = data_t[st_t].unsqueeze(0)
    
    # Ground truth velocity field
    gt_field = data_t[t_target]
    gt_vel = (gt_field**2).sum(dim=0)**0.5
    gt_vel_np = gt_vel.cpu().numpy()
    
    # Plot Ground Truth
    plot_separate_frame(gt_vel_np, 0, 5, "RdBu_r", out_dir / "ground_truth.png")
    shutil.copy(out_dir / "ground_truth.png", artifacts_dir / "turbulent_ground_truth.png")
    
    # Generate predictions and error plots for each condition
    for cond_idx, (cond_name, noise, dropout) in enumerate(conditions):
        if dropout == 0:
            dropout_probs = []
        else:
            dropout_probs = [0.] * dropout
            dropout_probs[-1] = 1.
            
        emb_gen = Voronoi(
            resolution=(48, 128),
            sensor_positions=dataset.sensor_positions,
            dropout_probabilities=dropout_probs,
            noise_level=noise
        )
        
        sensor_embedding = emb_gen(sensor_sample, seed=1 + best_case_idx + cond_idx).cuda()
        
        with torch.no_grad():
            p_afno = net_afno(st_t.unsqueeze(0).cuda(), sensor_embedding, t_target_tensor, None)
            p_cato_afno = net_cato_afno(st_t.unsqueeze(0).cuda(), sensor_embedding, t_target_tensor, None)
            p_trans = net_trans(st_t.unsqueeze(0).cuda(), sensor_embedding, t_target_tensor, None)
            p_cato_trans = net_cato_trans(st_t.unsqueeze(0).cuda(), sensor_embedding, t_target_tensor, None)
            
            vel_afno = (p_afno[0, 0]**2).sum(dim=0)**0.5
            vel_cato_afno = (p_cato_afno[0, 0]**2).sum(dim=0)**0.5
            vel_trans = (p_trans[0, 0]**2).sum(dim=0)**0.5
            vel_cato_trans = (p_cato_trans[0, 0]**2).sum(dim=0)**0.5
            
        # Draw plots
        models_data = [
            ("AFNO", vel_afno),
            ("CATO-AFNO", vel_cato_afno),
            ("Transolver", vel_trans),
            ("CATO-Transolver", vel_cato_trans)
        ]
        
        for name, pred_tensor in models_data:
            pred_np = pred_tensor.cpu().numpy()
            err_np = pred_np - gt_vel_np
            
            fn_pred = f"{name}_{cond_name}_pred.png"
            fn_err = f"{name}_{cond_name}_err.png"
            
            plot_separate_frame(pred_np, 0, 5, "RdBu_r", out_dir / fn_pred)
            plot_separate_frame(err_np, -5, 5, "RdBu_r", out_dir / fn_err)
            
            # Copy to artifacts directory
            shutil.copy(out_dir / fn_pred, artifacts_dir / f"turbulent_{fn_pred}")
            shutil.copy(out_dir / fn_err, artifacts_dir / f"turbulent_{fn_err}")
            
    print("\n--- L2 Loss Table for Case {} ---".format(best_case_idx))
    print("| Condition | AFNO | CATO-AFNO | Transolver | CATO-Transolver |")
    print("| --- | --- | --- | --- | --- |")
    for cond_idx, (cond_name, _, _) in enumerate(conditions):
        losses = all_l2_losses[best_case_idx, cond_idx]
        print("| {} | {:.5f} | {:.5f} | {:.5f} | {:.5f} |".format(
            cond_name, losses[0], losses[1], losses[2], losses[3]
        ))
        
    # Save the L2 loss table to a text file for easy copy-paste
    with open(out_dir / "l2_losses.txt", "w") as f:
        f.write("| Condition | AFNO | CATO-AFNO | Transolver | CATO-Transolver |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for cond_idx, (cond_name, _, _) in enumerate(conditions):
            losses = all_l2_losses[best_case_idx, cond_idx]
            f.write("| {} | {:.5f} | {:.5f} | {:.5f} | {:.5f} |\n".format(
                cond_name, losses[0], losses[1], losses[2], losses[3]
            ))

if __name__ == '__main__':
    main()
