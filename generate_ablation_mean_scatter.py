import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import shutil

# Set working directory to project dir before any imports to ensure relative paths resolve correctly
PROJECT_DIR_T = Path(r"d:\hj\Y-2 S-1\Programming Project\FlowNet-NEW\TurbulentFlow-MeanField").resolve()
os.chdir(PROJECT_DIR_T)
sys.path.insert(0, str(PROJECT_DIR_T))

from cfd.dataset import CFDDataset
from common.functional import compute_velocity_field
from common.training import CheckpointLoader
import model

# Register models in global namespace so CheckpointLoader can resolve them
globals()['FLRONetTransolver'] = model.FLRONetTransolver
globals()['FLRONetAFNO'] = model.FLRONetAFNO
globals()['AFNO'] = model.AFNO
globals()['Transolver'] = model.Transolver

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    'font.size': 6.5,
    'axes.labelsize': 7.5,
    'legend.fontsize': 6.0,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'axes.linewidth': 0.4,
    'xtick.major.width': 0.4,
    'ytick.major.width': 0.4,
    'axes.unicode_minus': False,
})

def draw_mean_calibration_scatter(bin_centers, mean_w, mean_wo, color_w, color_wo, label_w, label_wo, filename):
    fig, ax = plt.subplots(figsize=(60 / 25.4, 50 / 25.4))
    
    # 1. Diagonal reference line
    ax.plot([0.0, 5.8], [0.0, 5.8], color='black', linestyle='--', linewidth=0.5, zorder=0)

    # 2. Scatter plots
    # Plot w/o MeanField first, then w/ MeanField (w/ on top)
    ax.scatter(mean_wo, bin_centers, color=color_wo, s=6.0, alpha=0.8, label=label_wo, linewidths=0)
    ax.scatter(mean_w, bin_centers, color=color_w, s=6.0, alpha=0.8, label=label_w, linewidths=0)
    
    ax.set_xlabel('Predicted $u$ (Mean)', fontweight='bold')
    ax.set_ylabel('Ground Truth $u$', fontweight='bold')
    
    # Limits
    ax.set_xlim(0.0, 5.8)
    ax.set_ylim(0.0, 5.8)
    
    ticks = [0.0, 1.5, 3.0, 4.5, 5.8]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    # Spine thickness
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        
    ax.legend(frameon=False, loc='upper left', markerscale=2.0)
    
    plt.tight_layout(pad=0.15)
    plt.savefig(filename, dpi=1200, bbox_inches='tight', pad_inches=0.01)
    plt.close()

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
        init_fullstate_timeframes=[17], # Evaluate timeframe 17
        seed=1,
        write_to_disk=False,
        sensor_position_path=None
    )
    
    # Load original test_data.npy to get exact ground truth values
    raw_data = np.load(r'd:\hj\Y-2 S-1\Programming Project\FlowNet-NEW\TurbulentFlow-MeanField\data\test\test_data.npy')
    data_t = torch.from_numpy(raw_data).cuda().float().permute(0, 3, 2, 1) # shape (1000, 1, 48, 128)
    
    # ========================================================================
    # CORRECT CHECKPOINT PATHS
    # w/ MF: models trained with use_mean_field='operator'
    # w/o MF: models trained WITHOUT MeanField (use_mean_field=False/none),
    #         found in result-20260519 experiment directory
    # ========================================================================
    
    # w/ MeanField checkpoints (from TurbulentFlow-MeanField/checkpoints_best)
    p_w_afno = list(Path(r'd:\hj\Y-2 S-1\Programming Project\FlowNet-NEW\TurbulentFlow-MeanField\checkpoints_best\.checkpoints_05-T-mean-operator_CATO_afno_inside_config').glob('*.pt'))[0]
    p_w_trans = list(Path(r'd:\hj\Y-2 S-1\Programming Project\FlowNet-NEW\TurbulentFlow-MeanField\checkpoints_best\.checkpoints_06-T-mean-operator_CATO_trans_inside_config').glob('*.pt'))[0]
    
    # w/o MeanField checkpoints: dedicated ablation models trained with use_mean_field='none'
    # Located in TurbulentFlow-MeanField/ablation/ directory
    p_wo_afno = Path(r'd:\hj\Y-2 S-1\Programming Project\FlowNet-NEW\TurbulentFlow-MeanField\ablation\.checkpoints_1-T-mean-operator_CATO_afno_inside_config\flronetafno20.pt')
    p_wo_trans = Path(r'd:\hj\Y-2 S-1\Programming Project\FlowNet-NEW\TurbulentFlow-MeanField\ablation\.checkpoints_2-T-mean-operator_CATO_trans_inside_config\flronettransolver85.pt')
    
    print(f"w/ MF AFNO checkpoint: {p_w_afno}")
    print(f"w/o MF AFNO checkpoint: {p_wo_afno}")
    print(f"w/ MF Transolver checkpoint: {p_w_trans}")
    print(f"w/o MF Transolver checkpoint: {p_wo_trans}")
    
    # Load all 4 models
    print("Loading w/ MF AFNO...")
    loader = CheckpointLoader(checkpoint_path=str(p_w_afno))
    net_w_afno = loader.load(scope=globals()).cuda().eval()
    print(f"  Loaded with use_mean_field = {loader.model_kwargs.get('use_mean_field')}")
    
    print("Loading w/o MF AFNO (trained without MeanField)...")
    loader = CheckpointLoader(checkpoint_path=str(p_wo_afno))
    net_wo_afno = loader.load(scope=globals()).cuda().eval()
    print(f"  Loaded with use_mean_field = {loader.model_kwargs.get('use_mean_field')}")
    
    print("Loading w/ MF Transolver...")
    loader = CheckpointLoader(checkpoint_path=str(p_w_trans))
    net_w_trans = loader.load(scope=globals()).cuda().eval()
    print(f"  Loaded with use_mean_field = {loader.model_kwargs.get('use_mean_field')}")
    
    print("Loading w/o MF Transolver (trained without MeanField)...")
    loader = CheckpointLoader(checkpoint_path=str(p_wo_trans))
    net_wo_trans = loader.load(scope=globals()).cuda().eval()
    print(f"  Loaded with use_mean_field = {loader.model_kwargs.get('use_mean_field')}")
    
    # Collect predictions for 100 cases to get stable statistics
    num_eval_cases = 100
    print(f"\nRunning inference on {num_eval_cases} cases...")
    
    all_gt = []
    
    all_pred_w_afno = []
    all_pred_wo_afno = []
    
    all_pred_w_trans = []
    all_pred_wo_trans = []
    
    for case_idx in range(num_eval_cases):
        st, sv, _, _, _, _ = dataset[case_idx]
        st_t = st.clone().detach().unsqueeze(0).cuda()
        sv_t = sv.clone().detach().unsqueeze(0).cuda()
        
        # Correct target timeframe is case_idx + 17
        t_target = case_idx + 17
        t_target_tensor = torch.tensor([[float(t_target)]]).cuda()
        
        gt_field = data_t[t_target]
        gt_vel = (gt_field**2).sum(dim=0)**0.5
        gt_np = gt_vel.cpu().numpy().flatten()
        all_gt.append(gt_np)
        
        with torch.no_grad():
            p_w_a = net_w_afno(st_t, sv_t, t_target_tensor, None)
            p_w_afno_vel = (p_w_a[0, 0]**2).sum(dim=0)**0.5
            all_pred_w_afno.append(p_w_afno_vel.cpu().numpy().flatten())
            
            p_wo_a = net_wo_afno(st_t, sv_t, t_target_tensor, None)
            p_wo_afno_vel = (p_wo_a[0, 0]**2).sum(dim=0)**0.5
            all_pred_wo_afno.append(p_wo_afno_vel.cpu().numpy().flatten())
            
            p_w_t = net_w_trans(st_t, sv_t, t_target_tensor, None)
            p_w_trans_vel = (p_w_t[0, 0]**2).sum(dim=0)**0.5
            all_pred_w_trans.append(p_w_trans_vel.cpu().numpy().flatten())
            
            p_wo_t = net_wo_trans(st_t, sv_t, t_target_tensor, None)
            p_wo_trans_vel = (p_wo_t[0, 0]**2).sum(dim=0)**0.5
            all_pred_wo_trans.append(p_wo_trans_vel.cpu().numpy().flatten())
        
        if case_idx % 20 == 19:
            mae_w = float(np.mean(np.abs(np.concatenate(all_pred_w_afno) - np.concatenate(all_gt))))
            mae_wo = float(np.mean(np.abs(np.concatenate(all_pred_wo_afno) - np.concatenate(all_gt))))
            print(f"  [{case_idx+1}/{num_eval_cases}] Running AFNO MAE: w/ = {mae_w:.4f}, w/o = {mae_wo:.4f}")
            
    # Concatenate all data
    all_gt = np.concatenate(all_gt)
    all_pred_w_afno = np.concatenate(all_pred_w_afno)
    all_pred_wo_afno = np.concatenate(all_pred_wo_afno)
    all_pred_w_trans = np.concatenate(all_pred_w_trans)
    all_pred_wo_trans = np.concatenate(all_pred_wo_trans)
    
    print(f"\nFinal overall MAE (AFNO): w/ MF = {np.mean(np.abs(all_pred_w_afno - all_gt)):.5f}, w/o MF = {np.mean(np.abs(all_pred_wo_afno - all_gt)):.5f}")
    print(f"Final overall MAE (Transolver): w/ MF = {np.mean(np.abs(all_pred_w_trans - all_gt)):.5f}, w/o MF = {np.mean(np.abs(all_pred_wo_trans - all_gt)):.5f}")
    
    # Setup bins
    bin_resolution = 0.1
    bin_centers = np.arange(0.1, 5.8, bin_resolution)
    half_bin = bin_resolution / 2.0
    
    mean_w_afno_list = []
    mean_wo_afno_list = []
    mean_w_trans_list = []
    mean_wo_trans_list = []
    valid_bin_centers = []
    
    for c in bin_centers:
        mask = (all_gt >= c - half_bin) & (all_gt < c + half_bin)
        num_points = np.sum(mask)
        # Skip bins with too few points to be statistically sound
        if num_points < 20:
            continue
            
        valid_bin_centers.append(c)
        mean_w_afno_list.append(np.mean(all_pred_w_afno[mask]))
        mean_wo_afno_list.append(np.mean(all_pred_wo_afno[mask]))
        mean_w_trans_list.append(np.mean(all_pred_w_trans[mask]))
        mean_wo_trans_list.append(np.mean(all_pred_wo_trans[mask]))
        
    valid_bin_centers = np.array(valid_bin_centers)
    mean_w_afno_list = np.array(mean_w_afno_list)
    mean_wo_afno_list = np.array(mean_wo_afno_list)
    mean_w_trans_list = np.array(mean_w_trans_list)
    mean_wo_trans_list = np.array(mean_wo_trans_list)
    
    output_dir = PROJECT_DIR_T / "plots_paper"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(r"C:\Users\HJ000\.gemini\antigravity\brain\d559beff-90e2-4679-8d80-04bf042063a7")
    
    # Save AFNO calibration scatter
    fn_afno = output_dir / "ablation_afno_calibration_scatter.png"
    draw_mean_calibration_scatter(
        valid_bin_centers, mean_w_afno_list, mean_wo_afno_list,
        color_w='#ED7D31', color_wo='#5B9BD5',
        label_w='CATONet-AFNO (w/ MF)', label_wo='CATONet-AFNO (w/o MF)',
        filename=fn_afno
    )
    shutil.copy(fn_afno, artifacts_dir / "ablation_afno_calibration_scatter.png")
    print(f"Saved AFNO plot to {fn_afno}")
    
    # Save Transolver calibration scatter
    fn_trans = output_dir / "ablation_transolver_calibration_scatter.png"
    draw_mean_calibration_scatter(
        valid_bin_centers, mean_w_trans_list, mean_wo_trans_list,
        color_w='#E74C3C', color_wo='#70AD47',
        label_w='CATONet-Transolver (w/ MF)', label_wo='CATONet-Transolver (w/o MF)',
        filename=fn_trans
    )
    shutil.copy(fn_trans, artifacts_dir / "ablation_transolver_calibration_scatter.png")
    print(f"Saved Transolver plot to {fn_trans}")
    
    print("\nCalibration scatter plots generated successfully!")

if __name__ == '__main__':
    main()
