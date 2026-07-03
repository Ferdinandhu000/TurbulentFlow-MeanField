import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

# Ensure the project root is on PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model import FLRONetFNO, FLRONetAFNO, FLRONetUNet, FLRONetMLP, FNO3D, FLRONetTransolver, FNO, AFNO, Transolver, UNet
from cfd.dataset import CFDDataset
from common.training import CheckpointLoader

# Hardcoded correct checkpoint mapping to prevent YAML file errors
CHECKPOINT_MAPPING = {
    "inside": {
        "fno":             "checkpoints_best\\.checkpoints_01-T-FNO_inside_config\\fno7.pt",
        "afno":            "checkpoints_best\\.checkpoints_02-T-AFNO_inside_config\\afno17.pt",
        "transolver":      "checkpoints_best\\.checkpoints_03-T-Transolver_inside_config\\transolver90.pt",
        "flronet":         "checkpoints_best\\.checkpoints_04-T-FLRONet_inside_config\\flronetfno6.pt",
        "CATO-afno":       "checkpoints_best\\.checkpoints_05-T-mean-operator_CATO_afno_inside_config\\flronetafno16.pt",
        "CATO-transolver": "checkpoints_best\\.checkpoints_06-T-mean-operator_CATO_trans_inside_config\\flronettransolver86.pt"
    },
    "outside": {
        "fno":             "checkpoints_best\\.checkpoints_11-T-FNO_outside_config\\fno3.pt",
        "afno":            "checkpoints_best\\.checkpoints_12-T-AFNO_outside_config\\afno13.pt",
        "transolver":      "checkpoints_best\\.checkpoints_13-T-Transolver_outside_config\\transolver21.pt",
        "flronet":         "checkpoints_best\\.checkpoints_14-T-FLRONet_outside_config\\flronetfno3.pt",
        "CATO-afno":       "checkpoints_best\\.checkpoints_15-T-mean-operator_CATO_afno_outside_config\\flronetafno7.pt",
        "CATO-transolver": "checkpoints_best\\.checkpoints_16-T-mean-operator_CATO_trans_outside_config\\flronettransolver32.pt"
    }
}

def evaluate_model_for_frames(config_path: str, mode: str, target_frames: list, model_name: str, do_write: bool):
    print(f"\n[+] Starting evaluation: Mode={mode}, Model={model_name}, Frames={target_frames}, write_to_disk={do_write}")
    
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # Override checkpoint path
    from_checkpoint = CHECKPOINT_MAPPING[mode][model_name]
    print(f"  [Use Checkpoint] {from_checkpoint}")
    
    # Override configuration parameters
    config['dataset']['write_to_disk'] = do_write
    config['dataset']['n_fullstate_timeframes_per_chunk'] = len(target_frames)
    config['evaluate']['init_fullstate_timeframes'] = target_frames
    
    # Extract dataset options
    init_sensor_timeframes = list(config['dataset']['init_sensor_timeframes'])
    future_prediction_range = config['dataset'].get('future_prediction_range')
    n_fullstate_timeframes_per_chunk = int(config['dataset']['n_fullstate_timeframes_per_chunk'])
    n_samplings_per_chunk = int(config['dataset']['n_samplings_per_chunk'])
    resolution = tuple(config['dataset']['resolution'])
    n_sensors = int(config['dataset']['n_sensors'])
    sensor_generator = str(config['dataset']['sensor_generator'])
    embedding_generator = str(config['dataset']['embedding_generator'])
    is_load_sensor_pos = bool(config['evaluate'].get('is_load_sensor_pos', True))
    sensor_position_path = str(config['evaluate'].get('sensor_position_path', 'sensor_position_pt/pos_seed1.pt'))
    seed = int(config['dataset'].get('seed', 0))
    n_dropout_sensors = int(config['evaluate']['n_dropout_sensors'])
    noise_level = float(config['evaluate']['noise_level'])
    
    # Load the model weights
    checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
    net = checkpoint_loader.load(scope=globals())
    net = net.cuda()
    net.eval()
    
    # Setup dropout probability list
    if n_dropout_sensors == 0:
        implied_dropout_probabilities = []
    else:
        implied_dropout_probabilities = [0.] * n_dropout_sensors
        implied_dropout_probabilities[-1] = 1.
        
    # Instantiate the datasets
    dataset = CFDDataset(
        root='./data/test', 
        init_sensor_timeframes=init_sensor_timeframes,
        future_prediction_range=future_prediction_range,
        n_fullstate_timeframes_per_chunk=n_fullstate_timeframes_per_chunk,
        n_samplings_per_chunk=n_samplings_per_chunk,
        resolution=resolution,
        n_sensors=n_sensors,
        dropout_probabilities=implied_dropout_probabilities,
        noise_level=noise_level,
        sensor_generator=sensor_generator, 
        embedding_generator=embedding_generator,
        init_fullstate_timeframes=target_frames,
        seed=seed,
        write_to_disk=do_write,
        sensor_position_path=sensor_position_path if is_load_sensor_pos else None,
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = next(net.parameters()).device
    
    # Setup metrics storage per frame
    num_frames = len(target_frames)
    rmse_per_frame = [[] for _ in range(num_frames)]
    mae_per_frame = [[] for _ in range(num_frames)]
    l2_per_frame = [[] for _ in range(num_frames)]
    
    rmse_fn = nn.MSELoss(reduction='sum')
    mae_fn = nn.L1Loss(reduction='sum')
    
    with torch.no_grad():
        for sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames, case_names, sampling_ids in tqdm(dataloader, desc="Evaluating"):
            sensor_timeframes = sensor_timeframes.to(device)
            sensor_frames = sensor_frames.to(device)
            fullstate_timeframes = fullstate_timeframes.to(device)
            fullstate_frames = fullstate_frames.to(device)
            
            if isinstance(net, (FLRONetFNO, FLRONetAFNO, FLRONetMLP, FLRONetUNet, FLRONetTransolver, FNO, AFNO, Transolver, UNet)):
                preds = net(sensor_timeframes, sensor_frames, fullstate_timeframes, None)
            else:
                preds = net(
                    sensor_frames, 
                    out_resolution=(sensor_timeframes.max().item() - sensor_timeframes.min().item() + 1, *dataset.resolution),
                )
            
            reconstruction_frames = preds.squeeze(dim=0)
            fullstate_frames = fullstate_frames.squeeze(dim=0)
            fullstate_timeframes = fullstate_timeframes.squeeze(dim=0)
            
            for frame_idx, timeframe in enumerate(fullstate_timeframes):
                reconstruction_frame = reconstruction_frames[frame_idx]
                fullstate_frame = fullstate_frames[frame_idx]
                
                frame_total_mse = rmse_fn(
                    reconstruction_frame.unsqueeze(0).unsqueeze(0), 
                    fullstate_frame.unsqueeze(0).unsqueeze(0)
                )
                frame_mean_rmse = (frame_total_mse.item() / fullstate_frame.numel()) ** 0.5
                
                frame_total_mae = mae_fn(
                    reconstruction_frame.unsqueeze(0).unsqueeze(0), 
                    fullstate_frame.unsqueeze(0).unsqueeze(0)
                )
                frame_mean_mae = frame_total_mae.item() / fullstate_frame.numel()
                
                L2_numerator = frame_total_mse.item() ** 0.5
                L2_denominator = torch.linalg.norm(fullstate_frame).item()
                frame_mean_L2 = L2_numerator / L2_denominator
                
                rmse_per_frame[frame_idx].append(frame_mean_rmse)
                mae_per_frame[frame_idx].append(frame_mean_mae)
                l2_per_frame[frame_idx].append(frame_mean_L2)
                
    # Write to CSV files for each frame separately
    for frame_idx, actual_frame in enumerate(target_frames):
        # Create output directory: csv_inside/csv_inside_{frame} or csv_outside/csv_outside_{frame}
        out_dir = os.path.join(f"csv_{mode}", f"csv_{mode}_{actual_frame}")
        os.makedirs(out_dir, exist_ok=True)
        
        csv_path = os.path.join(out_dir, f"{model_name}_{mode}.csv")
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_index', 'rmse', 'mae', 'l2_loss'])
            for i, (r, m, l) in enumerate(zip(rmse_per_frame[frame_idx], mae_per_frame[frame_idx], l2_per_frame[frame_idx])):
                writer.writerow([i, r, m, l])
                
        print(f"  [->] Saved {csv_path} with {len(rmse_per_frame[frame_idx])} rows.")

if __name__ == "__main__":
    # Models to run and their configuration mappings
    model_mappings = [
        ('fno.yaml', 'fno'),
        ('afno.yaml', 'afno'),
        ('transolver.yaml', 'transolver'),
        ('flronet.yaml', 'flronet'),
        ('CATO-afno.yaml', 'CATO-afno'),
        ('CATO-trans.yaml', 'CATO-transolver'),  # Output matches the existing CATO-transolver naming
    ]
    
    # 1. Evaluate INSIDE (0-20 frames)
    inside_frames = list(range(21))  # [0, 1, 2, ..., 20]
    print("=================== STARTING INSIDE EVALUATIONS ===================")
    for idx, (yaml_file, model_name) in enumerate(model_mappings):
        cfg_path = os.path.join("evaluate_yaml_inside", yaml_file)
        # Only the first model (idx == 0) writes tensors to disk
        do_write = (idx == 0)
        evaluate_model_for_frames(cfg_path, "inside", inside_frames, model_name, do_write)
        
    # 2. Evaluate OUTSIDE (20-30 frames)
    outside_frames = list(range(20, 31))  # [20, 21, 22, ..., 30]
    print("\n=================== STARTING OUTSIDE EVALUATIONS ===================")
    for idx, (yaml_file, model_name) in enumerate(model_mappings):
        cfg_path = os.path.join("evaluate_yaml_outside", yaml_file)
        # Only the first model (idx == 0) writes tensors to disk
        do_write = (idx == 0)
        evaluate_model_for_frames(cfg_path, "outside", outside_frames, model_name, do_write)
        
    print("\n[***] All evaluations completed successfully!")
