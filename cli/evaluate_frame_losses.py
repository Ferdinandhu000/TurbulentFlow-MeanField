import argparse
import os
import sys
import csv
from typing import List, Tuple, Dict, Any
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure the project root is on PYTHONPATH when running as a script
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model import FLRONetFNO, FLRONetAFNO, FLRONetUNet, FLRONetMLP, FNO3D, FLRONetTransolver, FNO, AFNO, Transolver, UNet
from cfd.dataset import CFDDataset
from common.training import CheckpointLoader


def save_frame_losses(rmse_list: List[float], mae_list: List[float], l2_list: List[float], output_dir: str = "plots") -> None:
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'frame_losses.csv')
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_index', 'rmse', 'mae', 'l2_loss'])
        for i, (r, m, l) in enumerate(zip(rmse_list, mae_list, l2_list)):
            writer.writerow([i, r, m, l])
    print(f"\n[*] Saved frame-by-frame losses to {csv_path}")


def main(config: Dict[str, Any]) -> None:
    # Parse CLI arguments:
    init_sensor_timeframes: List[int]           = list(config['dataset']['init_sensor_timeframes'])
    future_prediction_range: List[int] | None   = config['dataset'].get('future_prediction_range')
    n_fullstate_timeframes_per_chunk: int       = int(config['dataset']['n_fullstate_timeframes_per_chunk'])
    n_samplings_per_chunk: int                  = int(config['dataset']['n_samplings_per_chunk'])
    resolution: tuple                           = tuple(config['dataset']['resolution'])
    n_sensors: int                              = int(config['dataset']['n_sensors'])
    sensor_generator: str                       = str(config['dataset']['sensor_generator'])
    embedding_generator: str                    = str(config['dataset']['embedding_generator'])
    is_load_sensor_pos: bool                    = bool(config['evaluate'].get('is_load_sensor_pos', True))
    sensor_position_path: str                   = str(config['evaluate'].get('sensor_position_path', 'sensor_position_pt/pos.pt'))

    if is_load_sensor_pos:
        seed: int = int(config['dataset'].get('seed', 0))
        print(f"\n[*] evaluate_frame_losses.py: Using fixed sensor positions from {sensor_position_path}.")
    else:
        import random
        seed = random.randint(0, 1000000)
        print(f"\n[*] evaluate_frame_losses.py: Forcing random seed {seed} to randomly select Test sensors.")

    write_to_disk: bool                         = True
    n_dropout_sensors: int                      = int(config['evaluate']['n_dropout_sensors'])
    noise_level: float                          = float(config['evaluate']['noise_level'])
    init_fullstate_timeframes: List[int] | None  = config['evaluate']['init_fullstate_timeframes']
    from_checkpoint: str                        = str(config['evaluate']['from_checkpoint'])

    # Load the model
    checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
    net = checkpoint_loader.load(scope=globals())
    net = net.cuda()
    net.eval()

    if isinstance(net, FNO3D):
        init_fullstate_timeframes = list(range(min(init_sensor_timeframes), max(init_sensor_timeframes) + 1))
        n_fullstate_timeframes_per_chunk = len(init_fullstate_timeframes)

    # Instantiate the datasets
    if n_dropout_sensors == 0:
        implied_dropout_probabilities: List[float] = []
    else:
        implied_dropout_probabilities = [0.] * n_dropout_sensors
        implied_dropout_probabilities[-1] = 1.

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
        init_fullstate_timeframes=init_fullstate_timeframes,
        seed=seed,
        write_to_disk=write_to_disk,
        sensor_position_path=sensor_position_path if is_load_sensor_pos else None,
    )
    
    print(f'Using checkpoint: {from_checkpoint}')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = next(net.parameters()).device
    
    rmse_values: List[float] = []
    mae_values: List[float] = []
    L2Loss_values: List[float] = []
    
    rmse_fn = nn.MSELoss(reduction='sum')
    mae_fn = nn.L1Loss(reduction='sum')
    
    with torch.no_grad():
        for sensor_timeframes, sensor_frames, fullstate_timeframes, fullstate_frames, case_names, sampling_ids in tqdm(dataloader, desc="Evaluating frame losses"):
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
                frame_mean_mse = frame_total_mse.item() / fullstate_frame.numel()
                frame_mean_rmse = frame_mean_mse ** 0.5
                
                frame_total_mae = mae_fn(
                    reconstruction_frame.unsqueeze(0).unsqueeze(0), 
                    fullstate_frame.unsqueeze(0).unsqueeze(0)
                )
                frame_mean_mae = frame_total_mae.item() / fullstate_frame.numel()
                
                L2_numerator = frame_total_mse.item() ** 0.5
                L2_denominator = torch.linalg.norm(fullstate_frame).item()
                frame_mean_L2 = L2_numerator / L2_denominator

                rmse_values.append(frame_mean_rmse)
                mae_values.append(frame_mean_mae)
                L2Loss_values.append(frame_mean_L2)

    avg_rmse = sum(rmse_values) / len(rmse_values)
    avg_mae = sum(mae_values) / len(mae_values)
    avg_l2 = sum(L2Loss_values) / len(L2Loss_values)
    
    print(f"\nEvaluation Averages:")
    print(f"RMSE:     {avg_rmse:.6f}")
    print(f"MAE:      {avg_mae:.6f}")
    print(f"L2 Loss:  {avg_l2:.6f}")
    
    # Save the frame metrics to CSV
    save_frame_losses(rmse_values, mae_values, L2Loss_values, output_dir="plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Configuration file name.')
    args = parser.parse_args()
    
    with open(file=args.config, mode='r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    main(config)
