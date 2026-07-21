import argparse
from typing import List, Dict, Any, Optional
import yaml
import sys
import os
from pathlib import Path
import shutil
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model import FLRONetFNO, FLRONetAFNO, FLRONetUNet, FLRONetMLP, FNO3D, FLRONetTransolver, FNO, AFNO, Transolver, UNet
from cfd.dataset import CFDDataset
from common.training import CheckpointLoader
from worker import Trainer


def main(config: Dict[str, Any], checkpoint_path: str = '.checkpoints', logfile: Optional[str] = None) -> None:
    """
    Main function to train FLRONet.

    Parameters:
        config (Dict[str, Any]): Configuration dictionary.
    """

    # Parse CLI arguments:
    init_sensor_timeframes: List[int]           = list(config['dataset']['init_sensor_timeframes'])
    future_prediction_range: List[int] | None   = config['dataset'].get('future_prediction_range')
    n_fullstate_timeframes_per_chunk: int       = int(config['dataset']['n_fullstate_timeframes_per_chunk'])
    n_samplings_per_chunk: int                  = int(config['dataset']['n_samplings_per_chunk'])
    resolution: tuple                           = tuple(config['dataset']['resolution'])
    n_sensors: int                              = int(config['dataset']['n_sensors'])
    sensor_generator: str                       = str(config['dataset']['sensor_generator'])
    embedding_generator: str                    = str(config['dataset']['embedding_generator'])
    dropout_probabilities: List[float]          = list(config['dataset']['dropout_probabilities'])
    seed: int                                   = int(config['dataset']['seed'])
    write_to_disk: bool                         = bool(config['dataset'].get('write_to_disk', True))
    model_name: str                             = str(config['architecture']['momdel_name'])
    n_channels: int                             = int(config['architecture']['n_channels'])
    embedding_dim: int                          = int(config['architecture']['embedding_dim'])
    n_stacked_networks: int                     = int(config['architecture']['n_stacked_networks'])
    n_fno_layers: int                           = int(config['architecture']['n_fno_layers'])
    n_hmodes: int                               = int(config['architecture']['n_hmodes'])
    n_wmodes: int                               = int(config['architecture']['n_wmodes'])
    n_tmodes: int                               = int(config['architecture']['n_tmodes'])
    # Transolver-specific configs:
    n_trans_layers: int                         = int(config['architecture'].get('n_trans_layers', 4))
    n_trans_hidden: int                         = int(config['architecture'].get('n_trans_hidden', 128))
    n_trans_head: int                           = int(config['architecture'].get('n_trans_head', 8))
    slice_num: int                              = int(config['architecture'].get('slice_num', 32))
    trans_dropout: float                        = float(config['architecture'].get('trans_dropout', 0.0))
    blur_kernel_size: int                       = int(config['architecture'].get('blur_kernel_size', 0))
    blur_sigma: float                           = float(config['architecture'].get('blur_sigma', 2.0))
    is_TC: bool                                 = bool(config['architecture'].get('is_TC', True))
    is_cross_attn: bool                         = bool(config['architecture'].get('is_cross_attn', False))
    _umf = config['architecture'].get('use_mean_field', 'operator')
    use_mean_field: str                         = ('operator' if _umf is True else 'none' if _umf is False else str(_umf))
    mean_field_hidden: int                      = int(config['architecture'].get('mean_field_hidden', 32))
    mean_field_time_embed_dim: int              = int(config['architecture'].get('mean_field_time_embed_dim', 32))

    from_checkpoint: Optional[str]              = config['training']['from_checkpoint']
    train_batch_size: int                       = int(config['training']['train_batch_size'])
    val_batch_size: int                         = int(config['training']['val_batch_size'])
    learning_rate: float                        = float(config['training']['learning_rate'])
    n_epochs: int                               = int(config['training']['n_epochs'])
    patience: int                               = int(config['training']['patience'])
    tolerance: int                              = float(config['training']['tolerance'])
    save_frequency: int                         = int(config['training']['save_frequency'])
    freeze_branchnets: bool                     = bool(config['training']['freeze_branchnets'])
    freeze_trunknets: bool                      = bool(config['training']['freeze_trunknets'])
    freeze_bias: bool                           = bool(config['training']['freeze_bias'])

    # Dataset
    if model_name.lower() == 'fno3d':
        n_fullstate_timeframes_per_chunk = len(init_sensor_timeframes)
        init_fullstate_timeframes = init_sensor_timeframes
    else:
        init_fullstate_timeframes = None

    train_dataset = CFDDataset(
        root='./data/train', 
        init_sensor_timeframes=init_sensor_timeframes, 
        future_prediction_range=future_prediction_range,
        n_fullstate_timeframes_per_chunk=n_fullstate_timeframes_per_chunk,
        n_samplings_per_chunk=n_samplings_per_chunk,
        resolution=resolution,
        n_sensors=n_sensors,
        dropout_probabilities=dropout_probabilities,
        noise_level=0.,
        sensor_generator=sensor_generator, 
        embedding_generator=embedding_generator,
        init_fullstate_timeframes=init_fullstate_timeframes,
        seed=seed,
        write_to_disk=write_to_disk,
    )
    val_dataset = CFDDataset(
        root='./data/val', 
        init_sensor_timeframes=init_sensor_timeframes,
        future_prediction_range=future_prediction_range,
        n_fullstate_timeframes_per_chunk=n_fullstate_timeframes_per_chunk,
        n_samplings_per_chunk=n_samplings_per_chunk,
        resolution=resolution,
        n_sensors=n_sensors,
        dropout_probabilities=dropout_probabilities,
        noise_level=0.,
        sensor_generator=sensor_generator, 
        embedding_generator=embedding_generator,
        init_fullstate_timeframes=init_fullstate_timeframes,
        seed=seed,
        write_to_disk=write_to_disk,
    )

    if model_name.lower() == 'flronet-fno':
        # Model
        if from_checkpoint is not None:
            checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
            net: FLRONetFNO = checkpoint_loader.load(scope=globals()).cuda()    # ignore optimizer
            assert isinstance(net, FLRONetFNO)
        else:
            net = FLRONetFNO(
                n_channels=n_channels, n_fno_layers=n_fno_layers,
                n_hmodes=n_hmodes, n_wmodes=n_wmodes, embedding_dim=embedding_dim,
                n_stacked_networks=n_stacked_networks,
                is_TC=is_TC,
                is_cross_attn=is_cross_attn,
                use_mean_field=use_mean_field,
                mean_field_hidden=mean_field_hidden,
                mean_field_time_embed_dim=mean_field_time_embed_dim,
            ).cuda()

    elif model_name.lower() == 'flronet-afno':
        # Model
        if from_checkpoint is not None:
            checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
            net: FLRONetAFNO = checkpoint_loader.load(scope=globals()).cuda()
            assert isinstance(net, FLRONetAFNO)
        else:
            net = FLRONetAFNO(
                n_channels=n_channels, n_fno_layers=n_fno_layers, embedding_dim=embedding_dim,
                n_stacked_networks=n_stacked_networks,
                resolution=resolution,
                is_cross_attn=is_cross_attn,
                use_mean_field=use_mean_field,
                mean_field_hidden=mean_field_hidden,
                mean_field_time_embed_dim=mean_field_time_embed_dim,
            ).cuda()

    elif model_name.lower() == 'flronet-unet':
        # Model
        if from_checkpoint is not None:
            checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
            net: FLRONetUNet = checkpoint_loader.load(scope=globals()).cuda()
            assert isinstance(net, FLRONetUNet)
        else:
            net = FLRONetUNet(
                n_channels=n_channels, embedding_dim=embedding_dim, n_stacked_networks=n_stacked_networks,
                is_cross_attn=is_cross_attn,
                use_mean_field=use_mean_field,
                mean_field_hidden=mean_field_hidden,
                mean_field_time_embed_dim=mean_field_time_embed_dim,
            ).cuda()
    
    elif model_name.lower() == 'flronet-mlp':
        # Model
        if from_checkpoint is not None:
            checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
            net: FLRONetMLP = checkpoint_loader.load(scope=globals()).cuda()
            assert isinstance(net, FLRONetMLP)
        else:
            net = FLRONetMLP(
                n_channels=n_channels, embedding_dim=embedding_dim, n_sensors=n_sensors,
                resolution=resolution, n_stacked_networks=n_stacked_networks,
                is_cross_attn=is_cross_attn,
            ).cuda()

    elif model_name.lower() == 'flronet-transolver':
        # Model
        if from_checkpoint is not None:
            checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
            net: FLRONetTransolver = checkpoint_loader.load(scope=globals()).cuda()
            assert isinstance(net, FLRONetTransolver)
        else:
            net = FLRONetTransolver(
                n_channels=n_channels,
                n_layers=n_trans_layers,
                n_hidden=n_trans_hidden,
                n_head=n_trans_head,
                embedding_dim=embedding_dim,
                n_stacked_networks=n_stacked_networks,
                resolution=resolution,
                n_timeframes=len(init_sensor_timeframes),
                slice_num=slice_num,
                dropout=trans_dropout,
                is_cross_attn=is_cross_attn,
                use_mean_field=use_mean_field,
                mean_field_hidden=mean_field_hidden,
                mean_field_time_embed_dim=mean_field_time_embed_dim,
            ).cuda()

    elif model_name.lower() == 'unet':
        # Model
        if from_checkpoint is not None:
            checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
            net: UNet = checkpoint_loader.load(scope=globals()).cuda()
            assert isinstance(net, UNet)
        else:
            net = UNet(
                n_channels=n_channels,
                embedding_dim=embedding_dim,
                n_timeframes=len(init_sensor_timeframes),
            ).cuda()

    elif model_name.lower() == 'afno':
        # Model
        if from_checkpoint is not None:
            checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
            net: AFNO = checkpoint_loader.load(scope=globals()).cuda()
            assert isinstance(net, AFNO)
        else:
            net = AFNO(
                n_channels=n_channels, n_fno_layers=n_fno_layers, 
                embedding_dim=embedding_dim, resolution=resolution,
                n_timeframes=len(init_sensor_timeframes),
            ).cuda()

    elif model_name.lower() == 'transolver':
        # Model
        if from_checkpoint is not None:
            checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
            net: Transolver = checkpoint_loader.load(scope=globals()).cuda()
            assert isinstance(net, Transolver)
        else:
            net = Transolver(
                n_channels=n_channels, 
                n_layers=n_trans_layers, 
                n_hidden=n_trans_hidden, 
                n_head=n_trans_head,
                resolution=resolution,
                n_timeframes=len(init_sensor_timeframes),
                slice_num=slice_num,
                dropout=trans_dropout
            ).cuda()

    elif model_name.lower() == 'fno':
        # Model
        if from_checkpoint is not None:
            checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
            net: FNO = checkpoint_loader.load(scope=globals()).cuda()
            assert isinstance(net, FNO)
        else:
            net = FNO(
                n_channels=n_channels, n_fno_layers=n_fno_layers, 
                n_hmodes=n_hmodes, n_wmodes=n_wmodes, embedding_dim=embedding_dim,
                n_timeframes=len(init_sensor_timeframes),
            ).cuda()

    elif model_name.lower() == 'fno3d':
        # Model
        if from_checkpoint is not None:
            checkpoint_loader = CheckpointLoader(checkpoint_path=from_checkpoint)
            net: FNO3D = checkpoint_loader.load(scope=globals()).cuda()
            assert isinstance(net, FNO3D)
        else:
            net = FNO3D(
                n_channels=n_channels, n_fno_layers=n_fno_layers, 
                n_hmodes=n_hmodes, n_wmodes=n_wmodes, n_tmodes=n_tmodes, embedding_dim=embedding_dim,
            ).cuda()

    else:
        raise ValueError(f'Invalid model_name {model_name}')
    
    if model_name.lower().startswith('flronet'):
        if freeze_branchnets:
            print('Freezed BranchNets')
            net.freeze_branchnets()
        if freeze_trunknets:
            print('Freezed TrunkNets')
            net.freeze_trunknets()
        if freeze_bias:
            if net.mean_field_net is not None:
                # Dynamic mean-field mode: freeze MeanFieldNet instead of scalar bias
                for param in net.mean_field_net.parameters():
                    param.requires_grad = False
                print('Freezed MeanFieldNet (dynamic bias)')
            else:
                net.freeze_bias()
                print('Freezed Bias')

    trainer = Trainer(
        net=net, 
        lr=learning_rate,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
    )
    trainer.train(
        n_epochs=n_epochs, 
        patience=patience,
        tolerance=tolerance, 
        checkpoint_path=checkpoint_path,
        logfile=logfile,
        save_frequency=save_frequency,
    )
    
def _load_config(config_path: Path) -> Dict[str, Any]:
    with open(file=str(config_path), mode='r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def _cleanup_tensors() -> None:
    tensors_dir = Path(ROOT_DIR) / 'tensors'
    if tensors_dir.is_dir():
        shutil.rmtree(tensors_dir)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _resolve_config_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (Path(ROOT_DIR) / path).resolve()

def _iter_yaml_files(config_dir: Path) -> List[Path]:
    yaml_files = list(config_dir.glob('*.yaml')) + list(config_dir.glob('*.yml'))
    return sorted(yaml_files, key=lambda p: p.name)


if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, default=None, help='Configuration file name.')
    parser.add_argument('--config-dir', type=str, default='yaml_T_A', help='Directory that contains YAML configs for batch training.')
    args: argparse.Namespace = parser.parse_args()

    if args.config is not None:
        config_path = _resolve_config_path(args.config)
        config = _load_config(config_path)
        checkpoint_dir = f'.checkpoints_{config_path.stem}'
        logfile = str(Path('.logs') / config_path.stem)
        main(config=config, checkpoint_path=checkpoint_dir, logfile=logfile)
    else:
        config_dir = _resolve_config_path(args.config_dir)
        yaml_files = _iter_yaml_files(config_dir)
        if not yaml_files:
            raise FileNotFoundError(f'No YAML files found in {config_dir}')

        print(f'Found {len(yaml_files)} config(s) in {config_dir}')
        for idx, config_path in enumerate(yaml_files, start=1):
            print(f'[{idx}/{len(yaml_files)}] Training with {config_path.name}')
            _cleanup_tensors()
            config = _load_config(config_path)
            config.setdefault('dataset', {})
            config['dataset']['write_to_disk'] = True
            checkpoint_dir = f'.checkpoints_{config_path.stem}'
            logfile = str(Path('.logs') / config_path.stem)
            main(config=config, checkpoint_path=checkpoint_dir, logfile=logfile)
