"""
Atmospheric Ingestion Ablation Pipeline (Single vs. Dual vs. Triple vs. Quad)
=============================================================================
Trains and evaluates Sobolev-PINN models across the 4 Ingestion Paradigms:
  1. Single-Input (1B): [ L_surf ]
  2. Dual-Input (2B):   [ L_surf , L_down ]
  3. Triple-Input (3B): [ L_surf , tau , L_down ]
  4. Quad-Input (4B):   [ L_sensor , tau , L_up , L_down ]

Migrated to Stage 1 Grounded Framework:
  - Band masking (223 clean HyTES bands).
  - Training-fold prior head initialization (mu_t, mu_e).
  - Training-fold input radiance standardization (mu_x, std_x).
  - Unit-free balanced multi-task loss with physical Sobolev scaling.
  - Comprehensive checkpoint saving with normalization statistics.

Author: Dr. Atilio Grondona & OpenTIR-PINN Research Team
Target Journal: IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.pinn import PhysicsInformedNN
from src.models.losses import SpectralShapeLoss
from src.config import SENSOR_CONFIG
from src.training.input_utils import build_atmospheric_band_cache, assemble_input_tensor, compute_spectral_angle_mapper
from src.training.stage1_core import (
    get_sensor_wavelengths,
    get_sensor_band_mask,
    extract_training_priors,
    compute_input_norm_stats,
    standardize_input,
    get_stage1_default_config,
    build_stage1_pinn,
    build_stage1_criterion,
    compute_unified_metrics,
    compute_like_for_like_null_rmse
)


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    mode: str,
    atm_cache: dict,
    wav_tensor: torch.Tensor,
    mu_x: torch.Tensor,
    std_x: torch.Tensor,
    mask: torch.Tensor,
    null_rmse_e: float,
    std_t: float,
    std_e: float,
    device: torch.device
) -> dict:
    model.eval()
    all_t_pred = []
    all_t_true = []
    all_e_pred = []
    all_e_true = []
    val_loss_sum = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            x_in = assemble_input_tensor(batch, mode, atm_cache, device, mask=mask)
            x_norm = standardize_input(x_in, mu_x, std_x)

            l_surf = batch['input_radiance'].to(device)[:, mask]
            l_down = batch['downwelling_radiance'].to(device)[:, mask]
            t_true = batch['target_temp'].to(device)
            e_true = batch['target_emissivity'].to(device)[:, mask]
            bsz = x_in.shape[0]

            t_pred, e_pred = model(x_norm)
            loss = criterion(
                t_pred=t_pred,
                t_true=t_true,
                eps_pred=e_pred,
                eps_true=e_true,
                l_surf_obs=l_surf,
                l_down=l_down,
                wavelengths=wav_tensor
            )

            val_loss_sum += loss.item() * bsz
            all_t_pred.append(t_pred)
            all_t_true.append(t_true)
            all_e_pred.append(e_pred)
            all_e_true.append(e_true)
            n_samples += bsz

    t_pred_cat = torch.cat(all_t_pred, dim=0)
    t_true_cat = torch.cat(all_t_true, dim=0)
    e_pred_cat = torch.cat(all_e_pred, dim=0)
    e_true_cat = torch.cat(all_e_true, dim=0)

    metrics = compute_unified_metrics(
        t_pred=t_pred_cat,
        t_true=t_true_cat,
        e_pred=e_pred_cat,
        e_true=e_true_cat,
        null_rmse_e=null_rmse_e,
        std_t=std_t,
        std_e=std_e
    )
    metrics['val_loss'] = val_loss_sum / n_samples
    metrics['n_samples'] = n_samples
    return metrics


def train_single_mode(
    sensor: str,
    mode: str,
    epochs: int = 10,
    batch_size: int = 32,
    device: Optional[torch.device] = None
) -> dict:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*75}")
    print(f"Training Ablation Mode: [{mode.upper()}] | Sensor: [{sensor.upper()}] | Epochs: {epochs}")
    print(f"{'='*75}")

    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets')
    train_path = os.path.join(data_dir, f'train_fold0_{sensor}.pt')
    val_path = os.path.join(data_dir, f'val_fold0_{sensor}.pt')

    train_data = torch.load(train_path, map_location='cpu', weights_only=False)
    val_data = torch.load(val_path, map_location='cpu', weights_only=False)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    atm_cache = build_atmospheric_band_cache(sensor)
    wav_tensor = get_sensor_wavelengths(sensor, device)
    mask = get_sensor_band_mask(sensor, train_data, device=device)
    n_bands_eff = int(mask.sum())
    wav_eff = wav_tensor[mask]

    # Training Priors
    priors = extract_training_priors(train_data, mask, device=device)
    mu_t, std_t, var_t = priors['mu_t'], priors['std_t'], priors['var_t']
    mu_e, std_e, var_e = priors['mu_e'], priors['std_e'], priors['var_e']

    # Null baseline
    null_rmse_e = compute_like_for_like_null_rmse(val_data, mu_e, mask)

    # Input Standardization
    mu_x, std_x = compute_input_norm_stats(train_data, mode, atm_cache, mask, device=device)

    # Config & Models
    config = get_stage1_default_config(sensor)
    config['input_mode'] = mode

    model = build_stage1_pinn(sensor, n_bands_eff, config, mu_t, mu_e, device=device)
    criterion = build_stage1_criterion(sensor, config, var_t, var_e, device=device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_score = float('inf')
    best_state = None
    start_time = time.time()
    w_phys_target = float(config.get('w_phys', 0.05))

    for epoch in range(epochs):
        criterion.w_phys = 0.0 if epoch == 0 else w_phys_target

        model.train()
        train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            x_in = assemble_input_tensor(batch, mode, atm_cache, device, mask=mask)
            x_norm = standardize_input(x_in, mu_x, std_x)

            l_surf = batch['input_radiance'].to(device)[:, mask]
            l_down = batch['downwelling_radiance'].to(device)[:, mask]
            t_true = batch['target_temp'].to(device)
            e_true = batch['target_emissivity'].to(device)[:, mask]

            optimizer.zero_grad()
            t_pred, e_pred = model(x_norm)

            loss = criterion(
                t_pred=t_pred,
                t_true=t_true,
                eps_pred=e_pred,
                eps_true=e_true,
                l_surf_obs=l_surf,
                l_down=l_down,
                wavelengths=wav_eff
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * x_in.shape[0]
            n_train += x_in.shape[0]

        val_metrics = evaluate_model(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            mode=mode,
            atm_cache=atm_cache,
            wav_tensor=wav_eff,
            mu_x=mu_x,
            std_x=std_x,
            mask=mask,
            null_rmse_e=null_rmse_e,
            std_t=std_t,
            std_e=std_e,
            device=device
        )
        scheduler.step(val_metrics['composite_score'])

        if val_metrics['composite_score'] < best_val_score:
            best_val_score = val_metrics['composite_score']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | Train Loss: {train_loss/n_train:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | dT MAE: {val_metrics['temp_mae']:.2f} K | eps RMSE: {val_metrics['eps_rmse']:.4f} (Gain: {val_metrics['eps_gain_over_null_pct']:.1f}%) | SAM: {val_metrics['sam_deg']:.2f} deg")

    elapsed = time.time() - start_time

    # Load best checkpoint and evaluate final metrics
    model.load_state_dict(best_state)
    final_metrics = evaluate_model(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        mode=mode,
        atm_cache=atm_cache,
        wav_tensor=wav_eff,
        mu_x=mu_x,
        std_x=std_x,
        mask=mask,
        null_rmse_e=null_rmse_e,
        std_t=std_t,
        std_e=std_e,
        device=device
    )

    # Save organized checkpoint
    models_dir = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'ablation_ingestion', mode, sensor)
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, f'model_{sensor}_{mode}_best.pth')

    torch.save({
        'sensor': sensor,
        'mode': mode,
        'epochs': epochs,
        'config': config,
        'model_state_dict': best_state,
        'norm_stats': {
            'mu_x': mu_x.cpu(),
            'std_x': std_x.cpu(),
            'mu_t': mu_t,
            'std_t': std_t,
            'var_t': var_t,
            'mu_e': mu_e.cpu(),
            'std_e': std_e,
            'var_e': var_e
        },
        'band_mask': mask.cpu(),
        'val_metrics': final_metrics
    }, checkpoint_path)

    print(f"Saved best model checkpoint to: {checkpoint_path}")
    final_metrics['mode'] = mode
    final_metrics['wall_clock_s'] = elapsed
    return final_metrics


def run_full_ingestion_ablation(sensor: str = 'aster', epochs: int = 10):
    modes = ['single', 'dual', 'triple', 'quad']
    records = []
    for mode in modes:
        res = train_single_mode(sensor=sensor, mode=mode, epochs=epochs)
        records.append(res)

    df_res = pd.DataFrame(records)
    benchmarks_dir = os.path.join(PROJECT_ROOT, 'outputs', 'benchmarks')
    os.makedirs(benchmarks_dir, exist_ok=True)
    out_csv = os.path.join(benchmarks_dir, f'ablation_ingestion_benchmark_{sensor}.csv')
    df_res.to_csv(out_csv, index=False)

    print(f"\n{'='*75}")
    print(f"Atmospheric Ingestion Ablation Benchmark Summary ({sensor.upper()}):")
    print(f"{'='*75}")
    print(df_res.to_string(index=False))
    print(f"\nSaved benchmark results to: {out_csv}")
    return df_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Atmospheric Ingestion Ablation (Stage 1)")
    parser.add_argument('--sensor', type=str, default='aster', choices=['aster', 'hytes'], help='Sensor')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs per mode')
    parser.add_argument('--mode', type=str, default=None, choices=['single', 'dual', 'triple', 'quad'], help='Single mode run')
    args = parser.parse_args()

    if args.mode:
        train_single_mode(sensor=args.sensor, mode=args.mode, epochs=args.epochs)
    else:
        run_full_ingestion_ablation(sensor=args.sensor, epochs=args.epochs)
