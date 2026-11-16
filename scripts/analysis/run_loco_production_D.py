"""
Leave-One-Class-Out (LOCO) Production Benchmark - Phase D (Audited)
==================================================================
Executes a fully audited, statistically rigorous LOCO evaluation using:
  1. The tuned Run D hyperparameters (SiLU, asymmetric depths, optimal Sobolev weights).
  2. The ratified class validity rule: only classes with >= 5 similarity groups
     (botanical, soil, sandstone, quartz).
  3. Group-aware 85/15 split on N-1 classes to prevent spectral leakage in checkpoint selection.
  4. 100 epochs with ReduceLROnPlateau and gradient clipping (1.0).
  5. Paired Classical TES benchmark on the exact same held-out test spectra.

Author: Dr. Atilio Grondona & OpenTIR-PINN Research Team
Target Journal: IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)
"""

import os
import sys
import time
import json
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
from src.baselines.classical_tes import ClassicalTES
from src.training.input_utils import build_atmospheric_band_cache, assemble_input_tensor
from src.training.stage1_core import (
    get_sensor_wavelengths,
    get_sensor_band_mask,
    extract_training_priors,
    compute_input_norm_stats,
    standardize_input,
    build_stage1_pinn,
    build_stage1_criterion,
    compute_unified_metrics,
    compute_like_for_like_null_rmse
)

# Ratified statistically sound classes with >= 5 similarity groups
VALID_LOCO_CLASSES = [
    'botanical',
    'soil',
    'sandstone',
    'quartz'
]


def load_run_d_config(sensor: str) -> dict:
    """Loads the optimal Optuna-tuned configuration for Run D."""
    cfg_path = os.path.join(PROJECT_ROOT, 'outputs', f'best_config_{sensor}.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Run D config not found: {cfg_path}")
    with open(cfg_path, 'r') as f:
        data = json.load(f)
    return data['params']


def evaluate_dataset_pinn(
    model: nn.Module,
    dataset: list,
    criterion: nn.Module,
    wav_tensor: torch.Tensor,
    device: torch.device,
    input_mode: str,
    atm_cache: dict,
    mu_x: torch.Tensor,
    std_x: torch.Tensor,
    mask: torch.Tensor,
    null_rmse_e: float = 0.0,
    std_t: float = 15.0,
    std_e: float = 0.05,
    batch_size: int = 32
) -> dict:
    """Evaluates PINN model on a dataset with standardized inputs and band masking."""
    if len(dataset) == 0:
        return {
            'val_loss': np.nan,
            'temp_rmse': np.nan,
            'temp_mae': np.nan,
            'eps_rmse': np.nan,
            'sam_deg': np.nan,
            'eps_gain_over_null_pct': np.nan,
            'composite_score': np.nan,
            'n_samples': 0
        }

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    all_t_pred, all_t_true = [], []
    all_e_pred, all_e_true = [], []
    val_loss_sum = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            bs = len(batch['target_temp'])
            x_in = assemble_input_tensor(batch, input_mode, atm_cache, device, mask=mask)
            x_norm = standardize_input(x_in, mu_x, std_x)

            l_surf = batch['input_radiance'].to(device)[:, mask]
            l_down = batch['downwelling_radiance'].to(device)[:, mask]
            t_true = batch['target_temp'].to(device)
            e_true = batch['target_emissivity'].to(device)[:, mask]

            t_pred, e_pred = model(x_norm)

            loss = criterion(
                t_pred=t_pred,
                t_true=t_true,
                eps_pred=e_pred,
                eps_true=e_true,
                l_surf_obs=l_surf,
                l_down=l_down,
                wavelengths=wav_tensor[mask]
            )

            val_loss_sum += float(loss.item()) * bs
            n_samples += bs

            all_t_pred.append(t_pred.cpu())
            all_t_true.append(t_true.cpu())
            all_e_pred.append(e_pred.cpu())
            all_e_true.append(e_true.cpu())

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
    metrics['val_loss'] = val_loss_sum / max(n_samples, 1)
    metrics['n_samples'] = n_samples
    return metrics


def train_loco_class_production(
    held_out_class: str,
    all_dev: list,
    sensor: str,
    config: dict,
    epochs: int,
    device: torch.device,
    wav_tensor: torch.Tensor,
    atm_cache: dict
) -> dict:
    """Trains a tuned Run D PINN model with held_out_class completely excluded, and pairs with TES."""
    print(f"\n{'='*75}")
    print(f"LOCO Production Run: Holding Out [{held_out_class.upper()}] | Sensor: {sensor.upper()} | Epochs: {epochs}")
    print(f"{'='*75}")

    # Partition dataset
    val_ood = [s for s in all_dev if s['material_class'] == held_out_class]
    other_samples = [s for s in all_dev if s['material_class'] != held_out_class]

    # Group-aware split for early stopping on in-distribution classes
    # Group by similarity_group or sample_id
    group_map = {}
    for s in other_samples:
        gid = s.get('similarity_group', s.get('sample_id', 'unknown'))
        group_map.setdefault(gid, []).append(s)

    unique_groups = list(group_map.keys())
    rng = np.random.RandomState(42)
    rng.shuffle(unique_groups)

    n_val_groups = max(1, int(0.15 * len(unique_groups)))
    val_groups = set(unique_groups[:n_val_groups])
    train_groups = set(unique_groups[n_val_groups:])

    train_loco = [s for gid in train_groups for s in group_map[gid]]
    val_id = [s for gid in val_groups for s in group_map[gid]]

    print(f"Train samples (N-1 classes): {len(train_loco)} across {len(train_groups)} groups")
    print(f"In-distribution val samples: {len(val_id)} across {len(val_groups)} groups")
    print(f"Held-out OOD test samples:   {len(val_ood)} (Class: {held_out_class})")

    # 1. Band Mask Setup
    mask = get_sensor_band_mask(sensor, train_loco, device=device)
    n_bands_eff = int(mask.sum())
    wav_eff = wav_tensor[mask]

    # 2. Extract Training Priors strictly on train_loco
    priors = extract_training_priors(train_loco, mask, device=device)
    mu_t, std_t, var_t = priors['mu_t'], priors['std_t'], priors['var_t']
    mu_e, std_e, var_e = priors['mu_e'], priors['std_e'], priors['var_e']

    id_null_rmse_e = compute_like_for_like_null_rmse(val_id, mu_e, mask)
    ood_null_rmse_e = compute_like_for_like_null_rmse(val_ood, mu_e, mask)

    # 3. Input Mode & Loader Setup
    batch_size = int(config.get('batch_size', 8))
    train_loader = DataLoader(train_loco, batch_size=batch_size, shuffle=True)
    input_mode = str(config.get('input_mode', 'triple' if sensor == 'aster' else 'single'))
    lr = float(config.get('lr', 2.5e-4))
    opt_name = str(config.get('optimizer_name', 'AdamW'))
    weight_decay = float(config.get('weight_decay', 1e-5))

    # 4. Input Standardization Statistics
    mu_x, std_x = compute_input_norm_stats(train_loco, input_mode, atm_cache, mask, device=device)

    # 5. Build Stage 1 PINN & Criterion with Run D tuned parameters
    model = build_stage1_pinn(sensor, n_bands_eff, config, mu_t, mu_e, device=device)
    criterion = build_stage1_criterion(sensor, config, var_t, var_e, device=device)

    if opt_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_score = float('inf')
    best_model_state = None
    start_time = time.time()
    w_phys_target = float(config.get('w_phys', 0.01 if sensor == 'aster' else 0.024))

    print(f"Beginning 100-Epoch LOCO Training with SiLU and Plateau Scheduler...")
    for epoch in range(epochs):
        # Warmup physics
        criterion.w_phys = 0.0 if epoch == 0 else w_phys_target

        model.train()
        for batch in train_loader:
            x_in = assemble_input_tensor(batch, input_mode, atm_cache, device, mask=mask)
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

        # Validation step
        id_metrics = evaluate_dataset_pinn(
            model=model,
            dataset=val_id,
            criterion=criterion,
            wav_tensor=wav_tensor,
            device=device,
            input_mode=input_mode,
            atm_cache=atm_cache,
            mu_x=mu_x,
            std_x=std_x,
            mask=mask,
            null_rmse_e=id_null_rmse_e,
            std_t=std_t,
            std_e=std_e,
            batch_size=batch_size
        )
        scheduler.step(id_metrics['composite_score'])

        if id_metrics['composite_score'] < best_val_score:
            best_val_score = id_metrics['composite_score']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or (epoch + 1) == epochs:
            print(f"  [Epoch {epoch+1:03d}/{epochs}] Val Comp Score: {id_metrics['composite_score']:.4f} | dT RMSE: {id_metrics['temp_rmse']:.2f} K | dE RMSE: {id_metrics['eps_rmse']:.4f}")

    # Load best checkpoint
    model.load_state_dict(best_model_state)

    # 6. Final PINN evaluation on unseen held-out class
    pinn_ood = evaluate_dataset_pinn(
        model=model,
        dataset=val_ood,
        criterion=criterion,
        wav_tensor=wav_tensor,
        device=device,
        input_mode=input_mode,
        atm_cache=atm_cache,
        mu_x=mu_x,
        std_x=std_x,
        mask=mask,
        null_rmse_e=ood_null_rmse_e,
        std_t=std_t,
        std_e=std_e,
        batch_size=batch_size
    )

    # 7. Paired Classical TES evaluation on the EXACT same held-out samples
    print(f"  Evaluating paired Classical TES on {len(val_ood)} held-out samples...")
    tes = ClassicalTES(sensor=sensor)
    df_tes = tes.evaluate_dataset(val_ood, dataset_name=f'loco_{held_out_class}')
    tes_temp_rmse = float(np.sqrt((df_tes['delta_t_k']**2).mean()))
    tes_temp_mae = float(df_tes['abs_delta_t_k'].mean())
    tes_eps_rmse = float(df_tes['rmse_emissivity'].mean())
    tes_sam_deg = float(df_tes['sam_deg'].mean())

    # Save model
    models_dir = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'loco_D', sensor)
    os.makedirs(models_dir, exist_ok=True)
    loco_model_path = os.path.join(models_dir, f'model_{sensor}_loco_{held_out_class}.pth')
    torch.save({
        'held_out_class': held_out_class,
        'sensor': sensor,
        'config': config,
        'model_state_dict': best_model_state,
        'norm_stats': {'mu_x': mu_x.cpu(), 'std_x': std_x.cpu(), 'mu_t': mu_t, 'std_t': std_t, 'mu_e': mu_e.cpu(), 'std_e': std_e},
        'band_mask': mask.cpu(),
        'pinn_metrics': pinn_ood,
        'tes_metrics': {
            'temp_rmse': tes_temp_rmse,
            'temp_mae': tes_temp_mae,
            'eps_rmse': tes_eps_rmse,
            'sam_deg': tes_sam_deg
        }
    }, loco_model_path)

    wall_clock = time.time() - start_time
    delta_t_impr_pct = (tes_temp_rmse - pinn_ood['temp_rmse']) / tes_temp_rmse * 100.0
    delta_e_impr_pct = (tes_eps_rmse - pinn_ood['eps_rmse']) / tes_eps_rmse * 100.0

    print(f"\n--- LOCO Summary: Held-out Class [{held_out_class.upper()}] ---")
    print(f"  PINN dT RMSE: {pinn_ood['temp_rmse']:.2f} K  |  TES dT RMSE: {tes_temp_rmse:.2f} K  |  Diff: {delta_t_impr_pct:+.1f} %")
    print(f"  PINN dE RMSE: {pinn_ood['eps_rmse']:.4f}  |  TES dE RMSE: {tes_eps_rmse:.4f}  |  Diff: {delta_e_impr_pct:+.1f} %")
    print(f"  PINN SAM:     {pinn_ood['sam_deg']:.2f}°   |  TES SAM:     {tes_sam_deg:.2f}°")
    print(f"  Wall time:    {wall_clock/60:.1f} min")

    return {
        'sensor': sensor,
        'held_out_class': held_out_class,
        'n_samples': len(val_ood),
        'pinn_temp_rmse': pinn_ood['temp_rmse'],
        'tes_temp_rmse': tes_temp_rmse,
        'lst_rmse_impr_pct': delta_t_impr_pct,
        'pinn_temp_mae': pinn_ood['temp_mae'],
        'tes_temp_mae': tes_temp_mae,
        'pinn_eps_rmse': pinn_ood['eps_rmse'],
        'tes_eps_rmse': tes_eps_rmse,
        'eps_rmse_impr_pct': delta_e_impr_pct,
        'pinn_sam_deg': pinn_ood['sam_deg'],
        'tes_sam_deg': tes_sam_deg,
        'wall_clock_s': wall_clock
    }


def run_loco_production(
    sensor: str = 'all',
    epochs: int = 100
) -> None:
    """Runs the production LOCO suite for aster, hytes, or both."""
    sensors = ['aster', 'hytes'] if sensor == 'all' else [sensor]
    benchmarks_dir = os.path.join(PROJECT_ROOT, 'outputs', 'benchmarks', 'loco_production_D')
    os.makedirs(benchmarks_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing LOCO Production Benchmark | Sensors: {sensors} | Epochs: {epochs} | Device: {device}")

    all_records = []
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets')

    for sens in sensors:
        config = load_run_d_config(sens)
        wav_tensor = get_sensor_wavelengths(sens, device)
        atm_cache = build_atmospheric_band_cache(sens)

        train_path = os.path.join(data_dir, f'train_fold0_{sens}.pt')
        val_path = os.path.join(data_dir, f'val_fold0_{sens}.pt')
        all_dev = list(torch.load(train_path, weights_only=False)) + list(torch.load(val_path, weights_only=False))

        sens_records = []
        for cls_name in VALID_LOCO_CLASSES:
            rec = train_loco_class_production(
                held_out_class=cls_name,
                all_dev=all_dev,
                sensor=sens,
                config=config,
                epochs=epochs,
                device=device,
                wav_tensor=wav_tensor,
                atm_cache=atm_cache
            )
            sens_records.append(rec)
            all_records.append(rec)

        df_sens = pd.DataFrame(sens_records)
        df_sens.to_csv(os.path.join(benchmarks_dir, f'loco_production_D_{sens}.csv'), index=False)
        print(f"\nSaved {sens.upper()} LOCO summary to: {os.path.join(benchmarks_dir, f'loco_production_D_{sens}.csv')}")

    df_all = pd.DataFrame(all_records)
    df_all.to_csv(os.path.join(benchmarks_dir, 'loco_production_D_master.csv'), index=False)
    print("\n" + "="*80)
    print("LOCO PRODUCTION BENCHMARK COMPLETE ACROSS ALL CLASSES AND SENSORS")
    print("="*80)
    print(df_all.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LOCO Production Benchmark - Run D")
    parser.add_argument('--sensor', type=str, default='all', choices=['aster', 'hytes', 'all'])
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    run_loco_production(sensor=args.sensor, epochs=args.epochs)

