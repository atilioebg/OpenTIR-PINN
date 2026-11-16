"""
Leave-One-Class-Out (LOCO 7+1) Zero-Shot Diagnostic Mineral Stress-Test Suite
=============================================================================
Evaluates model physical inductive bias and zero-shot transfer across 7 diagnostic
crystallographic mineral classes held out completely from training:
  1. Quartz (SiO2 reststrahlen doublet)
  2. Calcite (CaCO3 carbonate in-plane bending mode)
  3. Feldspar / Albite (Tectosilicate asymmetric wing)
  4. Kaolinite (Phyllosilicate Al-OH doublet)
  5. Basalt (Mafic volcanic silicate broad absorption)
  6. Sandstone (Quartzose sedimentary fundamental)
  7. Granite (Felsic plutonic multi-mineral lattice)

Migrated to Stage 1 Grounded Framework:
  - Band masking (223 clean HyTES bands).
  - Training-fold prior head initialization (mu_t, mu_e).
  - Training-fold input radiance standardization (mu_x, std_x).
  - Unit-free balanced multi-task loss.
  - Comprehensive checkpoint saving with normalization statistics.

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

# Project root
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

DIAGNOSTIC_LOCO_CLASSES = [
    'quartz',
    'calcite',
    'feldspar',
    'kaolinite',
    'basalt',
    'sandstone',
    'granite'
]


def evaluate_dataset(
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
    """Evaluates model on a dataset with standardized inputs and band masking."""
    if len(dataset) == 0:
        return {
            'val_loss': np.nan,
            'temp_rmse': np.nan,
            'temp_mae': np.nan,
            'eps_rmse': np.nan,
            'sam_deg': np.nan,
            'gain_over_null_pct': np.nan,
            'n_samples': 0
        }

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    all_t_pred = []
    all_t_true = []
    all_e_pred = []
    all_e_true = []
    val_loss_sum = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            x_in = assemble_input_tensor(batch, input_mode, atm_cache, device, mask=mask)
            x_norm = standardize_input(x_in, mu_x, std_x)

            l_surf = batch['input_radiance'].to(device)[:, mask]
            l_down = batch['downwelling_radiance'].to(device)[:, mask]
            t_true = batch['target_temp'].to(device)
            e_true = batch['target_emissivity'].to(device)[:, mask]

            t_pred, e_pred = model(x_norm)

            v_loss = criterion(
                t_pred=t_pred,
                t_true=t_true,
                eps_pred=e_pred,
                eps_true=e_true,
                l_surf_obs=l_surf,
                l_down=l_down,
                wavelengths=wav_tensor
            )
            val_loss_sum += v_loss.item() * len(t_true)

            all_t_pred.append(t_pred)
            all_t_true.append(t_true)
            all_e_pred.append(e_pred)
            all_e_true.append(e_true)
            n_samples += len(t_true)

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


def train_loco_class(
    held_out_class: str,
    all_dev: list,
    sensor: str,
    config: dict,
    epochs: int,
    device: torch.device,
    wav_tensor: torch.Tensor,
    atm_cache: dict
) -> dict:
    """Trains a model with held_out_class completely excluded from training set."""
    print(f"\n{'='*70}")
    print(f"LOCO Experiment: Holding Out [{held_out_class.upper()}] | Sensor: {sensor.upper()}")
    print(f"{'='*70}")

    # Partition dataset from full development corpus
    val_ood = [s for s in all_dev if s['material_class'] == held_out_class]
    other_samples = [s for s in all_dev if s['material_class'] != held_out_class]

    # Deterministic 85/15 train/val split on in-distribution classes for early stopping
    np.random.seed(42)
    indices = np.random.permutation(len(other_samples))
    split_idx = int(0.85 * len(other_samples))
    train_loco = [other_samples[i] for i in indices[:split_idx]]
    val_id = [other_samples[i] for i in indices[split_idx:]]

    print(f"Train samples (N-1 classes): {len(train_loco)}")
    print(f"In-distribution val samples: {len(val_id)}")
    print(f"Held-out OOD test samples:   {len(val_ood)} (Class: {held_out_class})")

    # 1. Band Mask Setup
    mask = get_sensor_band_mask(sensor, train_loco, device=device)
    n_bands_eff = int(mask.sum())
    wav_eff = wav_tensor[mask]

    # 2. Extract Training Priors strictly on train_loco
    priors = extract_training_priors(train_loco, mask, device=device)
    mu_t, std_t, var_t = priors['mu_t'], priors['std_t'], priors['var_t']
    mu_e, std_e, var_e = priors['mu_e'], priors['std_e'], priors['var_e']

    # Null baseline for in-distribution val and OOD test
    id_null_rmse_e = compute_like_for_like_null_rmse(val_id, mu_e, mask)
    ood_null_rmse_e = compute_like_for_like_null_rmse(val_ood, mu_e, mask)

    # 3. Input Mode & Loader Setup
    batch_size = int(config.get('batch_size', 32))
    train_loader = DataLoader(train_loco, batch_size=batch_size, shuffle=True)
    input_mode = str(config.get('input_mode', 'dual'))
    lr = float(config.get('lr', 1e-4))
    opt_name = str(config.get('optimizer_name', 'Adam'))
    weight_decay = float(config.get('weight_decay', 0.0))

    # 4. Input Standardization Statistics
    mu_x, std_x = compute_input_norm_stats(train_loco, input_mode, atm_cache, mask, device=device)

    # 5. Build Stage 1 PINN & Criterion
    model = build_stage1_pinn(sensor, n_bands_eff, config, mu_t, mu_e, device=device)
    criterion = build_stage1_criterion(sensor, config, var_t, var_e, device=device)

    optimizer = optim.Adam(model.parameters(), lr=lr) if opt_name == 'Adam' else optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    best_val_score = float('inf')
    best_model_state = None
    start_time = time.time()
    w_phys_target = float(config.get('w_phys', 0.05))

    for epoch in range(epochs):
        # Physics warm-up in Epoch 0
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

        # Validation on in-distribution partition for checkpoint selection
        id_metrics = evaluate_dataset(
            model=model,
            dataset=val_id,
            criterion=criterion,
            wav_tensor=wav_eff,
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

    # Load best checkpoint
    model.load_state_dict(best_model_state)

    # Final evaluation on the unseen held-out class
    ood_metrics = evaluate_dataset(
        model=model,
        dataset=val_ood,
        criterion=criterion,
        wav_tensor=wav_eff,
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

    # Save model with normalization stats
    models_dir = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'loco', sensor)
    os.makedirs(models_dir, exist_ok=True)
    loco_model_path = os.path.join(models_dir, f'model_{sensor}_loco_{held_out_class}.pth')
    torch.save({
        'held_out_class': held_out_class,
        'sensor': sensor,
        'config': config,
        'input_mode': input_mode,
        'model_state_dict': best_model_state,
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
        'ood_metrics': ood_metrics
    }, loco_model_path)

    wall_clock = time.time() - start_time
    print(f"Results on Held-Out OOD [{held_out_class.upper()}]:")
    print(f"  dT MAE:   {ood_metrics['temp_mae']:.2f} K")
    print(f"  dT RMSE:  {ood_metrics['temp_rmse']:.2f} K")
    print(f"  RMSE_eps: {ood_metrics['eps_rmse']:.4f} (Null: {ood_null_rmse_e:.4f}, Gain: {ood_metrics['eps_gain_over_null_pct']:.1f}%)")
    print(f"  SAM:      {ood_metrics['sam_deg']:.2f} deg | Corr: {ood_metrics['spectral_corr']:.2f}")
    print(f"  Time:     {wall_clock:.1f} s")

    return {
        'held_out_class': held_out_class,
        'ood_temp_mae': ood_metrics['temp_mae'],
        'ood_temp_rmse': ood_metrics['temp_rmse'],
        'ood_eps_rmse': ood_metrics['eps_rmse'],
        'ood_null_eps_rmse': ood_null_rmse_e,
        'ood_gain_over_null_pct': ood_metrics['eps_gain_over_null_pct'],
        'ood_sam_deg': ood_metrics['sam_deg'],
        'ood_spectral_corr': ood_metrics['spectral_corr'],
        'ood_samples': ood_metrics['n_samples'],
        'train_samples': len(train_loco),
        'wall_clock_s': wall_clock
    }


def run_loco_suite(
    sensor: str = 'aster',
    epochs: int = 50,
    classes: Optional[list] = None
) -> pd.DataFrame:
    """Executes full LOCO 7+1 suite."""
    if classes is None:
        classes = DIAGNOSTIC_LOCO_CLASSES

    print(f"\n{'='*75}")
    print(f"Starting Leave-One-Class-Out (LOCO 7+1) Suite | Sensor: {sensor.upper()}")
    print(f"Held-out Classes: {classes}")
    print(f"{'='*75}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Compute Device: {device}")
    wav_tensor = get_sensor_wavelengths(sensor, device)
    atm_cache = build_atmospheric_band_cache(sensor)
    config = get_stage1_default_config(sensor)
    print(f"Active Ingestion Mode: [{config.get('input_mode', 'dual').upper()}]")

    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets')
    train_path = os.path.join(data_dir, f'train_fold0_{sensor}.pt')
    val_path = os.path.join(data_dir, f'val_fold0_{sensor}.pt')

    train_full = torch.load(train_path, weights_only=False)
    val_full = torch.load(val_path, weights_only=False)
    all_dev = list(train_full) + list(val_full)

    loco_records = []
    for cls_name in classes:
        rec = train_loco_class(
            held_out_class=cls_name,
            all_dev=all_dev,
            sensor=sensor,
            config=config,
            epochs=epochs,
            device=device,
            wav_tensor=wav_tensor,
            atm_cache=atm_cache
        )
        loco_records.append(rec)

    # Save to CSV
    benchmarks_dir = os.path.join(PROJECT_ROOT, 'outputs', 'benchmarks')
    os.makedirs(benchmarks_dir, exist_ok=True)
    df_loco = pd.DataFrame(loco_records)

    # Add macro mean row
    mean_row = {'held_out_class': 'MEAN_OOD'}
    for col in ['ood_temp_mae', 'ood_temp_rmse', 'ood_eps_rmse', 'ood_null_eps_rmse', 'ood_gain_over_null_pct', 'ood_sam_deg', 'ood_spectral_corr', 'wall_clock_s']:
        if col in df_loco.columns:
            mean_row[col] = df_loco[col].mean()
    df_full = pd.concat([df_loco, pd.DataFrame([mean_row])], ignore_index=True)

    out_csv = os.path.join(benchmarks_dir, f'loco_benchmark_{sensor}.csv')
    df_full.to_csv(out_csv, index=False)

    print(f"\n{'='*75}")
    print(f"LOCO 7+1 Benchmark Summary for {sensor.upper()}:")
    print(f"{'='*75}")
    print(df_full.to_string(index=False))
    print(f"\nSaved LOCO benchmark ledger to: {out_csv}")

    return df_full


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LOCO 7+1 Suite (Stage 1)")
    parser.add_argument('--sensor', type=str, default='aster', choices=['aster', 'hytes'], help='Sensor')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs per LOCO model')
    args = parser.parse_args()

    run_loco_suite(
        sensor=args.sensor,
        epochs=args.epochs
    )
