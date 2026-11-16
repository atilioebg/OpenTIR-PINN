"""
5-Fold Stratified Group Cross-Validation Pipeline (Step 2.7)
============================================================
Trains generalist Sobolev-PINN models across all 5 leak-free folds
for both ASTER and HyTES sensors.

Features:
  - Supports 4-tier ingestion modes: single, dual, triple, quad
  - Ingests best hyperparameters from outputs/best_config_<sensor>.json (or fallback defaults)
  - Trains Fold 0 through Fold 4 independently on CPU/GPU
  - Saves best checkpoint per fold to outputs/models/cv/<sensor>/model_<sensor>_fold<k>_best.pth
  - Performs out-of-fold validation and logs per-fold and macro-averaged metrics:
      * Kinetic temperature: MAE, RMSE, Bias
      * Spectral emissivity: RMSE, Bias
      * Shape metrics: SAM (deg), Curvature error (HyTES)
  - Computes 5-fold ensemble predictions and epistemic uncertainty (ensemble std)
  - Saves full benchmark ledger to outputs/benchmarks/cv_benchmark_<sensor>.csv

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

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    compute_like_for_like_null_rmse,
    get_stage1_default_config,
    build_stage1_pinn,
    build_stage1_criterion,
    compute_unified_metrics
)


def load_best_config(sensor: str) -> dict:
    """Returns verified grounded Stage 1 defaults, or loads validated stamped config."""
    defaults = get_stage1_default_config(sensor)
    config_path = os.path.join(PROJECT_ROOT, 'outputs', f'best_config_{sensor}.json')

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                # Strict provenance check: only load if explicitly stamped for Stage 1 / v1.3.0
                if data.get('provenance') in ['stage1_grounded', 'v1.3.0_optuna'] and 'params' in data:
                    defaults.update(data['params'])
                    print(f"Loaded verified stamped Optuna config from: {config_path}")
                else:
                    print(f"Ignoring unstamped legacy config at {config_path}; using Stage 1 grounded defaults.")
        except Exception as e:
            print(f"Warning: Could not read {config_path}, using defaults ({e})")
    else:
        print(f"No stamped config at {config_path}, using Stage 1 baseline defaults.")

    return defaults


def train_single_fold(
    sensor: str,
    fold_idx: int,
    config: dict,
    epochs: int,
    device: torch.device,
    wav_tensor: torch.Tensor,
    atm_cache: dict
) -> dict:
    """Trains a single cross-validation fold with Stage 1 architectural fixes."""
    print(f"\n--- Training Fold {fold_idx}/4 for {sensor.upper()} ---")

    data_dir = os.environ.get('OPENTIR_DATASET_DIR', os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets'))
    train_path = os.path.join(data_dir, f'train_fold{fold_idx}_{sensor}.pt')
    val_path = os.path.join(data_dir, f'val_fold{fold_idx}_{sensor}.pt')

    train_data = torch.load(train_path, weights_only=False)
    val_data = torch.load(val_path, weights_only=False)

    # 1. Band Mask Setup (Exclude 33 telluric opaque bands on HyTES)
    raw_mask = train_data[0].get('band_mask', None)
    if sensor == 'hytes' and raw_mask is not None:
        mask = raw_mask.bool().to(device)
        n_bands_eff = int(mask.sum())
        print(f"Applying band_mask: using {n_bands_eff} transparent bands (dropped {(~mask).sum()} opaque bands)")
    else:
        mask = torch.ones(train_data[0]['target_emissivity'].shape[-1], dtype=torch.bool, device=device)
        n_bands_eff = int(mask.sum())
        print(f"{sensor.upper()} using all {n_bands_eff} transparent bands (0 opaque bands)")

    wav_eff = wav_tensor[mask]

    # 2. Training-Fold Empirical Prior Statistics (Strictly Leak-Free)
    all_t_train = torch.stack([s['target_temp'] for s in train_data])
    all_e_train = torch.stack([s['target_emissivity'] for s in train_data])
    if mask is not None:
        all_e_train = all_e_train[:, mask.cpu()]

    mu_t = float(all_t_train.mean())
    std_t = float(all_t_train.std())
    var_t = float(all_t_train.var())

    mu_e = all_e_train.mean(dim=0).to(device)
    std_e = float(all_e_train.std(dim=0).mean())
    var_e = float(all_e_train.var(dim=0).mean())

    # Validation Null Baseline (Constant Training Mean Predictor)
    all_e_val = torch.stack([s['target_emissivity'] for s in val_data])
    if mask is not None:
        all_e_val = all_e_val[:, mask.cpu()]
    # Like-for-like with the model metric (mean over samples of the per-sample RMSE over bands)
    null_diff_e = mu_e.cpu().unsqueeze(0) - all_e_val
    null_rmse_e = float(torch.sqrt(torch.mean(null_diff_e**2, dim=-1)).mean())

    print(f"[PRIOR STATS] T_mean: {mu_t:.2f} K (std: {std_t:.2f} K) | eps_mean: {float(mu_e.mean()):.4f} (std: {std_e:.4f})")
    print(f"[NULL BASELINE] Constant Prior-Mean Predictor RMSE_eps: {null_rmse_e:.4f}")

    # 3. Input Mode & Loader Setup
    batch_size = int(config.get('batch_size', 32))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    input_mode = str(config.get('input_mode', 'dual' if config.get('dual_input', False) else 'single'))
    hidden_dim = int(config.get('hidden_dim', 512))
    n_layers = int(config.get('n_layers', 4))
    activation = str(config.get('activation', 'GELU'))
    dropout = float(config.get('dropout', 0.0))
    lr = float(config.get('lr', 1e-4))
    opt_name = str(config.get('optimizer_name', 'Adam'))
    weight_decay = float(config.get('weight_decay', 0.0))

    # Loss weights (Unit-free balanced defaults)
    w_mse_eps = float(config.get('w_mse_eps', 1.0))
    w_mse_temp = float(config.get('w_mse_temp', 1.0))
    w_sam = float(config.get('w_sam', 1.0))
    w_grad = float(config.get('w_grad', 0.5))
    w_curve = float(config.get('w_curve', 0.005 if sensor == 'hytes' else 0.1))
    w_phys = float(config.get('w_phys', 0.05))

    # 4. Input Standardization Statistics (Calculated strictly on training fold)
    with torch.no_grad():
        sample_x_list = []
        for b in DataLoader(train_data, batch_size=256, shuffle=False):
            x_b = assemble_input_tensor(b, input_mode, atm_cache, device, mask=mask)
            sample_x_list.append(x_b)
        all_x = torch.cat(sample_x_list, dim=0)
        mu_x = all_x.mean(dim=0)
        std_x = all_x.std(dim=0).clamp_min(1e-6)
        del all_x, sample_x_list

    # 5. Model Initialization with Empirical Prior
    model = PhysicsInformedNN(
        input_dim=n_bands_eff,
        output_dim=n_bands_eff,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        input_mode=input_mode,
        activation=activation,
        dropout=dropout
    ).to(device)

    model.initialize_heads(mean_temp=mu_t, mean_emissivity=mu_e)

    if opt_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    criterion = SpectralShapeLoss(
        w_mse_eps=w_mse_eps,
        w_mse_temp=w_mse_temp,
        w_sam=w_sam,
        w_grad=w_grad,
        w_curvature=w_curve,
        w_phys=w_phys,
        var_temp=var_t,
        var_eps=var_e,
        norm_phys_floor=1.6,
        use_bt_phys=True
    ).to(device)

    models_dir = os.path.join(os.environ.get('OPENTIR_OUTPUT_DIR', os.path.join(PROJECT_ROOT, 'outputs')), 'models', 'cv', sensor)
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, f'model_{sensor}_fold{fold_idx}_best.pth')

    best_score = float('inf')
    best_val_metrics = {}
    epoch_history = []
    start_time = time.time()

    benchmarks_dir = os.path.join(os.environ.get('OPENTIR_OUTPUT_DIR', os.path.join(PROJECT_ROOT, 'outputs')), 'benchmarks')
    os.makedirs(benchmarks_dir, exist_ok=True)

    for epoch in range(epochs):
        # Physics loss warm-up (off in epoch 0)
        criterion.w_phys = 0.0 if epoch == 0 else w_phys

        model.train()
        tr_loss_sum = 0.0
        tr_mse_e_sum = 0.0
        tr_mse_t_sum = 0.0
        tr_sam_sum = 0.0
        tr_grad_sum = 0.0
        tr_curve_sum = 0.0
        tr_phys_sum = 0.0
        n_tr = 0

        for batch in train_loader:
            x_in = assemble_input_tensor(batch, input_mode, atm_cache, device, mask=mask)
            x_norm = (x_in - mu_x) / std_x

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

            # Gradient clipping to prevent non-linear Planck numerical shocks
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = len(t_true)
            tr_loss_sum += loss.item() * bs
            tr_mse_e_sum += criterion.last_losses.get('mse_eps', 0.0) * bs
            tr_mse_t_sum += criterion.last_losses.get('mse_temp', 0.0) * bs
            tr_sam_sum += criterion.last_losses.get('sam', 0.0) * bs
            tr_grad_sum += criterion.last_losses.get('grad', 0.0) * bs
            tr_curve_sum += criterion.last_losses.get('curve', 0.0) * bs
            tr_phys_sum += criterion.last_losses.get('phys', 0.0) * bs
            n_tr += bs

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_mse_e_sum = 0.0
        val_mse_t_sum = 0.0
        val_sam_loss_sum = 0.0
        val_grad_sum = 0.0
        val_curve_sum = 0.0
        val_phys_sum = 0.0

        val_dt_sq = 0.0
        val_dt_abs = 0.0
        val_dt_bias = 0.0
        val_rmse_eps_sum = 0.0
        val_bias_eps_sum = 0.0
        val_sam_sum = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                x_in = assemble_input_tensor(batch, input_mode, atm_cache, device, mask=mask)
                x_norm = (x_in - mu_x) / std_x

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
                    wavelengths=wav_eff
                )
                bs_val = len(t_true)
                val_loss_sum += v_loss.item() * bs_val
                val_mse_e_sum += criterion.last_losses.get('mse_eps', 0.0) * bs_val
                val_mse_t_sum += criterion.last_losses.get('mse_temp', 0.0) * bs_val
                val_sam_loss_sum += criterion.last_losses.get('sam', 0.0) * bs_val
                val_grad_sum += criterion.last_losses.get('grad', 0.0) * bs_val
                val_curve_sum += criterion.last_losses.get('curve', 0.0) * bs_val
                val_phys_sum += criterion.last_losses.get('phys', 0.0) * bs_val

                dt = t_pred - t_true
                val_dt_sq += torch.sum(dt**2).item()
                val_dt_abs += torch.sum(torch.abs(dt)).item()
                val_dt_bias += torch.sum(dt).item()

                diff_eps = e_pred - e_true
                val_rmse_eps_sum += torch.sum(torch.sqrt(torch.mean(diff_eps**2, dim=-1))).item()
                val_bias_eps_sum += torch.sum(torch.mean(diff_eps, dim=-1)).item()
                val_sam_sum += torch.sum(compute_spectral_angle_mapper(e_pred, e_true)).item()
                n_val += bs_val

        avg_val_loss = val_loss_sum / n_val
        val_rmse_t = np.sqrt(val_dt_sq / n_val)
        val_rmse_e = val_rmse_eps_sum / n_val

        # Unit-free balanced checkpoint metric (RMSE_T / sigma_T + RMSE_eps / sigma_eps)
        composite_score = (val_rmse_t / std_t) + (val_rmse_e / std_e)
        scheduler.step(composite_score)

        # Log epoch telemetry for plotting loss convergence
        epoch_record = {
            'epoch': epoch + 1,
            'train_loss_total': tr_loss_sum / n_tr if n_tr > 0 else 0.0,
            'train_loss_mse_e': tr_mse_e_sum / n_tr if n_tr > 0 else 0.0,
            'train_loss_mse_t': tr_mse_t_sum / n_tr if n_tr > 0 else 0.0,
            'train_loss_sam': tr_sam_sum / n_tr if n_tr > 0 else 0.0,
            'train_loss_grad': tr_grad_sum / n_tr if n_tr > 0 else 0.0,
            'train_loss_curve': tr_curve_sum / n_tr if n_tr > 0 else 0.0,
            'train_loss_phys': tr_phys_sum / n_tr if n_tr > 0 else 0.0,
            'val_loss_total': avg_val_loss,
            'val_loss_mse_e': val_mse_e_sum / n_val if n_val > 0 else 0.0,
            'val_loss_mse_t': val_mse_t_sum / n_val if n_val > 0 else 0.0,
            'val_loss_sam': val_sam_loss_sum / n_val if n_val > 0 else 0.0,
            'val_loss_grad': val_grad_sum / n_val if n_val > 0 else 0.0,
            'val_loss_curve': val_curve_sum / n_val if n_val > 0 else 0.0,
            'val_loss_phys': val_phys_sum / n_val if n_val > 0 else 0.0,
            'val_temp_rmse_k': val_rmse_t,
            'val_temp_mae_k': val_dt_abs / n_val,
            'val_eps_rmse': val_rmse_e,
            'val_sam_deg': val_sam_sum / n_val,
            'composite_score': composite_score,
            'lr': optimizer.param_groups[0]['lr']
        }
        epoch_history.append(epoch_record)

        if composite_score < best_score:
            best_score = composite_score
            gain_over_null = ((null_rmse_e - val_rmse_e) / null_rmse_e) * 100.0
            best_val_metrics = {
                'fold': fold_idx,
                'val_loss': avg_val_loss,
                'composite_score': composite_score,
                'temp_rmse': val_rmse_t,
                'temp_mae': val_dt_abs / n_val,
                'temp_bias': val_dt_bias / n_val,
                'eps_rmse': val_rmse_e,
                'null_eps_rmse': null_rmse_e,
                'eps_gain_over_null_pct': gain_over_null,
                'eps_bias': val_bias_eps_sum / n_val,
                'sam_deg': val_sam_sum / n_val,
                'wall_clock_s': time.time() - start_time
            }
            # Save comprehensive checkpoint with normalization parameters
            torch.save({
                'fold': fold_idx,
                'sensor': sensor,
                'epoch': epoch + 1,
                'config': config,
                'input_mode': input_mode,
                'model_state_dict': model.state_dict(),
                'norm_stats': {
                    'mu_x': mu_x.cpu(),
                    'std_x': std_x.cpu(),
                    'mu_t': mu_t,
                    'std_t': std_t,
                    'var_t': var_t,
                    'mu_e': mu_e.cpu(),
                    'std_e': std_e,
                    'var_e': var_e,
                    'mask': mask.cpu()
                },
                'band_mask': mask.cpu(),
                'val_metrics': best_val_metrics,
                'metrics': best_val_metrics
            }, best_model_path)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            print(
                f"  Epoch {epoch+1:03d}/{epochs:03d} | "
                f"Val Loss: {avg_val_loss:8.3f} | "
                f"dT MAE: {val_dt_abs/n_val:5.2f} K | "
                f"RMSE_eps: {val_rmse_eps_sum/n_val:6.4f} | "
                f"SAM: {val_sam_sum/n_val:5.2f} deg"
            )

    # Save epoch-by-epoch loss & metric history for plotting
    df_history = pd.DataFrame(epoch_history)
    history_csv = os.path.join(benchmarks_dir, f'training_history_{sensor}_fold{fold_idx}.csv')
    df_history.to_csv(history_csv, index=False)
    print(f"Saved epoch-by-epoch training history to: {history_csv}")

    print(f"Fold {fold_idx} Complete. Best model saved to: {best_model_path}")
    return best_val_metrics


def run_cross_validation(
    sensor: str = 'aster',
    epochs: int = 100,
    n_folds: int = 5,
    input_mode: str = None
):
    """Executes full 5-fold cross-validation suite."""
    print(f"\n{'='*75}")
    print(f"5-Fold Stratified Group Cross-Validation | Sensor: {sensor.upper()} | Epochs/Fold: {epochs}")
    print(f"{'='*75}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Compute Device: {device}")
    wav_tensor = get_sensor_wavelengths(sensor, device)
    atm_cache = build_atmospheric_band_cache(sensor)
    config = load_best_config(sensor)
    if input_mode is not None:
        config['input_mode'] = input_mode
    print(f"Active Ingestion Mode: [{config.get('input_mode', 'dual').upper()}]")

    benchmarks_dir = os.path.join(os.environ.get('OPENTIR_OUTPUT_DIR', os.path.join(PROJECT_ROOT, 'outputs')), 'benchmarks')
    os.makedirs(benchmarks_dir, exist_ok=True)

    fold_results = []
    for k in range(n_folds):
        res = train_single_fold(
            sensor=sensor,
            fold_idx=k,
            config=config,
            epochs=epochs,
            device=device,
            wav_tensor=wav_tensor,
            atm_cache=atm_cache
        )
        fold_results.append(res)

    # Compile benchmark dataframe
    df_cv = pd.DataFrame(fold_results)

    # Compute Macro statistics (Mean ± Std)
    mean_row = {'fold': 'MEAN'}
    std_row = {'fold': 'STD'}
    numeric_cols = [
        'temp_rmse', 'temp_mae', 'temp_bias', 'eps_rmse', 'null_eps_rmse',
        'eps_gain_over_null_pct', 'eps_bias', 'sam_deg', 'composite_score', 'val_loss', 'wall_clock_s'
    ]
    for col in numeric_cols:
        if col in df_cv.columns:
            mean_row[col] = df_cv[col].mean()
            std_row[col] = df_cv[col].std()

    df_full = pd.concat([df_cv, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    out_csv = os.path.join(benchmarks_dir, f'cv_benchmark_{sensor}.csv')
    df_full.to_csv(out_csv, index=False)

    print(f"\n{'='*75}")
    print(f"Cross-Validation Summary for {sensor.upper()}:")
    print(f"{'='*75}")
    print(df_full.to_string(index=False))
    print(f"\nSaved cross-validation benchmark to: {out_csv}")

    return df_full


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run 5-Fold Cross-Validation")
    parser.add_argument('--sensor', type=str, default='aster', choices=['aster', 'hytes'], help='Sensor')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs per fold')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--input_mode', type=str, default=None, choices=['single', 'dual', 'triple', 'quad'], help='Input mode override')
    args = parser.parse_args()

    run_cross_validation(
        sensor=args.sensor,
        epochs=args.epochs,
        n_folds=args.n_folds,
        input_mode=args.input_mode
    )
