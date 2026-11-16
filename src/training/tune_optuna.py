"""
Optuna Hyperparameter Optimization Suite for Sobolev-PINN (Step 2.7)
===================================================================
Automates systematic hyperparameter exploration with grounded Stage 1 SSoT:
  - Architecture:
      * batch_size in [8, 16, 32]
      * n_layers in [3, 5, 7]
      * hidden_dim in [384, 512, 768]
      * dropout in [0.0, 0.10]
      * input_mode in ['single', 'dual', 'triple', 'quad'] (4-tier atmospheric coupling)
      * activation in ['GELU', 'SiLU', 'ELU'] (smooth non-linearities for Sobolev derivatives)
  - Optimization:
      * lr in [5e-5, 3e-4] (log)
      * optimizer in ['Adam', 'AdamW']
      * weight_decay in [1e-6, 1e-4] (if AdamW)
  - Sobolev Loss Weights:
      * w_mse_eps in [0.5, 2.0] (log)
      * w_sam in [0.5, 2.0] (log)
      * w_grad in [0.2, 1.0]
      * w_curve: [0.0, 0.20] for ASTER; [1e-4, 1e-2] for HyTES (calibrated for 1/Delta_lambda^2)
      * w_phys: [0.01, 0.10] (log)
  - Objective: Unit-free composite score J = (RMSE_T / sigma_T) + (RMSE_eps / sigma_eps)

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
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.pinn import PhysicsInformedNN
from src.models.losses import SpectralShapeLoss
from src.training.input_utils import build_atmospheric_band_cache, assemble_input_tensor
from src.training.stage1_core import (
    get_sensor_wavelengths,
    get_sensor_band_mask,
    extract_training_priors,
    compute_input_norm_stats,
    standardize_input,
    compute_like_for_like_null_rmse,
    compute_unified_metrics,
    build_stage1_pinn,
    build_stage1_criterion
)


def create_objective(
    sensor: str,
    train_dataset: list,
    val_dataset: list,
    device: torch.device,
    wav_tensor: torch.Tensor,
    mask: torch.Tensor,
    atm_cache: dict,
    priors: dict,
    norm_cache: dict,
    null_rmse_e: float,
    n_epochs_per_trial: int = 20,
    search_space: str = 'unified'
):
    """Factory creating the Optuna objective function with Stage 1 SSoT fixes."""
    n_bands_eff = int(mask.sum())
    wav_eff = wav_tensor[mask]

    def objective(trial: optuna.Trial) -> float:
        # 1. Hyperparameter Sampling
        lr = trial.suggest_float('lr', 5e-5, 3e-4, log=True)
        opt_name = trial.suggest_categorical('optimizer_name', ['Adam', 'AdamW'])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True) if opt_name == 'AdamW' else 0.0

        if search_space == 'unified':
            batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
            if sensor == 'hytes':
                hidden_dim = trial.suggest_categorical('hidden_dim', [384, 512, 768, 1024])
                n_layers = trial.suggest_categorical('n_layers', [2, 3, 4, 5, 7])
            else:
                hidden_dim = trial.suggest_categorical('hidden_dim', [256, 384, 512, 768])
                n_layers = trial.suggest_categorical('n_layers', [3, 4, 5, 6, 7])
        elif search_space == 'run_c':
            batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
            if sensor == 'hytes':
                hidden_dim = trial.suggest_categorical('hidden_dim', [512, 768, 1024])
                n_layers = trial.suggest_categorical('n_layers', [2, 3, 4])
            else:
                hidden_dim = trial.suggest_categorical('hidden_dim', [256, 384, 512])
                n_layers = trial.suggest_categorical('n_layers', [4, 5, 6, 7])
        else:  # run_b
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
            hidden_dim = trial.suggest_categorical('hidden_dim', [384, 512, 768])
            n_layers = trial.suggest_categorical('n_layers', [3, 5, 7])

        dropout = trial.suggest_float('dropout', 0.0, 0.10)

        # 4 Atmospheric Ingestion Modes & Activations
        input_mode = trial.suggest_categorical('input_mode', ['single', 'dual', 'triple', 'quad'])
        activation = trial.suggest_categorical('activation', ['GELU', 'SiLU', 'ELU'])

        # Retrieve leak-free input standardization stats for selected mode
        mu_x, std_x = norm_cache[input_mode]

        # Sobolev & Physical loss weights
        w_mse_eps = trial.suggest_float('w_mse_eps', 0.5, 2.0, log=True)
        w_sam = trial.suggest_float('w_sam', 0.5, 2.0, log=True)
        w_grad = trial.suggest_float('w_grad', 0.2, 1.0)

        if sensor == 'hytes':
            # Dimensionally calibrated for 1 / Delta_lambda^2 noise amplification
            w_curve = trial.suggest_float('w_curve', 1e-4, 1e-2, log=True)
        else:
            # Taylor-Fornberg across 3 interior triplets
            w_curve = trial.suggest_float('w_curve', 0.0, 0.20)

        # Bounded physics weight: prevents unphysical distortion of eps under PWV noise
        w_phys = trial.suggest_float('w_phys', 0.01, 0.10, log=True)

        trial_config = {
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'input_mode': input_mode,
            'activation': activation,
            'dropout': dropout,
            'w_mse_eps': w_mse_eps,
            'w_mse_temp': 1.0,
            'w_sam': w_sam,
            'w_grad': w_grad,
            'w_curve': w_curve,
            'w_phys': w_phys
        }

        print(f"\n{'='*70}", flush=True)
        print(f"--- [Trial {trial.number:02d} START] ---", flush=True)
        print(f"Parameters: batch_size={batch_size}, lr={lr:.2e}, hidden_dim={hidden_dim}, n_layers={n_layers}, input_mode={input_mode}, act={activation}, dropout={dropout:.3f}", flush=True)
        print(f"Loss weights: w_eps={w_mse_eps:.3f}, w_sam={w_sam:.3f}, w_grad={w_grad:.3f}, w_curve={w_curve:.4f}, w_phys={w_phys:.4f}", flush=True)
        print(f"{'-'*70}", flush=True)

        # 2. DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 3. Model & Criterion Construction with Stage 1 SSoT
        model = build_stage1_pinn(
            sensor=sensor,
            n_bands_eff=n_bands_eff,
            config=trial_config,
            mu_t=priors['mu_t'],
            mu_e=priors['mu_e'],
            device=device
        )

        criterion = build_stage1_criterion(
            sensor=sensor,
            config=trial_config,
            var_t=priors['var_t'],
            var_e=priors['var_e'],
            device=device
        )

        # Optimizer
        if opt_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        start_time = time.time()
        best_composite = float('inf')
        best_metrics = {}

        # 4. Training Loop
        for epoch in range(n_epochs_per_trial):
            criterion.w_phys = 0.0 if epoch == 0 else w_phys

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

            # 5. Validation Evaluation
            model.eval()
            all_t_pred, all_t_true = [], []
            all_e_pred, all_e_true = [], []

            with torch.no_grad():
                for batch in val_loader:
                    x_in = assemble_input_tensor(batch, input_mode, atm_cache, device, mask=mask)
                    x_norm = standardize_input(x_in, mu_x, std_x)

                    t_true = batch['target_temp'].to(device)
                    e_true = batch['target_emissivity'].to(device)[:, mask]

                    t_pred, e_pred = model(x_norm)

                    all_t_pred.append(t_pred)
                    all_t_true.append(t_true)
                    all_e_pred.append(e_pred)
                    all_e_true.append(e_true)

            cat_t_pred = torch.cat(all_t_pred, dim=0)
            cat_t_true = torch.cat(all_t_true, dim=0)
            cat_e_pred = torch.cat(all_e_pred, dim=0)
            cat_e_true = torch.cat(all_e_true, dim=0)

            epoch_metrics = compute_unified_metrics(
                t_pred=cat_t_pred,
                t_true=cat_t_true,
                e_pred=cat_e_pred,
                e_true=cat_e_true,
                null_rmse_e=null_rmse_e,
                std_t=priors['std_t'],
                std_e=priors['std_e']
            )

            composite_j = epoch_metrics['composite_score']

            if composite_j < best_composite:
                best_composite = composite_j
                best_metrics = epoch_metrics

            # Pruning check
            trial.report(composite_j, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        wall_clock = time.time() - start_time

        for k, v in best_metrics.items():
            trial.set_user_attr(k, float(v))
        trial.set_user_attr('wall_clock_s', float(wall_clock))

        print(f"\n--- [Trial {trial.number:02d} END] ---", flush=True)
        print(f"Trial {trial.number:02d} Result: J_composite={best_composite:.5f} | Temp RMSE={best_metrics['temp_rmse']:.3f} K | Eps RMSE={best_metrics['eps_rmse']:.5f} | SAM={best_metrics['sam_deg']:.3f}° | Time={wall_clock:.1f}s", flush=True)
        print(f"{'='*70}\n", flush=True)

        return best_composite

    return objective


def run_optuna_study(
    sensor: str = 'aster',
    n_trials: int = 20,
    n_epochs: int = 20,
    fold_idx: int = 0,
    search_space: str = 'unified'
):
    """Executes Optuna study with SQLite persistence and Stage 1 SSoT."""
    print(f"\n{'='*75}")
    print(f"Starting Grounded Optuna Study | Sensor: {sensor.upper()} | Trials: {n_trials} | Epochs/Trial: {n_epochs} | Space: {search_space}")
    print(f"{'='*75}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Compute Device: {device}")

    output_dir = os.path.join(PROJECT_ROOT, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    db_path = os.path.join(output_dir, f'optuna_v2_{sensor}.db')
    storage_url = f"sqlite:///{db_path}"

    # Load Fold 0 Datasets
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets')
    train_path = os.path.join(data_dir, f'train_fold{fold_idx}_{sensor}.pt')
    val_path = os.path.join(data_dir, f'val_fold{fold_idx}_{sensor}.pt')

    if not os.path.exists(train_path):
        train_path = os.path.join(data_dir, f'train_dataset_{sensor}.pt')
    if not os.path.exists(val_path):
        val_path = os.path.join(data_dir, f'val_dataset_{sensor}.pt')

    print(f"Training dataset:   {train_path}")
    print(f"Validation dataset: {val_path}")

    train_data = torch.load(train_path, weights_only=False)
    val_data = torch.load(val_path, weights_only=False)

    wav_tensor = get_sensor_wavelengths(sensor, device)
    mask = get_sensor_band_mask(sensor, train_data, device)
    atm_cache = build_atmospheric_band_cache(sensor)

    # Leak-free training prior extraction & input standardization cache
    priors = extract_training_priors(train_data, mask, device)
    null_rmse_e = compute_like_for_like_null_rmse(val_data, priors['mu_e'], mask)

    print(f"Training Prior mu_T: {priors['mu_t']:.2f} K, sigma_T: {priors['std_t']:.2f} K")
    print(f"Null Baseline RMSE_eps: {null_rmse_e:.4f}")

    print("Precomputing input standardization statistics for all 4 ingestion modes...")
    norm_cache = {}
    with torch.no_grad():
        for m_mode in ['single', 'dual', 'triple', 'quad']:
            mu_m, std_m = compute_input_norm_stats(train_data, m_mode, atm_cache, mask, device)
            norm_cache[m_mode] = (mu_m, std_m)

    # Sampler and Pruner setup
    sampler = TPESampler(multivariate=True, seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    study_name = f"pinn_optuna_v2_{sensor}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction='minimize',
        sampler=sampler,
        pruner=pruner
    )

    # Enqueue Run B (Trial 0) and Run C (Trial 1) Validated Reference Trials
    if sensor == 'hytes':
        baseline_params_b = {
            'lr': 0.0002255,
            'optimizer_name': 'AdamW',
            'weight_decay': 7.32e-06,
            'batch_size': 8,
            'hidden_dim': 768,
            'n_layers': 3,
            'dropout': 0.0468,
            'input_mode': 'single',
            'activation': 'SiLU',
            'w_mse_eps': 1.6216,
            'w_sam': 1.5620,
            'w_grad': 0.3741,
            'w_curve': 0.00013,
            'w_phys': 0.02388
        }
        baseline_params_c = {
            'lr': 0.0002481,
            'optimizer_name': 'AdamW',
            'weight_decay': 3.27e-06,
            'batch_size': 8,
            'hidden_dim': 768,
            'n_layers': 3,
            'dropout': 0.0226,
            'input_mode': 'single',
            'activation': 'SiLU',
            'w_mse_eps': 1.2528,
            'w_sam': 1.4820,
            'w_grad': 0.4300,
            'w_curve': 0.0001737,
            'w_phys': 0.04599
        }
    else:
        baseline_params_b = {
            'lr': 0.0002014,
            'optimizer_name': 'AdamW',
            'weight_decay': 8.03e-06,
            'batch_size': 8,
            'hidden_dim': 384,
            'n_layers': 5,
            'dropout': 0.0116,
            'input_mode': 'triple',
            'activation': 'SiLU',
            'w_mse_eps': 1.5119,
            'w_sam': 0.7557,
            'w_grad': 0.3855,
            'w_curve': 0.05034,
            'w_phys': 0.01975
        }
        baseline_params_c = baseline_params_b

    try:
        study.enqueue_trial(baseline_params_b)
        print(f"Enqueued Run B validated SOTA configuration for {sensor.upper()} as Trial 0 reference.")
    except Exception as e:
        print(f"Note on Run B baseline trial enqueue: {e}")

    if sensor == 'hytes':
        try:
            study.enqueue_trial(baseline_params_c)
            print(f"Enqueued Run C validated SOTA configuration for {sensor.upper()} as Trial 1 reference.")
        except Exception as e:
            print(f"Note on Run C baseline trial enqueue: {e}")

    # Create Objective
    obj_fn = create_objective(
        sensor=sensor,
        train_dataset=train_data,
        val_dataset=val_data,
        device=device,
        wav_tensor=wav_tensor,
        mask=mask,
        atm_cache=atm_cache,
        priors=priors,
        norm_cache=norm_cache,
        null_rmse_e=null_rmse_e,
        n_epochs_per_trial=n_epochs,
        search_space=search_space
    )

    # Execute Optimization
    print(f"\nLaunching optimization loop ({n_trials} trials, max {n_epochs} epochs/trial)...")
    study.optimize(obj_fn, n_trials=n_trials)

    # Save summary results
    print(f"\n{'='*75}")
    print(f"Optuna Study Complete for {sensor.upper()}!")
    print(f"{'='*75}")
    print(f"Best Trial #{study.best_trial.number}")
    print(f"Best Composite Objective J: {study.best_trial.value:.4f}")
    print(f"Best Hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
    print(f"Best Performance Metrics:")
    for k, v in study.best_trial.user_attrs.items():
        print(f"  {k}: {v}")

    # Export best configuration with STRICT Stage 1 SSoT provenance
    best_config_path = os.path.join(output_dir, f'best_config_{sensor}.json')
    best_params = dict(study.best_trial.params)

    best_dict = {
        'sensor': sensor,
        'provenance': 'stage1_grounded',
        'best_trial_number': study.best_trial.number,
        'best_value': study.best_trial.value,
        'params': best_params,
        'metrics': study.best_trial.user_attrs,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(best_config_path, 'w') as f:
        json.dump(best_dict, f, indent=2)
    print(f"Saved verified best config to: {best_config_path}")

    # Export trials dataframe to CSV
    trials_df = study.trials_dataframe()
    trials_csv_path = os.path.join(output_dir, f'optuna_trials_{sensor}.csv')
    trials_df.to_csv(trials_csv_path, index=False)
    print(f"Saved trials log to:   {trials_csv_path}")

    return study


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Optuna Hyperparameter Optimization")
    parser.add_argument('--sensor', type=str, default='aster', choices=['aster', 'hytes'], help='Sensor')
    parser.add_argument('--n_trials', type=int, default=20, help='Total number of trials')
    parser.add_argument('--n_epochs', type=int, default=20, help='Max epochs per trial')
    parser.add_argument('--fold_idx', type=int, default=0, help='Development fold index')
    parser.add_argument('--search_space', type=str, default='unified', choices=['unified', 'run_c', 'run_b'], help='Search space definition')
    args = parser.parse_args()

    run_optuna_study(
        sensor=args.sensor,
        n_trials=args.n_trials,
        n_epochs=args.n_epochs,
        fold_idx=args.fold_idx,
        search_space=args.search_space
    )
