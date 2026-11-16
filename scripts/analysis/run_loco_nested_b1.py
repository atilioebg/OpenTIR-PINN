"""
Nested-HPO Leave-One-Class-Out (LOCO) Benchmark - Option B1 (Gold Standard)
==========================================================================
Executes the fully nested, zero-leakage LOCO benchmark for Q1 submission (RSE):
  1. For each held-out class, hyperparameter optimization (Optuna TPE) is run
     STRICTLY on the N-1 classes (20 trials x 15 epochs). The held-out class is
     completely absent from architecture selection, eliminating selection bias.
  2. The optimal configuration discovered for that specific N-1 subset is trained
     for 100 epochs with ReduceLROnPlateau and gradient clipping (1.0).
  3. The final model is evaluated strictly on PRISTINE held-out spectra under the
     standard Quadrant I (Q-I) evaluation grid (4 atmospheres x 11 temperatures).
  4. Point-to-point paired Classical TES benchmark is computed on the exact same
     pristine samples.
  5. Spectrum-level cluster bootstrap (1,000 resamplings) computes 95% confidence
     intervals for delta T RMSE, eps RMSE, and SAM.

Author: Dr. Atilio Grondona & OpenTIR-PINN Research Team
Target Journal: Remote Sensing of Environment (Elsevier)
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

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

VALID_LOCO_CLASSES = [
    'botanical',
    'soil',
    'sandstone',
    'quartz'
]


def evaluate_model_on_dataset(
    model: nn.Module,
    dataset: List[dict],
    device: torch.device,
    input_mode: str,
    atm_cache: dict,
    mu_x: torch.Tensor,
    std_x: torch.Tensor,
    mask: torch.Tensor,
    null_rmse_e: float,
    std_t: float,
    std_e: float,
    batch_size: int = 32
) -> Dict[str, Any]:
    """Evaluates a PINN model on a dataset and returns unified metrics and raw predictions."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_t_pred, all_t_true = [], []
    all_e_pred, all_e_true = [], []
    all_spec_ids = []

    with torch.no_grad():
        for batch in loader:
            x_in = assemble_input_tensor(batch, input_mode, atm_cache, device, mask=mask)
            x_std = standardize_input(x_in, mu_x, std_x)

            t_pred, e_pred = model(x_std)
            t_true = batch['target_temp'].to(device).view(-1, 1)
            e_true = batch['target_emissivity'].to(device)
            if mask is not None:
                e_true = e_true[:, mask]

            all_t_pred.append(t_pred.squeeze(-1).cpu())
            all_t_true.append(t_true.squeeze(-1).cpu())
            all_e_pred.append(e_pred.cpu())
            all_e_true.append(e_true.cpu())
            all_spec_ids.extend(batch['spectrum_id'])

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

    return metrics, {
        't_pred': t_pred_cat.numpy(),
        't_true': t_true_cat.numpy(),
        'e_pred': e_pred_cat.numpy(),
        'e_true': e_true_cat.numpy(),
        'spectrum_ids': all_spec_ids
    }


def compute_cluster_bootstrap_ci(
    pinn_raw: dict,
    df_tes: pd.DataFrame,
    n_resamples: int = 1000,
    seed: int = 42
) -> dict:
    """Computes 95% bootstrap confidence intervals resampled at the spectrum level."""
    rng = np.random.RandomState(seed)
    spec_ids = np.array(pinn_raw['spectrum_ids'])
    unique_spectra = np.unique(spec_ids)
    n_spectra = len(unique_spectra)

    # Index map per spectrum
    spec_to_indices = {s: np.where(spec_ids == s)[0] for s in unique_spectra}

    pinn_dt = pinn_raw['t_pred'] - pinn_raw['t_true']
    pinn_de = np.sqrt(np.mean((pinn_raw['e_pred'] - pinn_raw['e_true'])**2, axis=-1))
    
    # Cosine distance for SAM in degrees
    e_p = pinn_raw['e_pred']
    e_t = pinn_raw['e_true']
    dot = np.sum(e_p * e_t, axis=-1)
    norm_p = np.linalg.norm(e_p, axis=-1)
    norm_t = np.linalg.norm(e_t, axis=-1)
    cos_sim = np.clip(dot / (norm_p * norm_t + 1e-9), -1.0, 1.0)
    pinn_sam = np.degrees(np.arccos(cos_sim))

    tes_dt = df_tes['delta_t_k'].values
    tes_de = df_tes['rmse_emissivity'].values
    tes_sam = df_tes['sam_deg'].values

    boot_pinn_dt_rmse = []
    boot_tes_dt_rmse = []
    boot_pinn_de_rmse = []
    boot_tes_de_rmse = []
    boot_pinn_sam = []
    boot_tes_sam = []

    for _ in range(n_resamples):
        sampled_spectra = rng.choice(unique_spectra, size=n_spectra, replace=True)
        idx = np.concatenate([spec_to_indices[s] for s in sampled_spectra])

        boot_pinn_dt_rmse.append(np.sqrt(np.mean(pinn_dt[idx]**2)))
        boot_tes_dt_rmse.append(np.sqrt(np.mean(tes_dt[idx]**2)))
        boot_pinn_de_rmse.append(np.mean(pinn_de[idx]))
        boot_tes_de_rmse.append(np.mean(tes_de[idx]))
        boot_pinn_sam.append(np.mean(pinn_sam[idx]))
        boot_tes_sam.append(np.mean(tes_sam[idx]))

    return {
        'pinn_dt_rmse_ci95': [float(np.percentile(boot_pinn_dt_rmse, 2.5)), float(np.percentile(boot_pinn_dt_rmse, 97.5))],
        'tes_dt_rmse_ci95': [float(np.percentile(boot_tes_dt_rmse, 2.5)), float(np.percentile(boot_tes_dt_rmse, 97.5))],
        'pinn_de_rmse_ci95': [float(np.percentile(boot_pinn_de_rmse, 2.5)), float(np.percentile(boot_pinn_de_rmse, 97.5))],
        'tes_de_rmse_ci95': [float(np.percentile(boot_tes_de_rmse, 2.5)), float(np.percentile(boot_tes_de_rmse, 97.5))],
        'pinn_sam_ci95': [float(np.percentile(boot_pinn_sam, 2.5)), float(np.percentile(boot_pinn_sam, 97.5))],
        'tes_sam_ci95': [float(np.percentile(boot_tes_sam, 2.5)), float(np.percentile(boot_tes_sam, 97.5))]
    }


def run_nested_hpo_for_class(
    held_out_class: str,
    sensor: str,
    train_loco: List[dict],
    val_id: List[dict],
    mask: torch.Tensor,
    wav_tensor: torch.Tensor,
    atm_cache: dict,
    priors: dict,
    mu_x: torch.Tensor,
    std_x: torch.Tensor,
    id_null_rmse_e: float,
    n_trials: int = 15,
    hpo_epochs: int = 15,
    device: torch.device = torch.device('cpu')
) -> dict:
    """Runs a nested Optuna HPO study strictly on N-1 classes to eliminate selection bias."""
    n_bands_eff = int(mask.sum())
    input_mode = 'triple' if sensor == 'aster' else 'single'

    print(f"  [Nested HPO] Launching {n_trials} trials ({hpo_epochs} epochs/trial) on N-1 classes...")

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        if sensor == 'aster':
            hidden_dim = trial.suggest_categorical('hidden_dim', [256, 384, 512])
        else:
            hidden_dim = trial.suggest_categorical('hidden_dim', [384, 512, 768])
        n_layers = trial.suggest_int('n_layers', 3, 5)

        w_mse_eps = trial.suggest_float('w_mse_eps', 0.5, 2.0)
        w_sam = trial.suggest_float('w_sam', 0.05, 0.5)

        if sensor == 'hytes':
            w_grad = trial.suggest_float('w_grad', 1e-4, 0.02, log=True)
            w_curve = trial.suggest_float('w_curve', 1e-4, 1e-2, log=True)
        else:
            w_grad = trial.suggest_float('w_grad', 1e-4, 0.05, log=True)
            w_curve = trial.suggest_float('w_curve', 0.0, 0.20)

        w_phys = trial.suggest_float('w_phys', 0.01, 0.10, log=True)

        trial_cfg = {
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'input_mode': input_mode,
            'activation': 'silu',
            'dropout': 0.0,
            'w_mse_eps': w_mse_eps,
            'w_mse_temp': 1.0,
            'w_sam': w_sam,
            'w_grad': w_grad,
            'w_curve': w_curve,
            'w_phys': w_phys
        }

        model = build_stage1_pinn(
            sensor=sensor,
            n_bands_eff=n_bands_eff,
            config=trial_cfg,
            mu_t=priors['mu_t'],
            mu_e=priors['mu_e'],
            device=device
        )

        criterion = build_stage1_criterion(
            sensor=sensor,
            config=trial_cfg,
            var_t=priors['var_t'],
            var_e=priors['var_e'],
            device=device
        )

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        loader_train = DataLoader(train_loco, batch_size=batch_size, shuffle=True)

        for ep in range(hpo_epochs):
            criterion.w_phys = 0.0 if ep == 0 else w_phys
            model.train()
            for b in loader_train:
                x_in = assemble_input_tensor(b, input_mode, atm_cache, device, mask=mask)
                x_std = standardize_input(x_in, mu_x, std_x)
                t_pred, e_pred = model(x_std)

                t_true = b['target_temp'].to(device).view(-1, 1)
                e_true = b['target_emissivity'].to(device)
                if mask is not None:
                    e_true = e_true[:, mask]

                l_surf = assemble_input_tensor(b, 'single', atm_cache, device, mask=mask)[:, :n_bands_eff]
                l_down = assemble_input_tensor(b, 'triple', atm_cache, device, mask=mask)[:, 2*n_bands_eff:]
                loss = criterion(
                    t_pred=t_pred,
                    t_true=t_true,
                    eps_pred=e_pred,
                    eps_true=e_true,
                    l_surf_obs=l_surf,
                    l_down=l_down,
                    wavelengths=wav_tensor[mask]
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Evaluation for pruning
            val_metrics, _ = evaluate_model_on_dataset(
                model=model,
                dataset=val_id,
                device=device,
                input_mode=input_mode,
                atm_cache=atm_cache,
                mu_x=mu_x,
                std_x=std_x,
                mask=mask,
                null_rmse_e=id_null_rmse_e,
                std_t=priors['std_t'],
                std_e=priors['std_e'],
                batch_size=batch_size
            )

            score = val_metrics['composite_score']
            trial.report(score, step=ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return score

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    )
    study.optimize(objective, n_trials=n_trials)

    best_p = study.best_params
    best_config = {
        'hidden_dim': best_p['hidden_dim'],
        'n_layers': best_p['n_layers'],
        'input_mode': input_mode,
        'activation': 'silu',
        'dropout': 0.0,
        'lr': best_p['lr'],
        'batch_size': best_p['batch_size'],
        'w_mse_eps': best_p['w_mse_eps'],
        'w_mse_temp': 1.0,
        'w_sam': best_p['w_sam'],
        'w_grad': best_p['w_grad'],
        'w_curve': best_p['w_curve'],
        'w_phys': best_p['w_phys'],
        'best_val_score': study.best_value
    }
    print(f"  [Nested HPO] Best config for N-1 (excluding {held_out_class}): dim={best_config['hidden_dim']}, layers={best_config['n_layers']}, lr={best_config['lr']:.2e}, val_score={study.best_value:.4f}")
    return best_config


def execute_b1_for_class(
    held_out_class: str,
    sensor: str,
    train_data_full: List[dict],
    pristine_qi_samples: List[dict],
    n_trials: int,
    hpo_epochs: int,
    prod_epochs: int,
    device: torch.device
) -> dict:
    """Executes the full Option B1 protocol for a single held-out class."""
    print(f"\n{'='*75}")
    print(f"Option B1 LOCO Suite: Holding Out [{held_out_class.upper()}] | Sensor: {sensor.upper()}")
    print(f"{'='*75}")

    start_time = time.time()
    atm_cache = build_atmospheric_band_cache(sensor)
    wav_tensor = get_sensor_wavelengths(sensor, device)
    mask = get_sensor_band_mask(sensor, train_data_full, device=device)
    n_bands_eff = int(mask.sum())

    # 1. Filter training data strictly to N-1 classes
    n_minus_1_train = [s for s in train_data_full if s['material_class'] != held_out_class]

    # Group-aware 85/15 split for early stopping and HPO validation
    group_map = {}
    for s in n_minus_1_train:
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

    # Pristine held-out evaluation samples
    test_held_out = [s for s in pristine_qi_samples if s['material_class'] == held_out_class]
    n_test_spectra = len(set(s['spectrum_id'] for s in test_held_out))

    print(f"  Train samples (N-1 classes): {len(train_loco)} across {len(train_groups)} groups")
    print(f"  In-dist val samples (N-1):   {len(val_id)} across {len(val_groups)} groups")
    print(f"  Pristine Q-I Held-Out Test:  {len(test_held_out)} samples ({n_test_spectra} unique spectra)")

    # Priors and normalization strictly on train_loco
    priors = extract_training_priors(train_loco, mask, device=device)
    input_mode = 'triple' if sensor == 'aster' else 'single'
    mu_x, std_x = compute_input_norm_stats(train_loco, input_mode, atm_cache, mask, device=device)
    id_null_rmse_e = compute_like_for_like_null_rmse(val_id, priors['mu_e'], mask)
    test_null_rmse_e = compute_like_for_like_null_rmse(test_held_out, priors['mu_e'], mask)

    # 2. Run Nested HPO strictly on N-1
    best_config = run_nested_hpo_for_class(
        held_out_class=held_out_class,
        sensor=sensor,
        train_loco=train_loco,
        val_id=val_id,
        mask=mask,
        wav_tensor=wav_tensor,
        atm_cache=atm_cache,
        priors=priors,
        mu_x=mu_x,
        std_x=std_x,
        id_null_rmse_e=id_null_rmse_e,
        n_trials=n_trials,
        hpo_epochs=hpo_epochs,
        device=device
    )

    # 3. Train Production Model for 100 Epochs with Best Config
    print(f"  [Production Training] Training 100 epochs with nested-discovered config...")
    model = build_stage1_pinn(
        sensor=sensor,
        n_bands_eff=n_bands_eff,
        config=best_config,
        mu_t=priors['mu_t'],
        mu_e=priors['mu_e'],
        device=device
    )

    criterion = build_stage1_criterion(
        sensor=sensor,
        config=best_config,
        var_t=priors['var_t'],
        var_e=priors['var_e'],
        device=device
    )

    optimizer = optim.AdamW(model.parameters(), lr=best_config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    train_loader = DataLoader(train_loco, batch_size=best_config['batch_size'], shuffle=True)

    best_val_score = float('inf')
    best_model_state = None

    for epoch in range(1, prod_epochs + 1):
        criterion.w_phys = 0.0 if epoch == 1 else best_config['w_phys']
        model.train()
        for batch in train_loader:
            x_in = assemble_input_tensor(batch, input_mode, atm_cache, device, mask=mask)
            x_std = standardize_input(x_in, mu_x, std_x)

            t_pred, e_pred = model(x_std)
            t_true = batch['target_temp'].to(device).view(-1, 1)
            e_true = batch['target_emissivity'].to(device)
            if mask is not None:
                e_true = e_true[:, mask]

            l_surf = assemble_input_tensor(batch, 'single', atm_cache, device, mask=mask)[:, :n_bands_eff]
            l_down = assemble_input_tensor(batch, 'triple', atm_cache, device, mask=mask)[:, 2*n_bands_eff:]
            loss = criterion(
                t_pred=t_pred,
                t_true=t_true,
                eps_pred=e_pred,
                eps_true=e_true,
                l_surf_obs=l_surf,
                l_down=l_down,
                wavelengths=wav_tensor[mask]
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation on in-distribution N-1
        id_metrics, _ = evaluate_model_on_dataset(
            model=model,
            dataset=val_id,
            device=device,
            input_mode=input_mode,
            atm_cache=atm_cache,
            mu_x=mu_x,
            std_x=std_x,
            mask=mask,
            null_rmse_e=id_null_rmse_e,
            std_t=priors['std_t'],
            std_e=priors['std_e'],
            batch_size=best_config['batch_size']
        )

        comp_score = id_metrics['composite_score']
        scheduler.step(comp_score)

        if comp_score < best_val_score:
            best_val_score = comp_score
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == prod_epochs:
            print(f"    Epoch [{epoch:03d}/{prod_epochs}] Val Comp: {comp_score:.4f} | dT RMSE: {id_metrics['temp_rmse']:.2f} K | dE RMSE: {id_metrics['eps_rmse']:.4f}")

    # Load best checkpoint
    model.load_state_dict(best_model_state)

    # 4. Final PINN evaluation on PRISTINE held-out Q-I test samples
    print(f"  Evaluating final model on {len(test_held_out)} PRISTINE held-out samples...")
    pinn_ood, pinn_raw = evaluate_model_on_dataset(
        model=model,
        dataset=test_held_out,
        device=device,
        input_mode=input_mode,
        atm_cache=atm_cache,
        mu_x=mu_x,
        std_x=std_x,
        mask=mask,
        null_rmse_e=test_null_rmse_e,
        std_t=priors['std_t'],
        std_e=priors['std_e'],
        batch_size=best_config['batch_size']
    )

    # 5. Paired Classical TES on the EXACT same pristine samples
    print(f"  Evaluating paired Classical TES on {len(test_held_out)} PRISTINE samples...")
    tes = ClassicalTES(sensor=sensor)
    df_tes = tes.evaluate_dataset(test_held_out, dataset_name=f'loco_{held_out_class}')
    tes_temp_rmse = float(np.sqrt((df_tes['delta_t_k']**2).mean()))
    tes_temp_mae = float(df_tes['abs_delta_t_k'].mean())
    tes_eps_rmse = float(df_tes['rmse_emissivity'].mean())
    tes_sam_deg = float(df_tes['sam_deg'].mean())

    # 6. Cluster Bootstrap 95% Confidence Intervals
    print(f"  Computing 1,000 spectrum-level cluster bootstrap resamplings...")
    ci95 = compute_cluster_bootstrap_ci(pinn_raw, df_tes, n_resamples=1000)

    # 7. Save Models and Metadata
    models_dir = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'loco_b1', sensor)
    os.makedirs(models_dir, exist_ok=True)
    loco_model_path = os.path.join(models_dir, f'model_{sensor}_loco_{held_out_class}.pth')
    torch.save({
        'held_out_class': held_out_class,
        'sensor': sensor,
        'config': best_config,
        'model_state_dict': best_model_state,
        'norm_stats': {'mu_x': mu_x.cpu(), 'std_x': std_x.cpu(), 'mu_t': priors['mu_t'], 'std_t': priors['std_t'], 'mu_e': priors['mu_e'].cpu(), 'std_e': priors['std_e']},
        'band_mask': mask.cpu(),
        'pinn_metrics': pinn_ood,
        'tes_metrics': {
            'temp_rmse': tes_temp_rmse,
            'temp_mae': tes_temp_mae,
            'eps_rmse': tes_eps_rmse,
            'sam_deg': tes_sam_deg
        },
        'ci95': ci95
    }, loco_model_path)

    wall_clock = time.time() - start_time
    delta_t_impr_pct = (tes_temp_rmse - pinn_ood['temp_rmse']) / tes_temp_rmse * 100.0
    delta_e_impr_pct = (tes_eps_rmse - pinn_ood['eps_rmse']) / tes_eps_rmse * 100.0

    print(f"\n--- B1 LOCO Summary: Held-out Class [{held_out_class.upper()}] ---")
    print(f"  PINN dT RMSE: {pinn_ood['temp_rmse']:.2f} K [{ci95['pinn_dt_rmse_ci95'][0]:.2f}, {ci95['pinn_dt_rmse_ci95'][1]:.2f}] | TES: {tes_temp_rmse:.2f} K [{ci95['tes_dt_rmse_ci95'][0]:.2f}, {ci95['tes_dt_rmse_ci95'][1]:.2f}] | Gain: {delta_t_impr_pct:+.1f} %")
    print(f"  PINN dE RMSE: {pinn_ood['eps_rmse']:.4f} [{ci95['pinn_de_rmse_ci95'][0]:.4f}, {ci95['pinn_de_rmse_ci95'][1]:.4f}] | TES: {tes_eps_rmse:.4f} [{ci95['tes_de_rmse_ci95'][0]:.4f}, {ci95['tes_de_rmse_ci95'][1]:.4f}] | Gain: {delta_e_impr_pct:+.1f} %")
    print(f"  PINN SAM:     {pinn_ood['sam_deg']:.2f}° [{ci95['pinn_sam_ci95'][0]:.2f}, {ci95['pinn_sam_ci95'][1]:.2f}] | TES: {tes_sam_deg:.2f}° [{ci95['tes_sam_ci95'][0]:.2f}, {ci95['tes_sam_ci95'][1]:.2f}]")
    print(f"  Wall time:    {wall_clock/60:.1f} min\n")

    return {
        'sensor': sensor,
        'held_out_class': held_out_class,
        'n_samples': len(test_held_out),
        'n_spectra': n_test_spectra,
        'pinn_temp_rmse': pinn_ood['temp_rmse'],
        'pinn_temp_rmse_ci95_low': ci95['pinn_dt_rmse_ci95'][0],
        'pinn_temp_rmse_ci95_high': ci95['pinn_dt_rmse_ci95'][1],
        'tes_temp_rmse': tes_temp_rmse,
        'tes_temp_rmse_ci95_low': ci95['tes_dt_rmse_ci95'][0],
        'tes_temp_rmse_ci95_high': ci95['tes_dt_rmse_ci95'][1],
        'lst_rmse_impr_pct': delta_t_impr_pct,
        'pinn_temp_mae': pinn_ood['temp_mae'],
        'tes_temp_mae': tes_temp_mae,
        'pinn_eps_rmse': pinn_ood['eps_rmse'],
        'pinn_eps_rmse_ci95_low': ci95['pinn_de_rmse_ci95'][0],
        'pinn_eps_rmse_ci95_high': ci95['pinn_de_rmse_ci95'][1],
        'tes_eps_rmse': tes_eps_rmse,
        'tes_eps_rmse_ci95_low': ci95['tes_de_rmse_ci95'][0],
        'tes_eps_rmse_ci95_high': ci95['tes_de_rmse_ci95'][1],
        'eps_rmse_impr_pct': delta_e_impr_pct,
        'pinn_sam_deg': pinn_ood['sam_deg'],
        'pinn_sam_ci95_low': ci95['pinn_sam_ci95'][0],
        'pinn_sam_ci95_high': ci95['pinn_sam_ci95'][1],
        'tes_sam_deg': tes_sam_deg,
        'tes_sam_ci95_low': ci95['tes_sam_ci95'][0],
        'tes_sam_ci95_high': ci95['tes_sam_ci95'][1],
        'wall_clock_s': wall_clock,
        'optuna_trials': n_trials,
        'hpo_epochs': hpo_epochs,
        'prod_epochs': prod_epochs,
        'chosen_dim': best_config['hidden_dim'],
        'chosen_layers': best_config['n_layers'],
        'chosen_lr': best_config['lr']
    }


def main():
    parser = argparse.ArgumentParser(description="Option B1 Nested HPO LOCO Benchmark")
    parser.add_argument('--sensor', type=str, default='all', choices=['aster', 'hytes', 'all'])
    parser.add_argument('--n_trials', type=int, default=10, help='Optuna trials per class')
    parser.add_argument('--hpo_epochs', type=int, default=15, help='Epochs per Optuna trial')
    parser.add_argument('--prod_epochs', type=int, default=100, help='Epochs for production LOCO model')
    args = parser.parse_args()

    sensors = ['aster', 'hytes'] if args.sensor == 'all' else [args.sensor]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Option B1 Gold-Standard LOCO Benchmark")
    print(f"Sensors: {sensors} | Device: {device} | Optuna Trials: {args.n_trials} | Prod Epochs: {args.prod_epochs}")

    bench_dir = os.path.join(PROJECT_ROOT, 'outputs', 'benchmarks', 'loco_b1')
    os.makedirs(bench_dir, exist_ok=True)
    master_csv_path = os.path.join(bench_dir, 'loco_b1_master.csv')

    all_records = []
    if os.path.exists(master_csv_path):
        try:
            df_existing = pd.read_csv(master_csv_path)
            all_records = df_existing.to_dict(orient='records')
            print(f"Found existing master CSV with {len(all_records)} records.")
        except Exception:
            all_records = []

    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets')

    for sens in sensors:
        train_path = os.path.join(data_dir, f'train_fold0_{sens}.pt')
        train_data_full = torch.load(train_path, weights_only=False, map_location='cpu')

        # Load pristine Q-I evaluation union
        pristine_qi_samples = []
        for k in range(5):
            val_path = os.path.join(data_dir, f'val_fold{k}_{sens}.pt')
            pristine_qi_samples.extend(torch.load(val_path, weights_only=False, map_location='cpu'))

        for cls_name in VALID_LOCO_CLASSES:
            # Check if already completed
            already_done = any(r['sensor'] == sens and r['held_out_class'] == cls_name for r in all_records)
            if already_done:
                print(f"Skipping {sens.upper()} - {cls_name.upper()} (already completed in master CSV)")
                continue

            record = execute_b1_for_class(
                held_out_class=cls_name,
                sensor=sens,
                train_data_full=train_data_full,
                pristine_qi_samples=pristine_qi_samples,
                n_trials=args.n_trials,
                hpo_epochs=args.hpo_epochs,
                prod_epochs=args.prod_epochs,
                device=device
            )

            all_records.append(record)
            df_current = pd.DataFrame(all_records)
            df_current.to_csv(master_csv_path, index=False)
            print(f"Saved checkpoint to {master_csv_path}")

    print("\nOption B1 LOCO Benchmark Complete!")
    print(f"Master ledger saved at: {master_csv_path}")


if __name__ == '__main__':
    main()
