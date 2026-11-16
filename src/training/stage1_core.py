"""
Stage 1 Core Shared Training, Standardization & Metrics Framework
================================================================
Single Source of Truth (SSoT) module providing unified:
  1. Sensor wavelength and telluric opaque band mask loading (223 clean bands for HyTES).
  2. Training-fold empirical prior extraction (mu_t, std_t, var_t, mu_e, std_e, var_e).
  3. Input radiance standardization (mu_x, std_x) strictly computed on training folds.
  4. Analytical prior head initialization preventing sigmoid saturation (F6 fix).
  5. Like-for-like null baseline computation (per-sample RMSE averaged across samples).
  6. Unit-free balanced multi-task loss configuration with physical Sobolev scaling.
  7. Unified validation & benchmark metric evaluation.

Author: Dr. Atilio Grondona & OpenTIR-PINN Research Team
Target Journal: IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import SENSOR_CONFIG
from src.models.pinn import PhysicsInformedNN
from src.models.losses import SpectralShapeLoss
from src.training.input_utils import assemble_input_tensor, compute_spectral_angle_mapper


def get_sensor_wavelengths(sensor: str, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Returns band center wavelengths as a 1D PyTorch tensor."""
    sensor = sensor.lower()
    if sensor == 'aster':
        centers = [8.281, 8.633, 9.079, 10.662, 11.293]
    elif sensor == 'hytes':
        param_path = SENSOR_CONFIG['hytes']['processed_params_file']
        df = pd.read_csv(param_path).sort_values('band_idx')
        centers = df['center_wavelength'].values
    else:
        raise ValueError(f"Unknown sensor: {sensor}")
    return torch.tensor(centers, dtype=torch.float32, device=device)


def get_sensor_band_mask(
    sensor: str,
    sample_or_dataset: Optional[Any] = None,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Returns boolean band mask.
    For HyTES: 223 transparent channels (33 telluric opaque bands masked out).
    For ASTER: all 5 bands are transparent.
    """
    sensor = sensor.lower()
    if sensor == 'aster':
        return torch.ones(5, dtype=torch.bool, device=device)

    # 1. From sample dict or dataset list
    if sample_or_dataset is not None:
        item = sample_or_dataset[0] if isinstance(sample_or_dataset, (list, tuple)) and len(sample_or_dataset) > 0 else sample_or_dataset
        if isinstance(item, dict) and 'band_mask' in item and item['band_mask'] is not None:
            raw_mask = item['band_mask']
            if isinstance(raw_mask, torch.Tensor):
                return raw_mask.bool().to(device)
            return torch.tensor(raw_mask, dtype=torch.bool, device=device)

    # 2. From processed dataset file fallback
    fallback_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets', 'val_fold0_hytes.pt')
    if os.path.exists(fallback_path):
        try:
            sample = torch.load(fallback_path, weights_only=False, map_location='cpu')[0]
            if 'band_mask' in sample and sample['band_mask'] is not None:
                return sample['band_mask'].bool().to(device)
        except Exception:
            pass

    return torch.ones(256, dtype=torch.bool, device=device)


def extract_training_priors(
    train_data: list,
    mask: torch.Tensor,
    device: torch.device = torch.device('cpu')
) -> dict:
    """
    Computes empirical training-fold statistics strictly leak-free.
    Returns: mu_t, std_t, var_t, mu_e, std_e, var_e.
    """
    all_t_train = torch.stack([s['target_temp'] for s in train_data])
    all_e_train = torch.stack([s['target_emissivity'] for s in train_data])
    if mask is not None:
        all_e_train = all_e_train[:, mask.cpu()]

    mu_t = float(all_t_train.mean())
    std_t = float(all_t_train.std().clamp_min(1e-4))
    var_t = float(all_t_train.var().clamp_min(1e-8))

    mu_e = all_e_train.mean(dim=0).to(device)
    std_e = float(all_e_train.std(dim=0).mean().clamp_min(1e-4))
    var_e = float(all_e_train.var(dim=0).mean().clamp_min(1e-8))

    return {
        'mu_t': mu_t,
        'std_t': std_t,
        'var_t': var_t,
        'mu_e': mu_e,
        'std_e': std_e,
        'var_e': var_e
    }


def compute_input_norm_stats(
    train_data: list,
    input_mode: str,
    atm_cache: dict,
    mask: torch.Tensor,
    device: torch.device = torch.device('cpu'),
    batch_size: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes per-feature mean and standard deviation of input tensors strictly on training fold.
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    sample_x_list = []
    with torch.no_grad():
        for batch in loader:
            x_b = assemble_input_tensor(batch, input_mode, atm_cache, device, mask=mask)
            sample_x_list.append(x_b)
        all_x = torch.cat(sample_x_list, dim=0)
        mu_x = all_x.mean(dim=0)
        std_x = all_x.std(dim=0).clamp_min(1e-6)
    return mu_x, std_x


def standardize_input(x: torch.Tensor, mu_x: torch.Tensor, std_x: torch.Tensor) -> torch.Tensor:
    """Standardizes input tensor using training statistics."""
    return (x - mu_x) / std_x


def compute_like_for_like_null_rmse(
    val_data: list,
    mu_e_train: torch.Tensor,
    mask: torch.Tensor
) -> float:
    """
    Computes constant training-mean Null Baseline like-for-like with model metric:
    Mean across samples of per-sample RMSE across spectral bands.
    """
    all_e_val = torch.stack([s['target_emissivity'] for s in val_data])
    if mask is not None:
        all_e_val = all_e_val[:, mask.cpu()]
    mu_e_cpu = mu_e_train.cpu().unsqueeze(0)
    diff = mu_e_cpu - all_e_val
    null_rmse = float(torch.sqrt(torch.mean(diff**2, dim=-1)).mean())
    return null_rmse


def get_stage1_default_config(sensor: str) -> dict:
    """Returns verified grounded Stage 1 default hyperparameters."""
    sensor = sensor.lower()
    return {
        'lr': 1e-4,
        'optimizer_name': 'Adam',
        'batch_size': 32,
        'hidden_dim': 512,
        'n_layers': 4,
        'activation': 'GELU',
        'dropout': 0.0,
        'w_mse_eps': 1.0,      # Unit-free normalized MSE (order 1)
        'w_mse_temp': 1.0,     # Unit-free normalized MSE (order 1)
        'w_sam': 1.0,
        'w_grad': 0.5,
        'w_curve': 0.1 if sensor == 'aster' else 0.005,  # 0.005 for HyTES calibrated for 1/Delta_lambda^2
        'w_phys': 0.05,        # 0.05 default to avoid unphysical distortion under PWV noise
        'input_mode': 'dual'
    }


def build_stage1_pinn(
    sensor: str,
    n_bands_eff: int,
    config: dict,
    mu_t: float,
    mu_e: torch.Tensor,
    device: torch.device = torch.device('cpu')
) -> PhysicsInformedNN:
    """Constructs Sobolev-PINN and initializes output heads with training priors."""
    model = PhysicsInformedNN(
        input_dim=n_bands_eff,
        output_dim=n_bands_eff,
        hidden_dim=int(config.get('hidden_dim', 512)),
        n_layers=int(config.get('n_layers', 4)),
        input_mode=str(config.get('input_mode', 'dual')),
        activation=str(config.get('activation', 'GELU')),
        dropout=float(config.get('dropout', 0.0))
    ).to(device)

    model.initialize_heads(mean_temp=mu_t, mean_emissivity=mu_e)
    return model


def build_stage1_criterion(
    sensor: str,
    config: dict,
    var_t: float,
    var_e: float,
    device: torch.device = torch.device('cpu')
) -> SpectralShapeLoss:
    """Constructs balanced unit-free SpectralShapeLoss."""
    sensor = sensor.lower()
    return SpectralShapeLoss(
        w_mse_eps=float(config.get('w_mse_eps', 1.0)),
        w_mse_temp=float(config.get('w_mse_temp', 1.0)),
        w_sam=float(config.get('w_sam', 1.0)),
        w_grad=float(config.get('w_grad', 0.5)),
        w_curvature=float(config.get('w_curve', 0.1 if sensor == 'aster' else 0.005)),
        w_phys=float(config.get('w_phys', 0.05)),
        var_temp=var_t,
        var_eps=var_e,
        norm_phys_floor=1.6,
        use_bt_phys=True
    ).to(device)


def compute_spectral_correlation(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes mean per-sample Pearson correlation coefficient between predicted and ground-truth spectra.
    """
    p_mean = pred.mean(dim=-1, keepdim=True)
    t_mean = target.mean(dim=-1, keepdim=True)
    p_centered = pred - p_mean
    t_centered = target - t_mean

    numerator = torch.sum(p_centered * t_centered, dim=-1)
    denom = torch.sqrt(torch.sum(p_centered**2, dim=-1) * torch.sum(t_centered**2, dim=-1)).clamp_min(1e-8)
    corr = numerator / denom
    return float(corr.mean().item())


def compute_unified_metrics(
    t_pred: torch.Tensor,
    t_true: torch.Tensor,
    e_pred: torch.Tensor,
    e_true: torch.Tensor,
    null_rmse_e: float,
    std_t: float,
    std_e: float
) -> dict:
    """
    Computes unified benchmark metrics across temperature and emissivity predictions.
    """
    dt = t_pred - t_true
    temp_rmse = float(torch.sqrt(torch.mean(dt**2)).item())
    temp_mae = float(torch.mean(torch.abs(dt)).item())
    temp_bias = float(torch.mean(dt).item())

    de = e_pred - e_true
    # Per-sample RMSE across bands, then averaged across samples
    per_sample_rmse_e = torch.sqrt(torch.mean(de**2, dim=-1))
    eps_rmse = float(per_sample_rmse_e.mean().item())
    eps_mae = float(torch.mean(torch.abs(de)).item())
    eps_bias = float(torch.mean(de).item())

    # Gain over null
    gain_pct = ((null_rmse_e - eps_rmse) / null_rmse_e) * 100.0 if null_rmse_e > 0 else 0.0

    # Shape metrics
    sam_deg = float(compute_spectral_angle_mapper(e_pred, e_true).mean().item())
    spectral_corr = compute_spectral_correlation(e_pred, e_true)

    # Unit-free composite score
    composite_score = (temp_rmse / std_t) + (eps_rmse / std_e)

    return {
        'temp_rmse': temp_rmse,
        'temp_mae': temp_mae,
        'temp_bias': temp_bias,
        'eps_rmse': eps_rmse,
        'eps_mae': eps_mae,
        'eps_bias': eps_bias,
        'null_eps_rmse': null_rmse_e,
        'eps_gain_over_null_pct': gain_pct,
        'sam_deg': sam_deg,
        'spectral_corr': spectral_corr,
        'composite_score': composite_score
    }
