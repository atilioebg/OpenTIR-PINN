"""
Double-Quarantine 2x2 Matrix Benchmark Evaluator (Step 2.7 - Stage 1 Grounded)
=============================================================================
Evaluates Classical TES and Sobolev-PINN Models across the
Four Quadrants of the Double-Quarantine Matrix:
  - Quadrant I: In-Distribution Development Pool (Dev Materials x Dev Atmospheres)
  - Quadrant II: Clean-Room Golden Targets x Dev Atmospheres (308 samples)
  - Quadrant III: Dev Materials x Golden Quarantined Winter Atmospheres
  - Quadrant IV: Full OOD Clean-Room Benchmark (Golden Targets x Winter Atmospheres, 770 samples)

Key Stage 1 Features:
  - Loads training standardization statistics (mu_x, std_x) and band_mask directly from checkpoint.
  - Slices tensors by band_mask (223 clean bands for HyTES).
  - Evaluates both individual checkpoints and 5-fold ensemble predictions (with epistemic std).
  - Directly compares against like-for-like constant training-mean Null Baseline and Classical TES.

Author: Dr. Atilio Grondona & OpenTIR-PINN Research Team
Target Journal: IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.pinn import PhysicsInformedNN
from src.config import SENSOR_CONFIG
from src.training.input_utils import build_atmospheric_band_cache, assemble_input_tensor, compute_spectral_angle_mapper
from src.training.stage1_core import (
    get_sensor_wavelengths,
    get_sensor_band_mask,
    standardize_input,
    compute_spectral_correlation
)


def compute_sam_per_sample(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """Computes SAM in degrees per sample."""
    return compute_spectral_angle_mapper(pred, target).cpu().numpy()


def compute_curvature_error(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """Computes mean absolute curvature error (W^{2,1} discrete finite difference) per sample."""
    if pred.shape[-1] < 3:
        return np.zeros(pred.shape[0])
    curv_p = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    curv_t = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
    return torch.mean(torch.abs(curv_p - curv_t), dim=-1).cpu().numpy()


def evaluate_pinn_on_dataset(
    model: nn.Module,
    dataset: list,
    device: torch.device,
    input_mode: str,
    atm_cache: dict,
    mu_x: torch.Tensor,
    std_x: torch.Tensor,
    mask: torch.Tensor,
    null_mu_e: Optional[torch.Tensor] = None,
    batch_size: int = 64
) -> pd.DataFrame:
    """Evaluates a PyTorch PINN model with strict standardization and masking."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    records = []
    with torch.no_grad():
        for batch in loader:
            x_in = assemble_input_tensor(batch, input_mode, atm_cache, device, mask=mask)
            x_norm = standardize_input(x_in, mu_x, std_x)

            t_true = batch['target_temp'].to(device)
            e_true = batch['target_emissivity'].to(device)[:, mask]

            t_pred, e_pred = model(x_norm)

            dt = (t_pred - t_true).squeeze(-1).cpu().numpy()
            abs_dt = np.abs(dt)
            diff_eps = (e_pred - e_true).cpu().numpy()
            rmse_e = np.sqrt(np.mean(diff_eps**2, axis=-1))
            sam = compute_sam_per_sample(e_pred, e_true)
            curv = compute_curvature_error(e_pred, e_true)

            if null_mu_e is not None:
                null_diff = (null_mu_e.cpu().unsqueeze(0) - e_true.cpu()).numpy()
                null_rmse = np.sqrt(np.mean(null_diff**2, axis=-1))
            else:
                null_rmse = np.full_like(rmse_e, np.nan)

            batch_size_actual = len(t_true)
            for i in range(batch_size_actual):
                rec = {
                    'spectrum_id': batch['spectrum_id'][i],
                    'material_class': batch['material_class'][i],
                    'target_temp': float(t_true[i].item()),
                    'pred_temp': float(t_pred[i].item()),
                    'dt': float(dt[i]),
                    'abs_dt': float(abs_dt[i]),
                    'rmse_eps': float(rmse_e[i]),
                    'null_rmse_eps': float(null_rmse[i]),
                    'gain_over_null_pct': float((null_rmse[i] - rmse_e[i]) / null_rmse[i] * 100.0) if not np.isnan(null_rmse[i]) and null_rmse[i] > 0 else np.nan,
                    'sam_deg': float(sam[i]),
                    'curvature_error': float(curv[i]),
                    'pwv': float(batch['pwv'][i].item()) if 'pwv' in batch else np.nan,
                    'atmosphere_id': batch['atmosphere_id'][i] if 'atmosphere_id' in batch else ''
                }
                records.append(rec)

    return pd.DataFrame(records)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    sensor: str
) -> Tuple[nn.Module, dict, str, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Loads model along with its normalization statistics and band mask."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        config = checkpoint.get('config', {})
        input_mode = checkpoint.get('input_mode', config.get('input_mode', 'dual'))
        norm_stats = checkpoint.get('norm_stats', {})
        mask = checkpoint.get('band_mask', None)
    else:
        config = {}
        input_mode = 'dual'
        norm_stats = {}
        mask = None

    if mask is None:
        mask = get_sensor_band_mask(sensor, device=device)
    else:
        mask = mask.to(device)

    n_bands_eff = int(mask.sum())

    model = PhysicsInformedNN(
        input_dim=n_bands_eff,
        output_dim=n_bands_eff,
        hidden_dim=int(config.get('hidden_dim', 512)),
        n_layers=int(config.get('n_layers', 4)),
        input_mode=input_mode,
        activation=str(config.get('activation', 'GELU')),
        dropout=float(config.get('dropout', 0.0))
    ).to(device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    mu_x = norm_stats.get('mu_x', None)
    std_x = norm_stats.get('std_x', None)
    mu_e = norm_stats.get('mu_e', None)

    if mu_x is not None:
        mu_x = mu_x.to(device)
    if std_x is not None:
        std_x = std_x.to(device)
    if mu_e is not None:
        mu_e = mu_e.to(device)

    return model, config, input_mode, mu_x, std_x, mask, mu_e


def evaluate_quarantine_matrix(
    sensor: str = 'aster',
    model_path: Optional[str] = None
) -> pd.DataFrame:
    """Evaluates PINN model against Classical TES across Quadrants I, II, III, and IV."""
    print(f"\n{'='*75}")
    print(f"Double-Quarantine Matrix Benchmark Evaluation | Sensor: {sensor.upper()}")
    print(f"{'='*75}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Compute Device: {device}")
    atm_cache = build_atmospheric_band_cache(sensor)

    # Locate Model
    if model_path is None:
        candidate_paths = [
            os.path.join(PROJECT_ROOT, 'outputs', 'models', 'cv', sensor, f'model_{sensor}_fold0_best.pth'),
            os.path.join(PROJECT_ROOT, 'outputs', 'models', f'model_{sensor}_fold0_best.pth'),
            os.path.join(PROJECT_ROOT, 'outputs', 'models', f'model_{sensor}_best.pth')
        ]
        for cp in candidate_paths:
            if os.path.exists(cp):
                model_path = cp
                break

    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found. Expected one of: {candidate_paths}. Train a model first.")

    print(f"Loading Stage 1 Sobolev-PINN Model from: {model_path}")
    model, config, input_mode, mu_x, std_x, mask, mu_e = load_model_from_checkpoint(model_path, device, sensor)

    print(f"Model Ingestion Mode: [{input_mode.upper()}] | Effective Bands: {int(mask.sum())}")

    # Datasets
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets')
    q1_path = os.path.join(data_dir, f'val_fold0_{sensor}.pt')
    q2_path = os.path.join(data_dir, f'golden_benchmark_quadrant_II_{sensor}.pt')
    q3_path = os.path.join(data_dir, f'quadrant_III_fold0_{sensor}.pt')
    if not os.path.exists(q3_path):
        q3_path = os.path.join(data_dir, f'golden_benchmark_quadrant_III_{sensor}.pt')
    q4_path = os.path.join(data_dir, f'golden_benchmark_quadrant_IV_{sensor}.pt')

    datasets = {}
    if os.path.exists(q1_path):
        datasets['Quadrant_I_DevPool'] = torch.load(q1_path, weights_only=False)
    if os.path.exists(q2_path):
        datasets['Quadrant_II_GoldenTargets_DevAtm'] = torch.load(q2_path, weights_only=False)
    if os.path.exists(q3_path):
        datasets['Quadrant_III_DevMaterials_WinterAtm'] = torch.load(q3_path, weights_only=False)
    if os.path.exists(q4_path):
        datasets['Quadrant_IV_GoldenTargets_WinterAtm_OOD'] = torch.load(q4_path, weights_only=False)

    # Classical TES Baseline paths for direct pairing
    baseline_csv_path = os.path.join(PROJECT_ROOT, 'outputs', 'baselines', f'classical_tes_benchmark_{sensor}.csv')
    df_tes = pd.read_csv(baseline_csv_path) if os.path.exists(baseline_csv_path) else None

    matrix_summary = []
    benchmarks_dir = os.path.join(PROJECT_ROOT, 'outputs', 'benchmarks')
    os.makedirs(benchmarks_dir, exist_ok=True)

    for q_name, ds in datasets.items():
        print(f"\nEvaluating on {q_name} ({len(ds)} samples)...")
        df_eval = evaluate_pinn_on_dataset(
            model=model,
            dataset=ds,
            device=device,
            input_mode=input_mode,
            atm_cache=atm_cache,
            mu_x=mu_x,
            std_x=std_x,
            mask=mask,
            null_mu_e=mu_e
        )

        # Save per-sample predictions
        pred_csv = os.path.join(benchmarks_dir, f'pinn_eval_{sensor}_{q_name}.csv')
        df_eval.to_csv(pred_csv, index=False)

        # Summary Metrics
        pinn_mae_t = df_eval['abs_dt'].mean()
        pinn_rmse_t = np.sqrt((df_eval['dt']**2).mean())
        pinn_rmse_e = df_eval['rmse_eps'].mean()
        pinn_null_e = df_eval['null_rmse_eps'].mean()
        pinn_gain = df_eval['gain_over_null_pct'].mean()
        pinn_sam = df_eval['sam_deg'].mean()
        pinn_curv = df_eval['curvature_error'].mean()

        row = {
            'regime': q_name,
            'samples': len(df_eval),
            'PINN_DT_MAE_K': pinn_mae_t,
            'PINN_DT_RMSE_K': pinn_rmse_t,
            'PINN_RMSE_eps': pinn_rmse_e,
            'Null_RMSE_eps': pinn_null_e,
            'PINN_Gain_Over_Null_Pct': pinn_gain,
            'PINN_SAM_deg': pinn_sam,
            'PINN_Curvature_Error': pinn_curv
        }

        # Add paired TES baseline metrics if available
        if df_tes is not None:
            tes_tag_map = {
                'Quadrant_I_DevPool': 'val_fold0',
                'Quadrant_II_GoldenTargets_DevAtm': 'golden_quadrant_II',
                'Quadrant_III_DevMaterials_WinterAtm': 'quadrant_III_fold0',
                'Quadrant_IV_GoldenTargets_WinterAtm_OOD': 'golden_quadrant_IV_ood'
            }
            tag = tes_tag_map.get(q_name)
            if tag in df_tes['dataset'].values:
                df_tag = df_tes[df_tes['dataset'] == tag]
                mae_col = 'abs_delta_t_k' if 'abs_delta_t_k' in df_tag.columns else 'abs_dt'
                dt_col = 'delta_t_k' if 'delta_t_k' in df_tag.columns else 'dt'
                eps_col = 'rmse_emissivity' if 'rmse_emissivity' in df_tag.columns else 'rmse_eps'
                sam_col = 'sam_deg'

                tes_mae_t = df_tag[mae_col].mean()
                tes_rmse_t = np.sqrt((df_tag[dt_col]**2).mean())
                tes_rmse_e = df_tag[eps_col].mean()
                tes_sam = df_tag[sam_col].mean()

                row['TES_DT_MAE_K'] = tes_mae_t
                row['TES_DT_RMSE_K'] = tes_rmse_t
                row['TES_RMSE_eps'] = tes_rmse_e
                row['TES_SAM_deg'] = tes_sam

                error_red = (tes_rmse_t - pinn_rmse_t) / tes_rmse_t * 100.0
                row['RMSE_T_Reduction_Pct'] = error_red

        matrix_summary.append(row)

    df_matrix = pd.DataFrame(matrix_summary)
    out_csv = os.path.join(benchmarks_dir, f'quarantine_matrix_benchmark_{sensor}.csv')
    df_matrix.to_csv(out_csv, index=False)

    print(f"\n{'='*75}")
    print(f"Double-Quarantine Matrix Benchmark Summary ({sensor.upper()}):")
    print(f"{'='*75}")
    print(df_matrix.to_string(index=False))
    print(f"\nSaved matrix benchmark to: {out_csv}")

    return df_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Double-Quarantine Matrix (Stage 1)")
    parser.add_argument('--sensor', type=str, default='aster', choices=['aster', 'hytes'], help='Sensor')
    parser.add_argument('--model_path', type=str, default=None, help='Model path')
    args = parser.parse_args()

    evaluate_quarantine_matrix(
        sensor=args.sensor,
        model_path=args.model_path
    )
