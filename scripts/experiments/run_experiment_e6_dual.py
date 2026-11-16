"""
Experiment E6: Strict Dual-Input Atmospheric Parity Suite
=========================================================
Trains and evaluates production-grade Sobolev-PINN architectures across all 5 folds
under strict input parity with Classical TES:
    Input: [L_surf, L_down]  (input_mode = 'dual')
for both ASTER (5 bands x 2 = 10 input channels) and HyTES (223 transparent bands x 2 = 446 channels).

Research Questions Addressed:
1. Does the -14.2% LST improvement on ASTER persist when the PINN is stripped of tau_hat
   and receives identical input information to TES?
2. Does providing downwelling radiance (L_down) to HyTES PINN (which used 'single' in Table 2)
   further enhance or maintain its -44.0% advantage over TES?
3. What is the isolated marginal contribution of tau_hat to PINN retrieval accuracy?

Statistical Methodology:
- 5-fold Stratified Group Cross-Validation (100 epochs/fold).
- Cluster-bootstrapped paired difference (B = 2,000 iterations, clustered by spectrum_id).
- Evaluated on Out-of-Fold (OOF-I, 10,780 samples, 231 spectra) and Clean-Room Golden Targets (Q-II, Q-IV).
- Completely isolated under `outputs_dual/` to preserve Run D production files.

Author: Dr. Atilio Grondona & OpenTIR-PINN Research Team
Target Journal: Remote Sensing of Environment (Elsevier)

NOTE (09/11/2026): the statistics printed by this runner are NOT authoritative. Its emissivity RMSE is a
pooled RMSE (samples x bands), whereas the manuscript (D01 / Table 2) uses the mean over samples of the
per-sample RMSE; it also evaluates OOF-I only. Use scripts/experiments/evaluate_e6_d01_definitions.py
(outputs: E6_headline_D01_definitions.csv, E6_paired_production_vs_dual.csv). Training is unseeded.
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.training.train_cross_validation import run_cross_validation
from src.evaluation.evaluate_quarantine_matrix import load_model_from_checkpoint
from src.training.input_utils import build_atmospheric_band_cache, assemble_input_tensor
from src.training.stage1_core import standardize_input, get_sensor_wavelengths
from src.baselines.classical_tes import ClassicalTES

# Setup isolated directories
BASE_OUT = os.path.join(PROJECT_ROOT, 'outputs_dual')
BENCH_DIR = os.path.join(BASE_OUT, 'benchmarks')
MODELS_DIR = os.path.join(BASE_OUT, 'models', 'cv')
CACHE_DIR = os.path.join(BASE_OUT, 'cache')
LOG_FILE = os.path.join(BASE_OUT, 'experiment_e6_dual.log')

os.makedirs(BENCH_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Dataset paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets')
RUN_D_CACHE = os.path.join(PROJECT_ROOT, 'docs', 'analysis_D', 'cache')

PARTS = {
    **{f'V{k}': (f'val_fold{k}_{{s}}.pt', [k]) for k in range(5)},
    **{f'W{k}': (f'quadrant_III_fold{k}_{{s}}.pt', [k]) for k in range(5)},
    'Q2': ('golden_benchmark_quadrant_II_{s}.pt', [0, 1, 2, 3, 4]),
    'Q4': ('golden_benchmark_quadrant_IV_{s}.pt', [0, 1, 2, 3, 4])
}

NB = 2000
_W = {}


def log(msg: str):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(formatted + '\n')


def groups_by(ids):
    return [np.where(ids == g)[0] for g in pd.unique(ids)]


def Wm(G):
    if G not in _W:
        rng = np.random.default_rng(0)
        idx = rng.integers(0, G, (NB, G))
        W = np.zeros((NB, G))
        np.add.at(W, (np.repeat(np.arange(NB), G), idx.ravel()), 1.0)
        _W[G] = W
    return _W[G]


def gs(groups, x):
    return np.array([np.sum(x[g]) for g in groups], float)


def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def boot_rmse_diff(groups, a, b):
    W = Wm(len(groups))
    n = W @ gs(groups, np.ones(len(a)))
    return ci(np.sqrt(W @ gs(groups, a ** 2) / n) - np.sqrt(W @ gs(groups, b ** 2) / n))


def boot_mean_diff(groups, a, b):
    W = Wm(len(groups))
    n = W @ gs(groups, np.ones(len(a)))
    return ci(W @ gs(groups, a - b) / n)


def sam_deg(p, t):
    c = np.sum(p * t, -1) / (np.linalg.norm(p, axis=-1) * np.linalg.norm(t, axis=-1) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def meta_arrays(ds):
    return dict(
        spectrum_id=np.array([s['spectrum_id'] for s in ds]),
        material_class=np.array([s['material_class'] for s in ds]),
        atmosphere_id=np.array([s['atmosphere_id'] for s in ds]),
        t_true=np.array([float(s['target_temp']) for s in ds], dtype=np.float64),
        pwv=np.array([float(s['pwv']) for s in ds]),
        pwv_err=np.array([float(s['pwv_error_drawn']) for s in ds]),
        e_true_full=np.stack([s['target_emissivity'].numpy() for s in ds]).astype(np.float32),
    )


def predict(ckpt, sensor, ds, atm_cache, dev):
    model, cfg, mode, mu_x, std_x, mask, _ = load_model_from_checkpoint(ckpt, dev, sensor)
    model.eval()
    T, E = [], []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=256, shuffle=False):
            x = standardize_input(assemble_input_tensor(batch, mode, atm_cache, dev, mask=mask), mu_x, std_x)
            t, e = model(x)
            T.append(t.squeeze(-1).cpu().numpy())
            E.append(e.cpu().numpy())
    return np.concatenate(T), np.concatenate(E).astype(np.float32), mask.cpu().numpy(), mode


def run_prediction_cache(sensor: str, dev: torch.device):
    log(f"\nBuilding Prediction Cache for {sensor.upper()} under Dual Input...")
    atm_cache = build_atmospheric_band_cache(sensor)
    wav = get_sensor_wavelengths(sensor, dev).cpu().numpy()

    for part, (fname, folds) in PARTS.items():
        ds_path = os.path.join(DATA_DIR, fname.format(s=sensor))
        ds = torch.load(ds_path, weights_only=False)
        meta = meta_arrays(ds)
        out_path = os.path.join(CACHE_DIR, f'E6_dual_{sensor}_{part}.npz')

        Tp, Ep = [], []
        for k in folds:
            ck = os.path.join(MODELS_DIR, sensor, f'model_{sensor}_fold{k}_best.pth')
            t_k, e_k, mask, mode = predict(ck, sensor, ds, atm_cache, dev)
            Tp.append(t_k)
            Ep.append(e_k)

        np.savez_compressed(
            out_path,
            t_pred=np.stack(Tp),
            e_pred=np.stack(Ep),
            band_mask=mask,
            input_mode=mode,
            folds=np.array(folds),
            wav=wav,
            **meta
        )
        log(f"  Saved partition cache: {part} ({len(ds)} samples, mode={mode}) -> {out_path}")


def evaluate_statistical_parity(sensor: str):
    log(f"\n================================================================================")
    log(f"EVALUATING STATISTICAL PARITY: {sensor.upper()} PINN(dual) vs Classical TES(dual)")
    log(f"================================================================================")

    # 1. Pool OOF-I across 5 folds (V0 to V4)
    pinn_t_list, pinn_e_list = [], []
    tes_t_list, tes_e_list = [], []
    t_true_list, e_true_list = [], []
    spec_id_list = []

    for k in range(5):
        pinn_npz = np.load(os.path.join(CACHE_DIR, f'E6_dual_{sensor}_V{k}.npz'))
        tes_npz = np.load(os.path.join(RUN_D_CACHE, f'tes_{sensor}_V{k}.npz'))

        # PINN prediction for fold k is at index 0 of the stacked fold predictions
        pinn_t = pinn_npz['t_pred'][0]
        pinn_e = pinn_npz['e_pred'][0]
        mask = pinn_npz['band_mask']

        tes_t = tes_npz['t_pred']
        tes_e = tes_npz['e_pred_full']
        if sensor == 'hytes':
            # Compare on TES evaluation window mask
            win_mask = tes_npz['window_mask']
            # Evaluate emissivity on common window
            pinn_e_eval = pinn_e[:, win_mask[mask]]
            tes_e_eval = tes_e[:, win_mask]
            e_true_eval = pinn_npz['e_true_full'][:, win_mask]
        else:
            pinn_e_eval = pinn_e
            tes_e_eval = tes_e
            e_true_eval = pinn_npz['e_true_full']

        pinn_t_list.append(pinn_t)
        pinn_e_list.append(pinn_e_eval)
        tes_t_list.append(tes_t)
        tes_e_list.append(tes_e_eval)
        t_true_list.append(pinn_npz['t_true'])
        e_true_list.append(e_true_eval)
        spec_id_list.append(pinn_npz['spectrum_id'])

    p_t = np.concatenate(pinn_t_list)
    p_e = np.concatenate(pinn_e_list, axis=0)
    t_t = np.concatenate(tes_t_list)
    t_e = np.concatenate(tes_e_list, axis=0)
    true_t = np.concatenate(t_true_list)
    true_e = np.concatenate(e_true_list, axis=0)
    s_ids = np.concatenate(spec_id_list)

    groups = groups_by(s_ids)

    # Compute errors
    dt_pinn = p_t - true_t
    dt_tes = t_t - true_t

    de_pinn = np.sqrt(np.mean((p_e - true_e)**2, axis=-1))
    de_tes = np.sqrt(np.mean((t_e - true_e)**2, axis=-1))

    sam_pinn = sam_deg(p_e, true_e)
    sam_tes = sam_deg(t_e, true_e)

    # Base Metrics
    rmse_t_pinn = float(np.sqrt(np.mean(dt_pinn**2)))
    rmse_t_tes = float(np.sqrt(np.mean(dt_tes**2)))
    mae_t_pinn = float(np.mean(np.abs(dt_pinn)))
    mae_t_tes = float(np.mean(np.abs(dt_tes)))

    rmse_e_pinn = float(np.sqrt(np.mean((p_e - true_e)**2)))
    rmse_e_tes = float(np.sqrt(np.mean((t_e - true_e)**2)))

    sam_pinn_mean = float(np.mean(sam_pinn))
    sam_tes_mean = float(np.mean(sam_tes))

    # Relative Gains
    gain_t_rmse = (rmse_t_pinn - rmse_t_tes) / rmse_t_tes * 100.0
    gain_e_rmse = (rmse_e_pinn - rmse_e_tes) / rmse_e_tes * 100.0
    gain_sam = (sam_pinn_mean - sam_tes_mean) / sam_tes_mean * 100.0

    # Paired Bootstrap Differences (B = 2,000)
    diff_ci_t = boot_rmse_diff(groups, dt_pinn, dt_tes)
    diff_ci_mae = boot_mean_diff(groups, np.abs(dt_pinn), np.abs(dt_tes))
    diff_ci_e = boot_rmse_diff(groups, de_pinn, de_tes)
    diff_ci_sam = boot_mean_diff(groups, sam_pinn, sam_tes)

    log(f"--- OOF-I Results (N = {len(true_t)} samples, {len(groups)} spectra) ---")
    log(f"LST RMSE: PINN(dual) = {rmse_t_pinn:.3f} K vs TES(dual) = {rmse_t_tes:.3f} K | Gain: {gain_t_rmse:+.1f}% | Paired Delta 95% CI: [{diff_ci_t[0]:+.3f}, {diff_ci_t[1]:+.3f}] K")
    log(f"LST MAE : PINN(dual) = {mae_t_pinn:.3f} K vs TES(dual) = {mae_t_tes:.3f} K | Paired Delta 95% CI: [{diff_ci_mae[0]:+.3f}, {diff_ci_mae[1]:+.3f}] K")
    log(f"eps RMSE: PINN(dual) = {rmse_e_pinn:.4f} vs TES(dual) = {rmse_e_tes:.4f} | Gain: {gain_e_rmse:+.1f}% | Paired Delta 95% CI: [{diff_ci_e[0]:+.4f}, {diff_ci_e[1]:+.4f}]")
    log(f"SAM     : PINN(dual) = {sam_pinn_mean:.3f}° vs TES(dual) = {sam_tes_mean:.3f}° | Gain: {gain_sam:+.1f}% | Paired Delta 95% CI: [{diff_ci_sam[0]:+.3f}, {diff_ci_sam[1]:+.3f}]°")

    summary_row = {
        'sensor': sensor,
        'partition': 'OOF-I',
        'pinn_input_mode': 'dual',
        'tes_input_mode': 'dual',
        'n_samples': len(true_t),
        'n_spectra': len(groups),
        'pinn_lst_rmse_k': rmse_t_pinn,
        'tes_lst_rmse_k': rmse_t_tes,
        'gain_lst_rmse_pct': gain_t_rmse,
        'delta_lst_rmse_lo': diff_ci_t[0],
        'delta_lst_rmse_hi': diff_ci_t[1],
        'pinn_lst_mae_k': mae_t_pinn,
        'tes_lst_mae_k': mae_t_tes,
        'delta_lst_mae_lo': diff_ci_mae[0],
        'delta_lst_mae_hi': diff_ci_mae[1],
        'pinn_eps_rmse': rmse_e_pinn,
        'tes_eps_rmse': rmse_e_tes,
        'gain_eps_rmse_pct': gain_e_rmse,
        'delta_eps_rmse_lo': diff_ci_e[0],
        'delta_eps_rmse_hi': diff_ci_e[1],
        'pinn_sam_deg': sam_pinn_mean,
        'tes_sam_deg': sam_tes_mean,
        'gain_sam_pct': gain_sam,
        'delta_sam_lo': diff_ci_sam[0],
        'delta_sam_hi': diff_ci_sam[1],
    }

    return summary_row


def main():
    parser = argparse.ArgumentParser(description="Experiment E6: Strict Dual-Input Atmospheric Parity Suite")
    parser.add_argument('--sensor', type=str, default='all', choices=['all', 'aster', 'hytes'], help="Sensors to evaluate")
    parser.add_argument('--epochs', type=int, default=100, help="Epochs per fold (default: 100)")
    parser.add_argument('--skip_training', action='store_true', help="Skip training and run cache & statistical evaluation only")
    args = parser.parse_args()

    # Route output directory to outputs_dual
    os.environ['OPENTIR_OUTPUT_DIR'] = BASE_OUT

    total_start = time.time()
    log("================================================================================")
    log(f"STARTING EXPERIMENT E6 (DUAL-INPUT PARITY) | SENSORS: {args.sensor.upper()} | EPOCHS: {args.epochs}")
    log(f"Isolated Output Base: {BASE_OUT}")
    log("================================================================================")

    sensors = ['aster', 'hytes'] if args.sensor == 'all' else [args.sensor]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Compute Device: {dev}")

    # 1. 5-Fold Training
    if not args.skip_training:
        for s in sensors:
            t0 = time.time()
            log(f"\nLaunching 5-Fold Training for {s.upper()} with input_mode='dual' ({args.epochs} epochs/fold)...")
            run_cross_validation(sensor=s, epochs=args.epochs, n_folds=5, input_mode='dual')
            log(f"Training for {s.upper()} completed in {(time.time()-t0)/60.0:.2f} min.")
    else:
        log("Skipping training as requested (--skip_training).")

    # 2. Prediction Cache & Evaluation
    summary_rows = []
    for s in sensors:
        run_prediction_cache(s, dev)
        row = evaluate_statistical_parity(s)
        summary_rows.append(row)

    # 3. Export Summary Table
    df_sum = pd.DataFrame(summary_rows)
    out_csv = os.path.join(BENCH_DIR, 'experiment_e6_dual_summary.csv')
    df_sum.to_csv(out_csv, index=False)
    log(f"\n================================================================================")
    log(f"EXPERIMENT E6 COMPLETED SUCCESSFULLY IN {(time.time()-total_start)/60.0:.2f} MINUTES")
    log(f"Master Summary Table saved to: {out_csv}")
    log("================================================================================")
    print("\n" + df_sum.to_string(index=False))


if __name__ == '__main__':
    main()

