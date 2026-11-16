"""
Gate G6: Baseline 5-Fold Cross-Validation & Quarantine Evaluation Campaign
========================================================================
Executes the official initial baseline training campaign:
  1. Trains 5-fold cross-validation on ASTER (50 epochs) using Stage 1 SSoT defaults.
  2. Evaluates ASTER models across Quadrants I, II, III, and IV against Classical TES.
  3. Trains 5-fold cross-validation on HyTES (50 epochs) using Stage 1 SSoT defaults (223 bands).
  4. Evaluates HyTES models across Quadrants I, II, III, and IV against Classical TES.
  5. Compiles unified benchmark ledgers in outputs/benchmarks/.

Author: Dr. Atilio Grondona & OpenTIR-PINN Research Team
Target Journal: IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)
"""

import os
import sys
import time
import argparse
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.training.train_cross_validation import run_cross_validation
from src.evaluation.evaluate_quarantine_matrix import evaluate_quarantine_matrix


def run_campaign(epochs: int = 50, sensors: list = None):
    if sensors is None:
        sensors = ['aster', 'hytes']

    start_total = time.time()
    print(f"\n{'#'*80}")
    print(f"LAUNCHING GATE G6 BASELINE TRAINING CAMPAIGN")
    print(f"Sensors: {sensors} | Epochs per fold: {epochs} | Folds: 5")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}\n")

    summary_records = []

    for sensor in sensors:
        s_start = time.time()
        print(f"\n{'='*80}")
        print(f"STEP 1: Executing 5-Fold Cross-Validation for {sensor.upper()} ({epochs} Epochs/Fold)")
        print(f"{'='*80}")
        df_cv = run_cross_validation(sensor=sensor, epochs=epochs, n_folds=5)

        print(f"\n{'='*80}")
        print(f"STEP 2: Evaluating Double-Quarantine Matrix (Q-I..IV) for {sensor.upper()}")
        print(f"{'='*80}")
        df_matrix = evaluate_quarantine_matrix(sensor=sensor)

        s_elapsed = time.time() - s_start

        # Extract macro row
        cv_mean = df_cv[df_cv['fold'] == 'MEAN'].iloc[0] if 'MEAN' in df_cv['fold'].values else df_cv.iloc[-1]
        summary_records.append({
            'sensor': sensor.upper(),
            'cv_temp_rmse': cv_mean.get('temp_rmse', float('nan')),
            'cv_eps_rmse': cv_mean.get('eps_rmse', float('nan')),
            'cv_null_eps_rmse': cv_mean.get('null_eps_rmse', float('nan')),
            'cv_gain_over_null_pct': cv_mean.get('eps_gain_over_null_pct', float('nan')),
            'cv_sam_deg': cv_mean.get('sam_deg', float('nan')),
            'wall_clock_min': s_elapsed / 60.0
        })

    total_elapsed = time.time() - start_total
    df_summary = pd.DataFrame(summary_records)

    print(f"\n{'#'*80}")
    print(f"GATE G6 BASELINE CAMPAIGN COMPLETED SUCCESSFULLY!")
    print(f"Total Wall-Clock Time: {total_elapsed/60.0:.2f} minutes")
    print(f"{'#'*80}")
    print(df_summary.to_string(index=False))
    print(f"{'#'*80}\n")

    summary_csv = os.path.join(PROJECT_ROOT, 'outputs', 'benchmarks', 'gate_g6_baseline_summary.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved campaign summary to: {summary_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Gate G6 Baseline Training Campaign")
    parser.add_argument('--epochs', type=int, default=50, help='Epochs per fold (default: 50)')
    parser.add_argument('--sensor', type=str, default='all', choices=['all', 'aster', 'hytes'], help='Sensor to train')
    args = parser.parse_args()

    active_sensors = ['aster', 'hytes'] if args.sensor == 'all' else [args.sensor]
    run_campaign(epochs=args.epochs, sensors=active_sensors)

