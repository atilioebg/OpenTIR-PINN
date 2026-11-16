"""
Master Training & Benchmarking Orchestrator (Step 2.7)
=====================================================
Executes the complete retraining and evaluation campaign:
  - Stage 1: Optuna Hyperparameter Optimization (outputs/best_config_<sensor>.json)
  - Stage 2: 5-Fold Stratified Group Cross-Validation (outputs/models/model_<sensor>_fold<k>_best.pth)
  - Stage 3: Leave-One-Class-Out (LOCO 7+1) Generalization Suite (outputs/benchmarks/loco_benchmark_<sensor>.csv)
  - Stage 4: Double-Quarantine 2x2 Matrix Benchmarking vs. Classical TES (outputs/benchmarks/quarantine_matrix_benchmark_<sensor>.csv)

Author: Dr. Atilio Grondona & OpenTIR-PINN Research Team
Target Journal: IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)
"""

import os
import sys
import time
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.training.tune_optuna import run_optuna_study
from src.training.train_cross_validation import run_cross_validation
from src.training.train_loco_suite import run_loco_suite
from src.evaluation.evaluate_quarantine_matrix import evaluate_quarantine_matrix


def run_full_pipeline(
    sensors: list = ['aster', 'hytes'],
    optuna_trials: int = 25,
    optuna_epochs: int = 25,
    cv_epochs: int = 80,
    loco_epochs: int = 40,
    skip_optuna: bool = False
):
    """Runs the end-to-end retraining and benchmarking campaign."""
    master_start_time = time.time()
    print(f"\n{'#'*80}")
    print(f"MASTER RETRAINING & BENCHMARKING PIPELINE (STEP 2.7)")
    print(f"Sensors: {sensors} | Optuna Trials: {optuna_trials} | CV Epochs: {cv_epochs}")
    print(f"{'#'*80}")

    for sensor in sensors:
        s_start = time.time()
        print(f"\n{'*'*80}")
        print(f"STARTING COMPREHENSIVE CAMPAIGN FOR SENSOR: {sensor.upper()}")
        print(f"{'*'*80}")

        # -------------------------------------------------------------
        # Stage 1: Optuna Hyperparameter Optimization
        # -------------------------------------------------------------
        if not skip_optuna:
            print(f"\n[Stage 1/4] Running Optuna Optimization ({optuna_trials} trials, {optuna_epochs} epochs/trial)...")
            run_optuna_study(
                sensor=sensor,
                n_trials=optuna_trials,
                n_epochs=optuna_epochs,
                fold_idx=0
            )
        else:
            print("\n[Stage 1/4] Skipping Optuna Optimization (using existing best config or defaults).")

        # -------------------------------------------------------------
        # Stage 2: 5-Fold Stratified Group Cross-Validation
        # -------------------------------------------------------------
        print(f"\n[Stage 2/4] Training 5-Fold Stratified Group Cross-Validation ({cv_epochs} epochs/fold)...")
        run_cross_validation(
            sensor=sensor,
            epochs=cv_epochs,
            n_folds=5
        )

        # -------------------------------------------------------------
        # Stage 3: Leave-One-Class-Out (LOCO 7+1) Generalization Suite
        # -------------------------------------------------------------
        print(f"\n[Stage 3/4] Executing Leave-One-Class-Out (LOCO 7+1) Suite ({loco_epochs} epochs/model)...")
        run_loco_suite(
            sensor=sensor,
            epochs=loco_epochs
        )

        # -------------------------------------------------------------
        # Stage 4: Double-Quarantine Matrix Benchmark vs. Classical TES
        # -------------------------------------------------------------
        print(f"\n[Stage 4/4] Evaluating Double-Quarantine Matrix Benchmark vs. Classical TES...")
        evaluate_quarantine_matrix(
            sensor=sensor
        )

        s_elapsed = time.time() - s_start
        print(f"\nFinished Campaign for {sensor.upper()} in {s_elapsed/3600:.2f} hours ({s_elapsed:.1f} s).")

    total_elapsed = time.time() - master_start_time
    print(f"\n{'#'*80}")
    print(f"ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Total Execution Time: {total_elapsed/3600:.2f} hours ({total_elapsed:.1f} s).")
    print(f"All models, SQLite studies, and benchmark CSVs saved in outputs/.")
    print(f"{'#'*80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Master Retraining Pipeline")
    parser.add_argument('--sensors', nargs='+', default=['aster', 'hytes'], help='Sensors to run')
    parser.add_argument('--optuna_trials', type=int, default=25, help='Optuna trials per sensor')
    parser.add_argument('--optuna_epochs', type=int, default=25, help='Epochs per Optuna trial')
    parser.add_argument('--cv_epochs', type=int, default=80, help='Epochs per CV fold')
    parser.add_argument('--loco_epochs', type=int, default=40, help='Epochs per LOCO model')
    parser.add_argument('--skip_optuna', action='store_true', help='Skip Optuna and use existing config')
    args = parser.parse_args()

    run_full_pipeline(
        sensors=args.sensors,
        optuna_trials=args.optuna_trials,
        optuna_epochs=args.optuna_epochs,
        cv_epochs=args.cv_epochs,
        loco_epochs=args.loco_epochs,
        skip_optuna=args.skip_optuna
    )
