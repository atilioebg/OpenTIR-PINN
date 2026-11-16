"""
Production Training & Benchmark Orchestrator (Phase D)
=====================================================
Executes the Phase D production campaign with:
  1. Unified Search Space Optuna HPO (Union of Run B and Run C search spaces)
     - Pre-enqueues Trial 0 (Run B validated baseline) and Trial 1 (Run C validated baseline)
  2. Extended 100-Epoch 5-Fold Stratified Group Cross-Validation
     - Allows complete learning-rate annealing and asymptotic convergence
  3. Double-Quarantine Matrix Evaluation (Quadrants I–IV)
     - Full ensemble and fold-by-fold evaluation against Canonical Classical TES
  4. Diagnostic LOCO 7-Class Suite (50 epochs per held-out class)
  5. Atmospheric Ingestion Ablation (Single, Dual, Triple, Quad; 30 epochs)
  6. Automatic Compilation of Master Production Summary & Telemetry

Modes:
  --mode full         : Complete campaign (Steps 1 to 6).
  --mode skip_optuna  : Uses existing validated best configs and runs Steps 2 to 6 with 100 epochs.
  --mode cv_only      : Fast focused run (Step 2 CV with 100 epochs + Step 3 Quarantine Matrix).

Target Venues: IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS) / JSTARS
Authors: Dr. Atilio Grondona & OpenTIR-PINN Research Team
"""

import os
import sys
import time
import argparse
import subprocess
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_EXE = sys.executable
LOG_FILE = os.path.join(PROJECT_ROOT, 'outputs', 'production_phase_d.log')


def log(msg: str):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(formatted + '\n')


def run_step(step_name: str, cmd_args: list):
    log(f"\n{'='*80}")
    log(f"STARTING STEP: {step_name}")
    log(f"COMMAND: {' '.join(cmd_args)}")
    log(f"{'='*80}")
    start_t = time.time()

    env = os.environ.copy()
    env['PYTHONPATH'] = PROJECT_ROOT

    proc = subprocess.Popen(
        [PYTHON_EXE] + cmd_args,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in proc.stdout:
        print(line, end='', flush=True)
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line)

    proc.wait()
    elapsed = time.time() - start_t
    if proc.returncode != 0:
        log(f"ERROR: Step '{step_name}' failed with exit code {proc.returncode} after {elapsed/60.0:.2f} min.")
        sys.exit(proc.returncode)
    else:
        log(f"SUCCESS: Step '{step_name}' completed in {elapsed/60.0:.2f} min ({elapsed:.1f} s)")


def compile_production_summary():
    log("\nCompiling Phase D Master Production Summary...")
    benchmarks_dir = os.path.join(PROJECT_ROOT, 'outputs', 'benchmarks')
    summary_rows = []

    for sensor in ['aster', 'hytes']:
        # 1. 5-Fold CV
        cv_path = os.path.join(benchmarks_dir, f'cv_benchmark_{sensor}.csv')
        if os.path.exists(cv_path):
            df_cv = pd.read_csv(cv_path)
            mean_row = df_cv[df_cv['fold'] == 'MEAN'].iloc[0]
            summary_rows.append({
                'sensor': sensor,
                'benchmark': '5-Fold CV (Mean)',
                'temp_rmse_k': float(mean_row['temp_rmse']),
                'temp_mae_k': float(mean_row['temp_mae']),
                'eps_rmse': float(mean_row['eps_rmse']),
                'sam_deg': float(mean_row['sam_deg']),
                'spectral_corr': float(mean_row.get('spectral_corr', 0.0)),
                'wall_clock_s': float(mean_row['wall_clock_s'])
            })

        # 2. Quarantine Matrix
        qm_path = os.path.join(benchmarks_dir, f'quarantine_matrix_benchmark_{sensor}.csv')
        if os.path.exists(qm_path):
            df_qm = pd.read_csv(qm_path)
            for _, r in df_qm.iterrows():
                regime_name = str(r.get('regime', r.get('quadrant', 'Unknown')))
                t_rmse = float(r.get('PINN_DT_RMSE_K', r.get('temp_rmse_k', 0.0)))
                t_mae = float(r.get('PINN_DT_MAE_K', r.get('temp_mae_k', 0.0)))
                e_rmse = float(r.get('PINN_RMSE_eps', r.get('eps_rmse', 0.0)))
                s_deg = float(r.get('PINN_SAM_deg', r.get('sam_deg', 0.0)))
                summary_rows.append({
                    'sensor': sensor,
                    'benchmark': f"Quarantine_{regime_name}",
                    'temp_rmse_k': t_rmse,
                    'temp_mae_k': t_mae,
                    'eps_rmse': e_rmse,
                    'sam_deg': s_deg,
                    'spectral_corr': 0.0,
                    'wall_clock_s': 0.0
                })

        # 3. LOCO Suite
        loco_path = os.path.join(benchmarks_dir, f'loco_benchmark_{sensor}.csv')
        if os.path.exists(loco_path):
            df_loco = pd.read_csv(loco_path)
            mean_loco = df_loco[df_loco['held_out_class'] == 'MEAN_OOD'].iloc[0]
            summary_rows.append({
                'sensor': sensor,
                'benchmark': 'LOCO Zero-Shot (Mean OOD)',
                'temp_rmse_k': float(mean_loco['ood_temp_rmse']),
                'temp_mae_k': float(mean_loco['ood_temp_mae']),
                'eps_rmse': float(mean_loco['ood_eps_rmse']),
                'sam_deg': float(mean_loco['ood_sam_deg']),
                'spectral_corr': 0.0,
                'wall_clock_s': float(df_loco['wall_clock_s'].sum())
            })

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        out_path = os.path.join(benchmarks_dir, 'production_phase_d_master_summary.csv')
        df_summary.to_csv(out_path, index=False)
        log(f"Phase D Master Summary saved to: {out_path}")
        print("\n" + df_summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Phase D Production Campaign Orchestrator")
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'skip_optuna', 'cv_only'],
                        help="Execution mode: 'full' (Optuna+CV100+Downstream), 'skip_optuna' (CV100+Downstream), 'cv_only' (CV100+QM)")
    parser.add_argument('--epochs_cv', type=int, default=100, help="Epochs per fold in 5-fold cross-validation (default: 100)")
    parser.add_argument('--optuna_trials', type=int, default=20, help="Number of Optuna HPO trials (default: 20)")
    parser.add_argument('--optuna_epochs', type=int, default=20, help="Epochs per Optuna trial (default: 20)")
    parser.add_argument('--search_space', type=str, default='unified', choices=['unified', 'run_c', 'run_b'],
                        help="Optuna search space (default: 'unified' union of B and C)")
    parser.add_argument('--loco_epochs', type=int, default=50, help="Epochs per LOCO model (default: 50)")
    parser.add_argument('--ablation_epochs', type=int, default=30, help="Epochs per ablation mode (default: 30)")
    args = parser.parse_args()

    total_start = time.time()
    log("================================================================================")
    log(f"LAUNCHING PRODUCTION TRAINING CAMPAIGN (PHASE D) | MODE: {args.mode.upper()}")
    log(f"Settings: CV Epochs = {args.epochs_cv} | Search Space = {args.search_space} | Optuna Trials = {args.optuna_trials}")
    log("================================================================================")

    # -------------------------------------------------------------
    # 1. OPTUNA HYPERPARAMETER OPTIMIZATION (Unified Search Space)
    # -------------------------------------------------------------
    if args.mode == 'full':
        run_step(
            "Optuna HPO - ASTER (Unified Space)",
            ['src/training/tune_optuna.py', '--sensor', 'aster',
             '--n_trials', str(args.optuna_trials),
             '--n_epochs', str(args.optuna_epochs),
             '--search_space', args.search_space]
        )
        run_step(
            "Optuna HPO - HyTES (Unified Space)",
            ['src/training/tune_optuna.py', '--sensor', 'hytes',
             '--n_trials', str(args.optuna_trials),
             '--n_epochs', str(args.optuna_epochs),
             '--search_space', args.search_space]
        )
    else:
        log("Skipping Optuna HPO: using established validated best configuration checkpoints.")

    # -------------------------------------------------------------
    # 2. TUNED 5-FOLD CROSS-VALIDATION (100 Epochs / Fold)
    # -------------------------------------------------------------
    run_step(
        f"5-Fold Cross-Validation - ASTER ({args.epochs_cv} Epochs/Fold)",
        ['src/training/train_cross_validation.py', '--sensor', 'aster', '--epochs', str(args.epochs_cv)]
    )
    run_step(
        f"5-Fold Cross-Validation - HyTES ({args.epochs_cv} Epochs/Fold)",
        ['src/training/train_cross_validation.py', '--sensor', 'hytes', '--epochs', str(args.epochs_cv)]
    )

    # -------------------------------------------------------------
    # 3. DOUBLE-QUARANTINE MATRIX EVALUATION (Quadrants I–IV)
    # -------------------------------------------------------------
    run_step(
        "Double-Quarantine Matrix - ASTER",
        ['src/evaluation/evaluate_quarantine_matrix.py', '--sensor', 'aster']
    )
    run_step(
        "Double-Quarantine Matrix - HyTES",
        ['src/evaluation/evaluate_quarantine_matrix.py', '--sensor', 'hytes']
    )

    # -------------------------------------------------------------
    # 4. REDUCED LOCO 7-CLASS SUITE (Optional in cv_only mode)
    # -------------------------------------------------------------
    if args.mode in ['full', 'skip_optuna']:
        run_step(
            f"LOCO 7-Class Stress-Test - ASTER ({args.loco_epochs} Epochs/Class)",
            ['src/training/train_loco_suite.py', '--sensor', 'aster', '--epochs', str(args.loco_epochs)]
        )
        run_step(
            f"LOCO 7-Class Stress-Test - HyTES ({args.loco_epochs} Epochs/Class)",
            ['src/training/train_loco_suite.py', '--sensor', 'hytes', '--epochs', str(args.loco_epochs)]
        )

        # -------------------------------------------------------------
        # 5. ATMOSPHERIC INGESTION ABLATION
        # -------------------------------------------------------------
        run_step(
            f"Atmospheric Ingestion Ablation - ASTER ({args.ablation_epochs} Epochs)",
            ['src/training/train_ablation_ingestion.py', '--sensor', 'aster', '--epochs', str(args.ablation_epochs)]
        )
        run_step(
            f"Atmospheric Ingestion Ablation - HyTES ({args.ablation_epochs} Epochs)",
            ['src/training/train_ablation_ingestion.py', '--sensor', 'hytes', '--epochs', str(args.ablation_epochs)]
        )

    # -------------------------------------------------------------
    # 6. MASTER PRODUCTION SUMMARY COMPILATION
    # -------------------------------------------------------------
    compile_production_summary()

    total_time = time.time() - total_start
    log("================================================================================")
    log(f"PHASE D PRODUCTION CAMPAIGN COMPLETED SUCCESSFULLY IN {total_time/60.0:.2f} MINUTES ({total_time:.1f} s)")
    log("================================================================================")


if __name__ == '__main__':
    main()

