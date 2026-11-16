"""
Production Training & Benchmark Orchestrator (Phase C)
=====================================================
Executes the comprehensive production campaign with grounded Stage 1 SSoT:
  Step 1: Informed Optuna Hyperparameter Optimization (15 trials, 20 epochs)
          Generates verified stamped best_config_<sensor>.json
  Step 2: Tuned 5-Fold Cross-Validation (50 epochs per fold)
          Generates out-of-fold benchmark ledgers and best model checkpoints
  Step 3: Double-Quarantine Matrix Evaluation (Quadrants I–IV)
          Assesses out-of-distribution physical generalization vs Canonical Classical TES
  Step 4: Reduced LOCO 7+1 Diagnostic Mineral Suite (50 epochs per class)
          Tests zero-shot transfer on held-out crystallographic classes
  Step 5: Atmospheric Ingestion Ablation (Single, Dual, Triple, Quad; 30 epochs per mode)
          Quantifies value of downwelling radiance and atmospheric decoupling

All outputs are saved to outputs/models/ and outputs/benchmarks/.
Baseline Gate G6 models and benchmarks are safely preserved in outputs_baseline_g6/.

Author: Dr. Atilio Grondona & OpenTIR-PINN Research Team
Target Journal: IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)
"""

import os
import sys
import time
import subprocess
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_EXE = sys.executable


def log(msg: str):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    log_file = os.path.join(PROJECT_ROOT, 'outputs', 'production_phase_c.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a', encoding='utf-8') as f:
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

    proc.wait()
    duration = time.time() - start_t
    if proc.returncode != 0:
        log(f"ERROR: Step '{step_name}' failed with return code {proc.returncode} after {duration:.2f} s")
        sys.exit(proc.returncode)
    else:
        log(f"SUCCESS: Step '{step_name}' completed in {duration/60.0:.2f} min ({duration:.1f} s)")


def compile_production_summary():
    """Compiles key outputs from CV, Quarantine Matrix, LOCO, and Ablation into a single master summary."""
    benchmarks_dir = os.path.join(PROJECT_ROOT, 'outputs', 'benchmarks')
    summary_path = os.path.join(benchmarks_dir, 'production_phase_c_master_summary.csv')
    log(f"\nCompiling master production summary to: {summary_path}")

    summary_records = []

    # 1. Cross-Validation
    for sensor in ['aster', 'hytes']:
        cv_csv = os.path.join(benchmarks_dir, f'cv_benchmark_{sensor}.csv')
        if os.path.exists(cv_csv):
            df_cv = pd.read_csv(cv_csv)
            mean_row = df_cv[df_cv['fold'] == 'MEAN'].iloc[0] if 'MEAN' in df_cv['fold'].values else df_cv.mean(numeric_only=True)
            summary_records.append({
                'sensor': sensor.upper(),
                'evaluation_suite': '5-Fold CV (Mean)',
                'temp_rmse_k': mean_row.get('temp_rmse', None),
                'temp_mae_k': mean_row.get('temp_mae', None),
                'eps_rmse': mean_row.get('eps_rmse', None),
                'eps_gain_over_null_pct': mean_row.get('eps_gain_over_null_pct', None),
                'sam_deg': mean_row.get('sam_deg', None),
                'spectral_corr': mean_row.get('spectral_corr', None)
            })

    # 2. Quarantine Matrix (Q-I, Q-II, Q-III, Q-IV)
    for sensor in ['aster', 'hytes']:
        qm_csv = os.path.join(benchmarks_dir, f'quarantine_matrix_benchmark_{sensor}.csv')
        if os.path.exists(qm_csv):
            df_qm = pd.read_csv(qm_csv)
            for _, r in df_qm.iterrows():
                summary_records.append({
                    'sensor': sensor.upper(),
                    'evaluation_suite': f"Quarantine Matrix ({r['regime']})",
                    'temp_rmse_k': r.get('PINN_DT_RMSE_K', None),
                    'temp_mae_k': r.get('PINN_DT_MAE_K', None),
                    'eps_rmse': r.get('PINN_RMSE_eps', None),
                    'eps_gain_over_null_pct': r.get('PINN_Gain_Over_Null_Pct', None),
                    'sam_deg': r.get('PINN_SAM_deg', None),
                    'spectral_corr': None
                })

    # 3. LOCO Suite (Mean OOD)
    for sensor in ['aster', 'hytes']:
        loco_csv = os.path.join(benchmarks_dir, f'loco_benchmark_{sensor}.csv')
        if os.path.exists(loco_csv):
            df_loco = pd.read_csv(loco_csv)
            if 'held_out_class' in df_loco.columns and 'MEAN_OOD' in df_loco['held_out_class'].values:
                mean_ood = df_loco[df_loco['held_out_class'] == 'MEAN_OOD'].iloc[0]
                summary_records.append({
                    'sensor': sensor.upper(),
                    'evaluation_suite': 'LOCO 7-Class Mean OOD',
                    'temp_rmse_k': mean_ood.get('ood_temp_rmse', None),
                    'temp_mae_k': mean_ood.get('ood_temp_mae', None),
                    'eps_rmse': mean_ood.get('ood_eps_rmse', None),
                    'eps_gain_over_null_pct': mean_ood.get('ood_gain_over_null_pct', None),
                    'sam_deg': mean_ood.get('ood_sam_deg', None),
                    'spectral_corr': mean_ood.get('ood_spectral_corr', None)
                })

    # 4. Ingestion Ablation (Dual vs Single vs Triple vs Quad)
    for sensor in ['aster', 'hytes']:
        ab_csv = os.path.join(benchmarks_dir, f'ablation_ingestion_benchmark_{sensor}.csv')
        if os.path.exists(ab_csv):
            df_ab = pd.read_csv(ab_csv)
            for _, r in df_ab.iterrows():
                summary_records.append({
                    'sensor': sensor.upper(),
                    'evaluation_suite': f"Ablation Ingestion ({r['mode']})",
                    'temp_rmse_k': r.get('temp_rmse', None),
                    'temp_mae_k': r.get('temp_mae', None),
                    'eps_rmse': r.get('eps_rmse', None),
                    'eps_gain_over_null_pct': r.get('gain_over_null_pct', None),
                    'sam_deg': r.get('sam_deg', None),
                    'spectral_corr': r.get('spectral_corr', None)
                })

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(summary_path, index=False)
    log(f"Master Summary Generated Successfully:\n{df_summary.to_string(index=False)}")


def main():
    total_start = time.time()
    log("================================================================================")
    log("LAUNCHING FULL PRODUCTION TRAINING CAMPAIGN (PHASE C)")
    log("SSoT Grounding: Stage 1 Verified Architecture & Parameter Space")
    log("================================================================================")

    # -------------------------------------------------------------
    # 1. OPTUNA HYPERPARAMETER OPTIMIZATION (20 Trials, 20 Epochs, Layers in [3, 5, 7])
    # -------------------------------------------------------------
    run_step(
        "Optuna HPO - ASTER",
        ['src/training/tune_optuna.py', '--sensor', 'aster', '--n_trials', '20', '--n_epochs', '20']
    )
    run_step(
        "Optuna HPO - HyTES",
        ['src/training/tune_optuna.py', '--sensor', 'hytes', '--n_trials', '20', '--n_epochs', '20']
    )

    # -------------------------------------------------------------
    # 2. TUNED 5-FOLD CROSS-VALIDATION (50 Epochs / Fold)
    # -------------------------------------------------------------
    run_step(
        "5-Fold Cross-Validation - ASTER",
        ['src/training/train_cross_validation.py', '--sensor', 'aster', '--epochs', '50']
    )
    run_step(
        "5-Fold Cross-Validation - HyTES",
        ['src/training/train_cross_validation.py', '--sensor', 'hytes', '--epochs', '50']
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
    # 4. REDUCED LOCO 7-CLASS SUITE (50 Epochs / Class)
    # -------------------------------------------------------------
    run_step(
        "LOCO 7-Class Stress-Test - ASTER",
        ['src/training/train_loco_suite.py', '--sensor', 'aster', '--epochs', '50']
    )
    run_step(
        "LOCO 7-Class Stress-Test - HyTES",
        ['src/training/train_loco_suite.py', '--sensor', 'hytes', '--epochs', '50']
    )

    # -------------------------------------------------------------
    # 5. ATMOSPHERIC INGESTION ABLATION (Single, Dual, Triple, Quad; 30 Epochs)
    # -------------------------------------------------------------
    run_step(
        "Atmospheric Ingestion Ablation - ASTER",
        ['src/training/train_ablation_ingestion.py', '--sensor', 'aster', '--epochs', '30']
    )
    run_step(
        "Atmospheric Ingestion Ablation - HyTES",
        ['src/training/train_ablation_ingestion.py', '--sensor', 'hytes', '--epochs', '30']
    )

    # -------------------------------------------------------------
    # 6. MASTER PRODUCTION SUMMARY COMPILATION
    # -------------------------------------------------------------
    compile_production_summary()

    total_time = time.time() - total_start
    log("================================================================================")
    log(f"PHASE C PRODUCTION CAMPAIGN COMPLETED SUCCESSFULLY IN {total_time/60.0:.2f} MINUTES")
    log("================================================================================")


if __name__ == '__main__':
    main()

