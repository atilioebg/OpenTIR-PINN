"""
Archive Phase C Run 2 and Prepare Environment for Run 3
======================================================
1. Copies all outputs (models, benchmarks, figures, logs, configs) to outputs_phase_c_run2/
2. Generates comprehensive provenance README.md in outputs_phase_c_run2/
3. Cleans outputs/ leaving only outputs/baselines/ preserved for Run 3
"""

import os
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, 'outputs')
DEST_DIR = os.path.join(PROJECT_ROOT, 'outputs_phase_c_run2')

os.makedirs(DEST_DIR, exist_ok=True)

# 1. Directories to copy
for sub in ['models', 'benchmarks', 'figures', 'baselines']:
    s = os.path.join(SRC_DIR, sub)
    d = os.path.join(DEST_DIR, sub)
    if os.path.exists(s):
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)
        print(f"Copied directory: {sub} -> {d}")

# 2. Individual files to copy
for f in os.listdir(SRC_DIR):
    s = os.path.join(SRC_DIR, f)
    d = os.path.join(DEST_DIR, f)
    if os.path.isfile(s):
        shutil.copy2(s, d)
        print(f"Copied file: {f}")

# 3. Create comprehensive Provenance README.md
readme_path = os.path.join(DEST_DIR, 'README.md')
with open(readme_path, 'w', encoding='utf-8') as out_f:
    out_f.write("# Phase C Production Campaign: Run 2 Provenance & Benchmark Archive\n\n")
    out_f.write("**Date of Execution:** 06/11/2026 (Completed at 20:15:24)\n")
    out_f.write("**Total Wall-Clock Time:** 289.81 minutes (4.83 hours)\n")
    out_f.write("**Execution Orchestrator:** `scripts/run_production_phase_c.py`\n")
    out_f.write("**Execution State:** 100% Complete & Verified\n\n")
    out_f.write("---\n\n")
    out_f.write("## 1. Architectural Search Space & Key Hyperparameters (Run 2)\n")
    out_f.write("In this campaign, the search space was expanded to address Reviewer #2 concerns regarding depth, batch size, and activation function smoothness:\n")
    out_f.write("* **Batch Sizes Evaluated:** `[8, 16, 32]` (Both sensors converged on **Batch Size = 8**)\n")
    out_f.write("* **Depths Evaluated:** `[3, 5, 7]` (HyTES selected **3 layers**; ASTER selected **5 layers**)\n")
    out_f.write("* **Activations Evaluated:** `['SiLU', 'GELU', 'ELU']` (Both sensors selected **SiLU**)\n")
    out_f.write("* **Atmospheric Ingestion Modes Evaluated:** `['single', 'dual', 'triple', 'quad']` (HyTES selected **single**; ASTER selected **triple**)\n\n")
    out_f.write("### Stamped Best Configurations:\n")
    out_f.write("* **HyTES (Trial #15):** `batch_size = 8`, `hidden_dim = 768`, `n_layers = 3`, `activation = SiLU`, `input_mode = single`, `lr = 2.26e-4`, `optimizer = AdamW`, Score = 0.5214\n")
    out_f.write("* **ASTER (Trial #18):** `batch_size = 8`, `hidden_dim = 384`, `n_layers = 5`, `activation = SiLU`, `input_mode = triple`, `lr = 2.01e-4`, `optimizer = AdamW`, Score = 0.6860\n\n")
    out_f.write("---\n\n")
    out_f.write("## 2. Master Results Summary\n\n")
    out_f.write("### A. 5-Fold Stratified Group Cross-Validation\n")
    out_f.write("* **HyTES (Mean ± STD):**\n")
    out_f.write("  * T_RMSE = 1.298 K ± 0.108 K (Run 1 was 1.524 K)\n")
    out_f.write("  * T_MAE = 0.977 K ± 0.088 K\n")
    out_f.write("  * eps_RMSE = 0.0146 ± 0.0019 (Run 1 was 0.0174)\n")
    out_f.write("  * Gain over Null Prior: +64.96% ± 4.39%\n")
    out_f.write("  * Spectral Angle Mapper (SAM): 0.454° ± 0.071°\n")
    out_f.write("* **ASTER (Mean ± STD):**\n")
    out_f.write("  * T_RMSE = 1.581 K ± 0.201 K (Run 1 was 1.619 K)\n")
    out_f.write("  * T_MAE = 1.182 K ± 0.163 K\n")
    out_f.write("  * eps_RMSE = 0.0233 ± 0.0054\n")
    out_f.write("  * Gain over Null Prior: +50.49% ± 6.13%\n")
    out_f.write("  * Spectral Angle Mapper (SAM): 0.576° ± 0.059°\n\n")
    out_f.write("### B. Double-Quarantine Matrix vs. Canonical Classical TES (HyTES)\n")
    out_f.write("* **Quadrant I (DevPool):** PINN T_RMSE = 1.363 K vs. Classical TES 2.295 K (+40.63% lower error) | eps_RMSE = 0.0171 vs. 0.0449 (2.6x more accurate)\n")
    out_f.write("* **Quadrant II (Golden Targets):** PINN T_RMSE = 1.775 K vs. Classical TES 2.311 K (+23.17% lower error) | eps_RMSE = 0.0297 vs. 0.0392\n")
    out_f.write("* **Quadrant III (Winter Atm):** PINN T_RMSE = 1.837 K, eps_RMSE = 0.0180\n")
    out_f.write("* **Quadrant IV (Golden Targets OOD):** PINN T_RMSE = 1.841 K vs. Classical TES 2.087 K (+11.79% lower error)\n\n")
    out_f.write("### C. Reduced LOCO 7-Class Zero-Shot Mineral Transfer\n")
    out_f.write("* **HyTES Mean OOD:** T_RMSE = 1.616 K, eps_RMSE = 0.0284, Gain over Null = +40.42%, SAM = 1.278°\n")
    out_f.write("* **ASTER Mean OOD:** T_RMSE = 1.686 K, eps_RMSE = 0.0291, Gain over Null = +41.59%, SAM = 1.142°\n\n")
    out_f.write("---\n\n")
    out_f.write("## 3. Contents of this Archive\n")
    out_f.write("* `models/cv/`: 10 fully trained cross-validation checkpoints (5 ASTER folds + 5 HyTES folds)\n")
    out_f.write("* `benchmarks/`: Master summary, cross-validation CSVs, quarantine matrix CSVs, LOCO CSVs, ablation CSVs, and all 10 decoupled training history CSVs\n")
    out_f.write("* `figures/`: 300 DPI publication figures comparing PINN against Classical TES\n")
    out_f.write("* `best_config_*.json`: Formally stamped Optuna configuration parameters\n")
    out_f.write("* `optuna_trials_*.csv`: Full tabular log of all 20 Optuna trials per sensor\n")
    out_f.write("* `production_phase_c.log`: Complete timestamped execution log\n")

print(f"Created archive README: {readme_path}")

# 4. Clean workspace outputs/ for Run 3 (Preserving outputs/baselines/)
print("\nCleaning workspace outputs/ for fresh Run 3...")
for item in os.listdir(SRC_DIR):
    if item == 'baselines':
        print("  Preserved outputs/baselines/ (Static invariant baseline)")
        continue
    item_path = os.path.join(SRC_DIR, item)
    if os.path.isdir(item_path):
        shutil.rmtree(item_path)
        print(f"  Deleted directory: outputs/{item}")
    elif os.path.isfile(item_path):
        os.remove(item_path)
        print(f"  Deleted file: outputs/{item}")

# Re-create empty target subdirectories in outputs/
os.makedirs(os.path.join(SRC_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(SRC_DIR, 'benchmarks'), exist_ok=True)
os.makedirs(os.path.join(SRC_DIR, 'figures'), exist_ok=True)

print("\nWorkspace outputs/ is now clean and ready for Phase C Run 3 execution!")
