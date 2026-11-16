"""
Publication Figure Generator: Multi-Task Loss Decomposition & Convergence Analysis (Phase C Run 3)
==================================================================================================
Decomposes the Sobolev-PINN objective function into its 6 constituent physical and regularization terms
across all 5 cross-validation folds for both ASTER and HyTES in the freshly completed Phase C Run 3:
  1. Total Composite Objective: L_total = sum(w_i * L_i)
  2. Supervised Emissivity MSE: L_eps
  3. Supervised Temperature MSE: L_T
  4. Spectral Angle Mapper (SAM): L_sam
  5. Sobolev First-Order Gradient Penalty: L_grad (W^{1,1} smoothness)
  6. Sobolev Second-Order Curvature Penalty: L_curve (W^{2,1} feature preservation)
  7. Forward Radiative Transfer Residual: L_phys (Planck RTE consistency)
  8. Physical Metric Trajectories: T_RMSE (K) and eps_RMSE

Target Journal: IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)
Author: Dr. Atilio Grondona & OpenTIR-PINN Research Team
"""

import os
import sys
import shutil
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR_ACTIVE = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
ARTIFACTS_DIR = r"C:\Users\Atilio\.gemini\antigravity\brain\89b8c610-4766-4143-b9ea-f65c9e0ab256"

os.makedirs(FIGURES_DIR_ACTIVE, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Publication styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.labelsize'] = 10.5
plt.rcParams['axes.titlesize'] = 11.5
plt.rcParams['xtick.labelsize'] = 9.0
plt.rcParams['ytick.labelsize'] = 9.0
plt.rcParams['legend.fontsize'] = 8.0
plt.rcParams['figure.titlesize'] = 13.0
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.35
plt.rcParams['grid.linestyle'] = '--'


def load_fold_data(sensor):
    benchmarks_dir = os.path.join(PROJECT_ROOT, 'outputs', 'benchmarks')
    files = sorted(glob.glob(os.path.join(benchmarks_dir, f'training_history_{sensor}_fold*.csv')))
    if not files:
        raise FileNotFoundError(f"No training history files found for {sensor} in {benchmarks_dir}")
    dfs = [pd.read_csv(f) for f in files]
    return dfs


def compute_mean_std(dfs, col):
    matrix = np.array([df[col].values for df in dfs])
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    return mean, std


def save_and_mirror(fig, filename):
    p1 = os.path.join(FIGURES_DIR_ACTIVE, filename)
    p2 = os.path.join(ARTIFACTS_DIR, filename)
    fig.savefig(p1, dpi=300, bbox_inches='tight')
    shutil.copyfile(p1, p2)
    print(f"Saved & mirrored: {filename} -> (Outputs, Artifacts)")


def plot_master_decomposition(dfs_ast, dfs_hy):
    epochs = dfs_ast[0]['epoch'].values
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    c_ast_tr = '#2563EB'
    c_ast_val = '#1D4ED8'
    c_hy_tr = '#059669'
    c_hy_val = '#047857'

    components = [
        (0, 0, r"(a) Total Composite Loss $\mathcal{L}_{\mathrm{total}}$", "Loss Value", 'train_loss_total', 'val_loss_total', True),
        (0, 1, r"(b) Emissivity MSE $\mathcal{L}_{\varepsilon}$", "Normalized MSE", 'train_loss_mse_e', 'val_loss_mse_e', True),
        (0, 2, r"(c) Temperature MSE $\mathcal{L}_{T}$", "Normalized MSE", 'train_loss_mse_t', 'val_loss_mse_t', True),
        (0, 3, r"(d) Spectral Angle Mapper $\mathcal{L}_{\mathrm{SAM}}$", "SAM Loss", 'train_loss_sam', 'val_loss_sam', True),
        (1, 0, r"(e) Sobolev Gradient $\mathcal{L}_{\nabla}$ ($W^{1,1}$)", "Gradient Penalty", 'train_loss_grad', 'val_loss_grad', True),
        (1, 1, r"(f) Sobolev Curvature $\mathcal{L}_{\nabla^2}$ ($W^{2,1}$)", "Curvature Penalty", 'train_loss_curve', 'val_loss_curve', True),
        (1, 2, r"(g) Forward RTE Physics $\mathcal{L}_{\mathrm{phys}}$", "Planck Residual", 'train_loss_phys', 'val_loss_phys', False),
        (1, 3, r"(h) Validation $T_{\mathrm{RMSE}}$ (K)", "Temperature RMSE (K)", None, 'val_temp_rmse_k', False),
    ]

    for r, c, title, ylabel, tr_key, val_key, is_log in components:
        ax = axes[r, c]

        if tr_key is not None:
            m_ast_tr, s_ast_tr = compute_mean_std(dfs_ast, tr_key)
            ax.plot(epochs, m_ast_tr, color=c_ast_tr, lw=1.6, label='ASTER (Train)')
            ax.fill_between(epochs, m_ast_tr - s_ast_tr, m_ast_tr + s_ast_tr, color=c_ast_tr, alpha=0.15)

            m_hy_tr, s_hy_tr = compute_mean_std(dfs_hy, tr_key)
            ax.plot(epochs, m_hy_tr, color=c_hy_tr, lw=1.6, label='HyTES (Train)')
            ax.fill_between(epochs, m_hy_tr - s_hy_tr, m_hy_tr + s_hy_tr, color=c_hy_tr, alpha=0.15)

        m_ast_val, s_ast_val = compute_mean_std(dfs_ast, val_key)
        ax.plot(epochs, m_ast_val, color=c_ast_val, lw=1.6, linestyle='--', label='ASTER (Val)')
        ax.fill_between(epochs, m_ast_val - s_ast_val, m_ast_val + s_ast_val, color=c_ast_val, alpha=0.15)

        m_hy_val, s_hy_val = compute_mean_std(dfs_hy, val_key)
        ax.plot(epochs, m_hy_val, color=c_hy_val, lw=1.6, linestyle='--', label='HyTES (Val)')
        ax.fill_between(epochs, m_hy_val - s_hy_val, m_hy_val + s_hy_val, color=c_hy_val, alpha=0.15)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if is_log:
            ax.set_yscale('log')
        ax.set_xlim(1, len(epochs))

        if r == 0 and c == 0:
            ax.legend(loc='upper right', framealpha=0.9)
        elif r == 1 and c == 3:
            ax.legend(loc='upper right', framealpha=0.9)

    plt.suptitle("Sobolev-PINN Multi-Task Loss Decomposition & Generalization Across 5 Folds (Phase C Run 3)",
                 fontweight='bold', fontsize=14, y=0.995)
    plt.tight_layout()
    save_and_mirror(fig, "fig_loss_decomposition_master_run3.png")
    plt.close()


def plot_sensor_loss_breakdown(dfs, sensor_name, sensor_code):
    epochs = dfs[0]['epoch'].values
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    c_tr = '#1E40AF' if sensor_code == 'aster' else '#047857'
    c_val = '#DC2626'

    panels = [
        (0, 0, r"$\mathcal{L}_{\mathrm{total}}$ (Composite Loss)", 'train_loss_total', 'val_loss_total', True),
        (0, 1, r"$\mathcal{L}_{\varepsilon}$ (Emissivity Normalized MSE)", 'train_loss_mse_e', 'val_loss_mse_e', True),
        (0, 2, r"$\mathcal{L}_{T}$ (Temperature Normalized MSE)", 'train_loss_mse_t', 'val_loss_mse_t', True),
        (1, 0, r"$\mathcal{L}_{\nabla}$ (Sobolev 1st-Derivative Gradient)", 'train_loss_grad', 'val_loss_grad', True),
        (1, 1, r"$\mathcal{L}_{\nabla^2}$ (Sobolev 2nd-Derivative Curvature)", 'train_loss_curve', 'val_loss_curve', True),
        (1, 2, r"$\mathcal{L}_{\mathrm{phys}}$ (Forward Planck RTE Consistency)", 'train_loss_phys', 'val_loss_phys', False),
    ]

    for r, c, title, tr_key, val_key, is_log in panels:
        ax = axes[r, c]
        m_tr, s_tr = compute_mean_std(dfs, tr_key)
        m_val, s_val = compute_mean_std(dfs, val_key)

        ax.plot(epochs, m_tr, color=c_tr, lw=2.0, label=f'Training (5-Fold Mean)')
        ax.fill_between(epochs, m_tr - s_tr, m_tr + s_tr, color=c_tr, alpha=0.18)

        ax.plot(epochs, m_val, color=c_val, lw=1.8, linestyle='--', label=f'Validation (5-Fold Mean)')
        ax.fill_between(epochs, m_val - s_val, m_val + s_val, color=c_val, alpha=0.15)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Component Value")
        if is_log:
            ax.set_yscale('log')
        ax.set_xlim(1, len(epochs))
        ax.legend(loc='upper right', framealpha=0.92)

    plt.suptitle(f"{sensor_name}: Loss Component Convergence & Train-Validation Dynamics (5-Fold Mean $\\pm 1\\sigma$ - Run 3)",
                 fontweight='bold', fontsize=13.5, y=0.995)
    plt.tight_layout()
    save_and_mirror(fig, f"fig_loss_parts_{sensor_code}_run3.png")
    plt.close()


def main():
    print("=" * 80)
    print("GENERATING MULTI-TASK LOSS DECOMPOSITION FIGURES (PHASE C RUN 3)")
    print("=" * 80)

    dfs_ast = load_fold_data('aster')
    dfs_hy = load_fold_data('hytes')

    print(f"Loaded 5 folds for ASTER and 5 folds for HyTES (50 epochs each from Run 3).")

    plot_master_decomposition(dfs_ast, dfs_hy)
    plot_sensor_loss_breakdown(dfs_ast, "ASTER (5 Broad Bands, Triple Atmospheric Ingestion)", "aster")
    plot_sensor_loss_breakdown(dfs_hy, "HyTES (223 Contiguous Bands, Single Surface Ingestion)", "hytes")

    print("\nAll Run 3 loss decomposition figures generated and mirrored successfully!")


if __name__ == '__main__':
    main()

