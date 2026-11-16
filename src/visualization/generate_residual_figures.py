"""
Residual & Error Distributions Analysis Generator (IEEE TGRS Standards)
Generates high-resolution figures (300 DPI) specifically analyzing error residuals:
- Figure 8: Temperature Residual Distributions & Bland-Altman Agreement (PINN vs. Classical TES)
- Figure 9: Per-Material Residual Boxplots and Failure Mode Dissection
Uses scipy.stats.gaussian_kde and matplotlib (zero seaborn dependency).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCHMARK_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'benchmarks')
BASELINE_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'baselines')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'manuscript', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set global publication styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['mathtext.fontset'] = 'cm'

C_PINN = '#1A73E8'     # Google Blue / Sobolev PINN
C_TES = '#D93025'      # Red / Classical TES
C_ACCENT = '#188038'   # Green / Improvement


def generate_figure_8():
    """
    Figure 8: Temperature Residual Distributions & Bland-Altman Agreement (3-Panel)
    Panel A: HyTES Quadrant II (Clean-Room) Error Probability Density (PINN vs. TES)
    Panel B: HyTES Quadrant IV (Sub-Arctic Winter OOD) Error Probability Density (PINN vs. TES)
    Panel C: Bland-Altman Plot (Difference vs. Mean) under Winter OOD with Limits of Agreement (±1.96σ)
    """
    print("Generating Figure 8 (Temperature Residual Distributions & Bland-Altman Agreement)...")

    # Load data
    pinn_q2 = pd.read_csv(os.path.join(BENCHMARK_DIR, 'pinn_eval_hytes_Quadrant_II_GoldenTargets_DevAtm.csv'))
    pinn_q4 = pd.read_csv(os.path.join(BENCHMARK_DIR, 'pinn_eval_hytes_Quadrant_IV_GoldenTargets_WinterAtm_OOD.csv'))
    
    tes_all = pd.read_csv(os.path.join(BASELINE_DIR, 'classical_tes_benchmark_hytes.csv'))
    tes_q2 = tes_all[tes_all['dataset'] == 'golden_quadrant_II'].copy()
    tes_q4 = tes_all[tes_all['dataset'] == 'golden_quadrant_IV_ood'].copy()

    # Residuals: (Predicted - True)
    dt_pinn_q2 = pinn_q2['dt'].values
    dt_tes_q2 = tes_q2['delta_t_k'].values

    dt_pinn_q4 = pinn_q4['dt'].values
    dt_tes_q4 = tes_q4['delta_t_k'].values

    fig = plt.figure(figsize=(16, 5.2), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25])

    # ---------------- PANEL A: Quadrant II KDE ----------------
    ax1 = fig.add_subplot(gs[0])
    x_grid_q2 = np.linspace(-25, 15, 300)
    
    kde_pinn_q2 = gaussian_kde(dt_pinn_q2)(x_grid_q2)
    kde_tes_q2 = gaussian_kde(dt_tes_q2)(x_grid_q2)

    ax1.plot(x_grid_q2, kde_pinn_q2, color=C_PINN, lw=2.4,
             label=f'Sobolev-PINN\n(μ = +{dt_pinn_q2.mean():.2f} K, σ = {dt_pinn_q2.std():.2f} K)')
    ax1.fill_between(x_grid_q2, kde_pinn_q2, color=C_PINN, alpha=0.30)

    ax1.plot(x_grid_q2, kde_tes_q2, color=C_TES, lw=2.4,
             label=f'Classical TES\n(μ = {dt_tes_q2.mean():.2f} K, σ = {dt_tes_q2.std():.2f} K)')
    ax1.fill_between(x_grid_q2, kde_tes_q2, color=C_TES, alpha=0.18)

    ax1.axvline(0, color='black', linestyle='--', lw=1.2, alpha=0.7)
    ax1.set_xlabel(r'Temperature Residual $\Delta T = \hat{T}_s - T_s^{\mathrm{true}}$ (K)', fontweight='bold')
    ax1.set_ylabel('Probability Density', fontweight='bold')
    ax1.set_title('(a) Quadrant II Clean-Room Vault (308 Samples)', fontweight='bold')
    ax1.set_xlim(-25, 15)
    ax1.set_ylim(0, 0.25)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', frameon=True, fontsize=9.0)

    # ---------------- PANEL B: Quadrant IV KDE ----------------
    ax2 = fig.add_subplot(gs[1])
    x_grid_q4 = np.linspace(-45, 20, 300)

    kde_pinn_q4 = gaussian_kde(dt_pinn_q4)(x_grid_q4)
    kde_tes_q4 = gaussian_kde(dt_tes_q4)(x_grid_q4)

    ax2.plot(x_grid_q4, kde_pinn_q4, color=C_PINN, lw=2.4,
             label=f'Sobolev-PINN\n(μ = +{dt_pinn_q4.mean():.2f} K, σ = {dt_pinn_q4.std():.2f} K)')
    ax2.fill_between(x_grid_q4, kde_pinn_q4, color=C_PINN, alpha=0.30)

    ax2.plot(x_grid_q4, kde_tes_q4, color=C_TES, lw=2.4,
             label=f'Classical TES\n(μ = {dt_tes_q4.mean():.2f} K, σ = {dt_tes_q4.std():.2f} K)')
    ax2.fill_between(x_grid_q4, kde_tes_q4, color=C_TES, alpha=0.18)

    ax2.axvline(0, color='black', linestyle='--', lw=1.2, alpha=0.7)
    ax2.set_xlabel(r'Temperature Residual $\Delta T = \hat{T}_s - T_s^{\mathrm{true}}$ (K)', fontweight='bold')
    ax2.set_ylabel('Probability Density', fontweight='bold')
    ax2.set_title('(b) Quadrant IV Sub-Arctic Winter OOD (770 Samples)', fontweight='bold')
    ax2.set_xlim(-45, 20)
    ax2.set_ylim(0, 0.20)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', frameon=True, fontsize=9.0)

    # ---------------- PANEL C: Bland-Altman Plot (Quadrant IV) ----------------
    ax3 = fig.add_subplot(gs[2])
    
    t_true_q4 = pinn_q4['target_temp'].values
    t_pred_pinn = pinn_q4['pred_temp'].values
    t_pred_tes = tes_q4['pred_temp_k'].values

    mean_pinn = (t_pred_pinn + t_true_q4) / 2.0
    diff_pinn = t_pred_pinn - t_true_q4

    mean_tes = (t_pred_tes + t_true_q4) / 2.0
    diff_tes = t_pred_tes - t_true_q4

    # Scatter
    ax3.scatter(mean_tes, diff_tes, color=C_TES, alpha=0.35, s=20, label='Classical TES Residuals', edgecolors='none')
    ax3.scatter(mean_pinn, diff_pinn, color=C_PINN, alpha=0.65, s=22, label='Sobolev-PINN Residuals', edgecolors='none')

    # Mean and Limits of Agreement (±1.96σ) for PINN
    mu_p = diff_pinn.mean()
    sd_p = diff_pinn.std()
    ax3.axhline(mu_p, color=C_PINN, linestyle='-', lw=1.6, label=f'PINN Bias ({mu_p:+.2f} K)')
    ax3.axhline(mu_p + 1.96 * sd_p, color=C_PINN, linestyle='--', lw=1.2, alpha=0.85)
    ax3.axhline(mu_p - 1.96 * sd_p, color=C_PINN, linestyle='--', lw=1.2, alpha=0.85)

    # Limits for TES
    mu_t = diff_tes.mean()
    sd_t = diff_tes.std()
    ax3.axhline(mu_t, color=C_TES, linestyle='-', lw=1.6, label=f'TES Bias ({mu_t:+.2f} K)')
    ax3.axhline(mu_t + 1.96 * sd_t, color=C_TES, linestyle='--', lw=1.2, alpha=0.85)
    ax3.axhline(mu_t - 1.96 * sd_t, color=C_TES, linestyle='--', lw=1.2, alpha=0.85)

    ax3.set_xlabel(r'Mean Temperature $(T_{\mathrm{pred}} + T_{\mathrm{true}})/2$ (K)', fontweight='bold')
    ax3.set_ylabel(r'Residual $T_{\mathrm{pred}} - T_{\mathrm{true}}$ (K)', fontweight='bold')
    ax3.set_title('(c) Bland-Altman Agreement Plot (Quadrant IV Winter OOD)', fontweight='bold')
    ax3.set_xlim(260, 335)
    ax3.set_ylim(-45, 25)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='lower left', frameon=True, fontsize=8.5, ncol=2)

    # Annotations on limits of agreement
    ax3.text(333, mu_p + 1.96 * sd_p, f'+1.96σ: {mu_p + 1.96*sd_p:.1f} K', color=C_PINN, fontsize=8, va='bottom', ha='right', fontweight='bold')
    ax3.text(333, mu_p - 1.96 * sd_p, f'-1.96σ: {mu_p - 1.96*sd_p:.1f} K', color=C_PINN, fontsize=8, va='top', ha='right', fontweight='bold')
    ax3.text(333, mu_t - 1.96 * sd_t, f'-1.96σ: {mu_t - 1.96*sd_t:.1f} K', color=C_TES, fontsize=8, va='top', ha='right', fontweight='bold')

    plt.suptitle('Figure 8: Error Residual Distributions and Bland-Altman Limits of Agreement (HyTES 256ch)',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig8_temperature_residual_distributions.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Generated Figure 8: {out_path}")


def generate_figure_9():
    """
    Figure 9: Per-Material Residual Boxplots and Failure Mode Dissection
    Panel A: HyTES Residual Boxplots by Mineral Class (Comparing PINN vs. TES)
    Panel B: Absolute Error vs. Surface Kinetic Temperature (Temperature Dependence Invariance)
    """
    print("Generating Figure 9 (Per-Material Residual Boxplots & Thermal Stability)...")

    pinn_q4 = pd.read_csv(os.path.join(BENCHMARK_DIR, 'pinn_eval_hytes_Quadrant_IV_GoldenTargets_WinterAtm_OOD.csv'))
    tes_all = pd.read_csv(os.path.join(BASELINE_DIR, 'classical_tes_benchmark_hytes.csv'))
    tes_q4 = tes_all[tes_all['dataset'] == 'golden_quadrant_IV_ood'].copy()

    classes = [
        'quartz', 'feldspar', 'kaolinite', 'calcite', 
        'urban_concrete', 'urban_asphalt', 'vegetation_green_leaf'
    ]
    labels = ['Quartz', 'Feldspar', 'Kaolinite', 'Calcite', 'Concrete', 'Asphalt', 'NPV Bark']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    # ---------------- PANEL A: Grouped Boxplots ----------------
    pinn_data = [pinn_q4[pinn_q4['material_class'] == c]['dt'].values for c in classes]
    tes_data = [tes_q4[tes_q4['material_class'] == c]['delta_t_k'].values for c in classes]

    x = np.arange(len(classes))
    width = 0.32

    # Plot PINN boxplots
    bp1 = ax1.boxplot(pinn_data, positions=x - width/2, widths=width*0.85, patch_artist=True,
                      boxprops=dict(facecolor=C_PINN, color='black', alpha=0.7),
                      medianprops=dict(color='white', lw=1.5),
                      flierprops=dict(marker='o', markersize=3, markerfacecolor=C_PINN, alpha=0.5, markeredgecolor='none'))

    # Plot TES boxplots
    bp2 = ax1.boxplot(tes_data, positions=x + width/2, widths=width*0.85, patch_artist=True,
                      boxprops=dict(facecolor=C_TES, color='black', alpha=0.7),
                      medianprops=dict(color='white', lw=1.5),
                      flierprops=dict(marker='o', markersize=3, markerfacecolor=C_TES, alpha=0.5, markeredgecolor='none'))

    ax1.axhline(0, color='black', linestyle='--', lw=1.2, alpha=0.7)
    ax1.set_ylabel(r'Temperature Residual $\Delta T = \hat{T}_s - T_s^{\mathrm{true}}$ (K)', fontweight='bold')
    ax1.set_xlabel('Quarantined Mineral / Terrestrial Target', fontweight='bold')
    ax1.set_title('(a) Temperature Residual Distribution by Target Endmember', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight='bold', fontsize=9.5)
    ax1.set_ylim(-45, 20)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)

    # Custom legend
    import matplotlib.patches as mpatches
    patch1 = mpatches.Patch(facecolor=C_PINN, label='Sobolev-PINN (Tight IQRs around 0 K)', alpha=0.7)
    patch2 = mpatches.Patch(facecolor=C_TES, label='Classical TES (High Scatter & Negative Bias)', alpha=0.7)
    ax1.legend(handles=[patch1, patch2], loc='lower left', frameon=True)

    # Callout for Feldspar
    ax1.annotate('Classical TES Catastrophic Collapse\n(Residual < -40 K due to MMD Failure)',
                 xy=(1 + width/2, -38), xytext=(0.8, -43),
                 arrowprops=dict(facecolor=C_TES, arrowstyle='->', lw=1.2),
                 fontsize=8.5, fontweight='bold', color=C_TES)

    # ---------------- PANEL B: Error vs. Surface Temperature ----------------
    temp_levels = sorted(pinn_q4['target_temp'].unique())
    pinn_temp_mae = [pinn_q4[pinn_q4['target_temp'] == t]['abs_dt'].mean() for t in temp_levels]
    tes_temp_mae = [tes_q4[tes_q4['true_temp_k'] == t]['abs_delta_t_k'].mean() for t in temp_levels]

    ax2.plot(temp_levels, tes_temp_mae, color=C_TES, marker='s', lw=2.2, markersize=7,
             label='Classical TES (Strong Cold-Temperature Instability)')
    ax2.plot(temp_levels, pinn_temp_mae, color=C_PINN, marker='o', lw=2.2, markersize=7,
             label='Sobolev-PINN (Uniform High Precision Across All Temperatures)')

    # Add Mission Baseline (1.0 K)
    ax2.axhline(1.0, color='darkgreen', linestyle='--', lw=1.4, label='NASA SBG / ESA LSTM Requirement (1.0 K)')

    ax2.set_xlabel('Ground-Truth Surface Temperature $T_s$ (K)', fontweight='bold')
    ax2.set_ylabel(r'Mean Absolute Error $\mathrm{MAE}_T$ (K)', fontweight='bold')
    ax2.set_title('(b) Error Invariance Across Terrestrial Kinetic Temperatures', fontweight='bold')
    ax2.set_ylim(0, 10)
    ax2.set_xticks(temp_levels)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', frameon=True, fontsize=9.0)

    plt.suptitle('Figure 9: Mineralogical and Thermal Error Dissection in Out-of-Distribution Winter Atmospheres',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig9_residual_boxplots_by_mineral.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Generated Figure 9: {out_path}")


if __name__ == '__main__':
    print("="*70)
    print("STARTING RESIDUAL & ERROR DISTRIBUTIONS GENERATION")
    print("="*70)
    generate_figure_8()
    generate_figure_9()
    print("="*70)
    print("RESIDUAL FIGURES GENERATED SUCCESSFULLY IN manuscript/figures/")
    print("="*70)
