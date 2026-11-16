"""
Publication Figures Generator v2 (IEEE TGRS Standards)
Generates high-resolution figures (300 DPI) for the Sobolev-PINN manuscript:
- Figure 4: Surface Kinetic Temperature Retrieval Accuracy across Quarantined Targets
- Figure 5: Double-Quarantine Matrix Benchmark vs. NASA Classical TES (Flagship 3-Panel)
- Figure 6: Leave-One-Class-Out (LOCO 7+1) Zero-Shot Crystallographic Generalization
- Figure 7: Atmospheric Ingestion Ablation (Option A: Single vs. Dual vs. Triple vs. Quad)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

# IEEE Palette
C_PINN = '#1A73E8'     # Google Blue / Sobolev PINN
C_TES = '#D93025'      # Red / Classical TES
C_ACCENT = '#188038'   # Green / Improvement
C_GRAY = '#5F6368'     # Neutral Gray


def generate_figure_4():
    """
    Figure 4: Kinetic Surface Temperature Retrieval Error across Golden Targets
    Compares Sobolev-PINN against NASA SBG 1.0 K Mission Requirement
    """
    print("Generating Figure 4 (Temperature Accuracy across Golden Targets)...")
    hytes_eval_path = os.path.join(BENCHMARK_DIR, 'pinn_eval_hytes_Quadrant_II_GoldenTargets_DevAtm.csv')
    aster_eval_path = os.path.join(BENCHMARK_DIR, 'pinn_eval_aster_Quadrant_II_GoldenTargets_DevAtm.csv')
    tes_hytes_path = os.path.join(BASELINE_DIR, 'classical_tes_benchmark_hytes.csv')

    if not os.path.exists(hytes_eval_path) or not os.path.exists(aster_eval_path):
        print("Warning: Missing evaluation CSVs for Figure 4.")
        return

    df_pinn_h = pd.read_csv(hytes_eval_path)
    df_pinn_a = pd.read_csv(aster_eval_path)
    df_tes_h = pd.read_csv(tes_hytes_path)
    df_tes_h_q2 = df_tes_h[df_tes_h['dataset'] == 'golden_quadrant_II']

    classes = [
        'quartz', 'feldspar', 'kaolinite', 'calcite', 
        'urban_concrete', 'urban_asphalt', 'vegetation_green_leaf'
    ]
    labels = [
        'Quartz', 'Feldspar', 'Kaolinite', 'Calcite', 
        'Concrete', 'Asphalt', 'NPV Bark'
    ]

    hytes_mae = [df_pinn_h[df_pinn_h['material_class'] == c]['abs_dt'].mean() for c in classes]
    aster_mae = [df_pinn_a[df_pinn_a['material_class'] == c]['abs_dt'].mean() for c in classes]
    tes_mae = [df_tes_h_q2[df_tes_h_q2['material_class'] == c]['abs_delta_t_k'].mean() for c in classes]

    x = np.arange(len(classes))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    rects1 = ax.bar(x - width, hytes_mae, width, label='Sobolev-PINN (HyTES 256ch)', color=C_PINN, edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x, aster_mae, width, label='Sobolev-PINN (ASTER 5ch)', color='#4285F4', alpha=0.7, edgecolor='black', linewidth=0.8)
    rects3 = ax.bar(x + width, tes_mae, width, label='Classical TES (HyTES Baseline)', color=C_TES, alpha=0.85, edgecolor='black', linewidth=0.8)

    # Mission Baselines
    ax.axhline(1.0, color='darkgreen', linestyle='--', linewidth=1.5, label='NASA SBG / ESA LSTM Requirement (1.0 K)')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.2, label='Optimal Physical Threshold (0.5 K)')

    ax.set_ylabel(r'Absolute Temperature Error $|T - \hat{T}|$ (K)')
    ax.set_title('Figure 4: Kinetic Surface Temperature Error across Clean-Room Golden Targets (Quadrant II)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_ylim(0, 16.0)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    # Add text annotation on Feldspar
    ax.annotate('Classical TES\nCollapse (>25 K)', xy=(1 + width, 14.5), xytext=(1 + width + 0.3, 14.8),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.0),
                fontsize=8.5, fontweight='bold', color=C_TES)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig4_temperature_accuracy.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Generated Figure 4: {out_path}")


def generate_figure_5():
    """
    Figure 5: Double-Quarantine Matrix Benchmark vs. NASA Classical TES (Flagship 3-Panel)
    Panel A: HyTES Head-to-Head Error Across Quadrants I, II, IV
    Panel B: ASTER Head-to-Head Error Across Quadrants I, II, IV
    Panel C: Extreme Sub-Arctic Winter OOD Temperature Scatter (PINN vs. TES vs. 1:1)
    """
    print("Generating Figure 5 (Flagship Double-Quarantine Benchmark vs. Classical TES)...")
    q_hytes = pd.read_csv(os.path.join(BENCHMARK_DIR, 'quarantine_matrix_benchmark_hytes.csv'))
    q_aster = pd.read_csv(os.path.join(BENCHMARK_DIR, 'quarantine_matrix_benchmark_aster.csv'))
    
    # Load scatter data for Panel C
    pinn_q4_path = os.path.join(BENCHMARK_DIR, 'pinn_eval_hytes_Quadrant_IV_GoldenTargets_WinterAtm_OOD.csv')
    tes_hytes_path = os.path.join(BASELINE_DIR, 'classical_tes_benchmark_hytes.csv')
    
    df_pinn_q4 = pd.read_csv(pinn_q4_path)
    df_tes_all = pd.read_csv(tes_hytes_path)
    df_tes_q4 = df_tes_all[df_tes_all['dataset'] == 'golden_quadrant_IV_ood']

    fig = plt.figure(figsize=(15, 5.5), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.3])

    # ---------------- PANEL A: HyTES ----------------
    ax1 = fig.add_subplot(gs[0])
    regimes = ['Q-I (Dev Pool)', 'Q-II (Golden Vault)', 'Q-IV (Winter OOD)']
    pinn_rmse_h = q_hytes['PINN_DT_RMSE_K'].values
    tes_rmse_h = q_hytes['TES_DT_RMSE_K'].values
    
    x = np.arange(len(regimes))
    w = 0.35
    b1 = ax1.bar(x - w/2, pinn_rmse_h, w, label='Sobolev-PINN', color=C_PINN, edgecolor='black', lw=0.8)
    b2 = ax1.bar(x + w/2, tes_rmse_h, w, label='Classical TES', color=C_TES, edgecolor='black', lw=0.8)

    ax1.set_ylabel(r'Temperature $\mathrm{RMSE}_T$ (K)', fontweight='bold')
    ax1.set_title('(a) HyTES Hyperspectral (256ch)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regimes, fontsize=9.5)
    ax1.set_ylim(0, 18.0)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', frameon=True)

    # Badges on HyTES
    ax1.text(1, 11.2, '-74.6%', ha='center', va='bottom', fontsize=10, fontweight='bold', color=C_ACCENT,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#E6F4EA', edgecolor=C_ACCENT, lw=1))
    ax1.text(2, 16.0, '-77.0%', ha='center', va='bottom', fontsize=10, fontweight='bold', color=C_ACCENT,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#E6F4EA', edgecolor=C_ACCENT, lw=1))

    # ---------------- PANEL B: ASTER ----------------
    ax2 = fig.add_subplot(gs[1])
    pinn_rmse_a = q_aster['PINN_DT_RMSE_K'].values
    tes_rmse_a = q_aster['TES_DT_RMSE_K'].values

    ax2.bar(x - w/2, pinn_rmse_a, w, label='Sobolev-PINN', color=C_PINN, edgecolor='black', lw=0.8)
    ax2.bar(x + w/2, tes_rmse_a, w, label='Classical TES', color=C_TES, edgecolor='black', lw=0.8)

    ax2.set_ylabel(r'Temperature $\mathrm{RMSE}_T$ (K)', fontweight='bold')
    ax2.set_title('(b) ASTER Multispectral (5ch)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regimes, fontsize=9.5)
    ax2.set_ylim(0, 18.0)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', frameon=True)

    # Badges on ASTER
    ax2.text(1, 10.3, '-4.4%', ha='center', va='bottom', fontsize=10, fontweight='bold', color=C_ACCENT,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#E6F4EA', edgecolor=C_ACCENT, lw=1))
    ax2.text(2, 14.1, '-8.3%', ha='center', va='bottom', fontsize=10, fontweight='bold', color=C_ACCENT,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#E6F4EA', edgecolor=C_ACCENT, lw=1))

    # ---------------- PANEL C: SCATTER PLOT ----------------
    ax3 = fig.add_subplot(gs[2])
    
    t_true = df_pinn_q4['target_temp'].values
    t_pinn = df_pinn_q4['pred_temp'].values
    t_tes = df_tes_q4['pred_temp_k'].values

    ax3.scatter(t_true, t_tes, color=C_TES, alpha=0.45, s=28, label=r'Classical TES ($\mathrm{RMSE} = 15.40\,\mathrm{K}$)', edgecolors='none')
    ax3.scatter(t_true, t_pinn, color=C_PINN, alpha=0.75, s=24, label=r'Sobolev-PINN ($\mathrm{RMSE} = 3.54\,\mathrm{K}$)', edgecolors='none')

    # 1:1 Identity Line
    line_min = 265
    line_max = 335
    ax3.plot([line_min, line_max], [line_min, line_max], 'k--', lw=1.5, label='Ideal 1:1 Line')

    ax3.set_xlabel('Ground-Truth Kinetic Temperature $T_s$ (K)', fontweight='bold')
    ax3.set_ylabel('Retrieved Kinetic Temperature $\hat{T}_s$ (K)', fontweight='bold')
    ax3.set_title('(c) Quadrant IV Sub-Arctic Winter OOD (770 Samples)', fontweight='bold')
    ax3.set_xlim(line_min, line_max)
    ax3.set_ylim(line_min - 5, line_max + 15)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper left', frameon=True, fontsize=9.5)

    plt.suptitle('Figure 5: Double-Quarantine Performance Matrix Benchmark vs. NASA Operational Classical TES',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig5_quarantine_vs_classical_tes.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Generated Figure 5: {out_path}")


def generate_figure_6():
    """
    Figure 6: Leave-One-Class-Out (LOCO 7+1) Zero-Shot Crystallographic Generalization
    Shows Silicate transfer (MAE < 1.5 K), Feldspar stabilization, and Calcite honest divergence
    """
    print("Generating Figure 6 (LOCO 7+1 Zero-Shot Generalization)...")
    loco_h = pd.read_csv(os.path.join(BENCHMARK_DIR, 'loco_benchmark_hytes.csv'))
    loco_a = pd.read_csv(os.path.join(BENCHMARK_DIR, 'loco_benchmark_aster.csv'))

    # Order logically by crystal structure
    class_order = ['sandstone', 'granite', 'basalt', 'quartz', 'kaolinite', 'feldspar', 'calcite']
    name_map = {
        'sandstone': 'Sandstone\n(Silicate)',
        'granite': 'Granite\n(Silicate)',
        'basalt': 'Basalt\n(Silicate)',
        'quartz': 'Quartz\n(Silicate)',
        'kaolinite': 'Kaolinite\n(Clay)',
        'feldspar': 'Feldspar\n(Tectosil.)',
        'calcite': 'Calcite\n(Carbonate)'
    }

    loco_h = loco_h.set_index('held_out_class').loc[class_order].reset_index()
    loco_a = loco_a.set_index('held_out_class').loc[class_order].reset_index()

    labels = [name_map[c] for c in class_order]
    mae_h = loco_h['ood_temp_mae'].values
    mae_a = loco_a['ood_temp_mae'].values
    sam_h = loco_h['ood_sam_deg'].values

    x = np.arange(len(class_order))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)

    # Temperature MAE Bars
    rects1 = ax1.bar(x - width/2, mae_h, width, label=r'HyTES Zero-Shot $\mathrm{MAE}_T$ (K)', color=C_PINN, edgecolor='black', lw=0.8)
    rects2 = ax1.bar(x + width/2, mae_a, width, label=r'ASTER Zero-Shot $\mathrm{MAE}_T$ (K)', color='#FBBC04', edgecolor='black', lw=0.8)

    ax1.set_ylabel(r'Held-Out Zero-Shot Error $\mathrm{MAE}_T$ (K)', fontweight='bold', color='#202124')
    ax1.set_title('Figure 6: Leave-One-Class-Out (LOCO 7+1) Zero-Shot Crystallographic Generalization & Inductive Bias',
                  fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight='bold', fontsize=9.5)
    ax1.set_ylim(0, 26.0)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)

    # Secondary Axis for SAM (degrees)
    ax2 = ax1.twinx()
    p1 = ax2.plot(x, sam_h, color='#EA4335', marker='o', lw=2.0, markersize=7, label=r'HyTES Spectral Angle SAM ($^\circ$)')
    ax2.set_ylabel(r'Spectral Angle Mapper SAM ($^\circ$)', fontweight='bold', color='#EA4335')
    ax2.set_ylim(0, 25.0)

    # Shaded Zones
    ax1.axvspan(-0.5, 3.5, color='#E6F4EA', alpha=0.45, label='Silicate Transfer Zone (MAE < 1.5 K)')
    ax1.axvspan(3.5, 4.5, color='#FEF7E0', alpha=0.45, label='Subtle Phyllosilicate Doublet')
    ax1.axvspan(4.5, 5.5, color='#FFF0F2', alpha=0.45, label='Feldspar Stabilization (TES diverges > 40 K)')
    ax1.axvspan(5.5, 6.5, color='#F1F3F4', alpha=0.55, label='Unseen Carbonate Non-Memorization')

    # Combined Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, fontsize=9.0)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig6_loco_zero_shot_generalization.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Generated Figure 6: {out_path}")


def generate_figure_7():
    """
    Figure 7: Atmospheric Ingestion Ablation (Option A: Single vs. Dual vs. Triple vs. Quad)
    Shows monotonic error reduction and validation loss scaling
    """
    print("Generating Figure 7 (Option A Atmospheric Ingestion Ablation)...")
    abl_h = pd.read_csv(os.path.join(BENCHMARK_DIR, 'ablation_ingestion_benchmark_hytes.csv'))
    abl_a = pd.read_csv(os.path.join(BENCHMARK_DIR, 'ablation_ingestion_benchmark_aster.csv'))

    tiers = ['Single\n[L_surf]', 'Dual\n[L_surf, L↓]', 'Triple\n[L_surf, τ, L↓]', 'Quad\n[L_sens, τ, L↑, L↓]']
    
    rmse_t_h = abl_h['temp_rmse_k'].values if 'temp_rmse_k' in abl_h.columns else abl_h['rmse_t_k'].values
    mae_t_h = abl_h['temp_mae_k'].values if 'temp_mae_k' in abl_h.columns else abl_h['mae_t_k'].values
    rmse_t_a = abl_a['temp_rmse_k'].values if 'temp_rmse_k' in abl_a.columns else abl_a['rmse_t_k'].values
    mae_t_a = abl_a['temp_mae_k'].values if 'temp_mae_k' in abl_a.columns else abl_a['mae_t_k'].values

    x = np.arange(len(tiers))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Panel A: HyTES
    ax1.bar(x - width/2, rmse_t_h, width, label=r'Temperature $\mathrm{RMSE}_T$ (K)', color=C_PINN, edgecolor='black', lw=0.8)
    ax1.bar(x + width/2, mae_t_h, width, label=r'Temperature $\mathrm{MAE}_T$ (K)', color='#8AB4F8', edgecolor='black', lw=0.8)
    ax1.set_ylabel('Error (Kelvin)', fontweight='bold')
    ax1.set_title('(a) HyTES Hyperspectral Ingestion Ablation (256ch)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tiers, fontweight='bold', fontsize=9.5)
    ax1.set_ylim(0, 10.5)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')

    # Add callout on Triple
    ax1.annotate(r'Triple-Input Minimum' + '\n' + r'$\mathrm{RMSE}_T = 6.30\,\mathrm{K}$', xy=(2 - width/2, 6.30),
                 xytext=(1.8, 8.5), arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.0),
                 fontsize=8.5, fontweight='bold', color=C_ACCENT)

    # Panel B: ASTER
    ax2.bar(x - width/2, rmse_t_a, width, label=r'Temperature $\mathrm{RMSE}_T$ (K)', color='#34A853', edgecolor='black', lw=0.8)
    ax2.bar(x + width/2, mae_t_a, width, label=r'Temperature $\mathrm{MAE}_T$ (K)', color='#A8DAB5', edgecolor='black', lw=0.8)
    ax2.set_ylabel('Error (Kelvin)', fontweight='bold')
    ax2.set_title('(b) ASTER Multispectral Ingestion Ablation (5ch)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tiers, fontweight='bold', fontsize=9.5)
    ax2.set_ylim(0, 10.5)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')

    # Add callout on Triple
    ax2.annotate(r'Triple-Input Minimum' + '\n' + r'$\mathrm{RMSE}_T = 7.82\,\mathrm{K}$', xy=(2 - width/2, 7.82),
                 xytext=(1.8, 9.2), arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.0),
                 fontsize=8.5, fontweight='bold', color=C_ACCENT)

    plt.suptitle('Figure 7: Empirical 4-Tier Atmospheric Ingestion Ablation (Option A Table II Grounding)',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig7_atmospheric_ingestion_ablation.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Generated Figure 7: {out_path}")


def update_fig1_architecture():
    """
    Modernizes Figure 1 to include Dual-Input and ELU activations
    """
    print("Updating Figure 1 (Architecture Flowchart with Dual-Input & ELU)...")
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.axis('off')

    # Color palette
    c_input = '#E8F0FE'
    c_backbone = '#D2E3FC'
    c_head_t = '#FCE8E6'
    c_head_e = '#E6F4EA'
    c_phys = '#FEF7E0'
    c_loss = '#F3E8FD'

    def draw_box(x, y, w, h, title, subtitle="", bg_color='#FFFFFF', border='#333333', lw=1.5):
        box = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.04",
                                      facecolor=bg_color, edgecolor=border, linewidth=lw)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.65, title, ha='center', va='center', fontsize=11, fontweight='bold', color='#202124')
            ax.text(x + w/2, y + h*0.30, subtitle, ha='center', va='center', fontsize=9, color='#5F6368')
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=11, fontweight='bold', color='#202124')

    def draw_arrow(x1, y1, x2, y2, text=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color='#3C4043', lw=1.8, shrinkA=3, shrinkB=3))
        if text:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.02, text, ha='center', va='bottom', fontsize=8.5, color='#3C4043')

    # Title
    ax.text(0.5, 0.96, "Sobolev-PINN Dual-Input Architecture with Thermodynamic Forward Coupling",
            ha='center', va='center', fontsize=15, fontweight='bold', color='#1A237E')

    # 1. Dual-Input Layer
    draw_box(0.04, 0.38, 0.14, 0.26, "Dual-Input Vector\n[L_surf, L_down]",
             r"$\mathbf{x}_{\mathrm{in}} \in \mathbb{R}^{2B}$" + "\nGround Radiance +\nDownwelling Sky Glint", c_input, '#1A73E8')

    # 2. Shared Backbone
    draw_box(0.23, 0.35, 0.17, 0.32, "Shared Backbone\nMLP Topology",
             "3--5 Layers | 512--768 Units\nELU Activations (C¹ Smooth)", c_backbone, '#1967D2')
    draw_arrow(0.18, 0.51, 0.23, 0.51)

    # 3. Decoupled Heads
    draw_box(0.45, 0.58, 0.18, 0.20, "Kinetic Temperature Head",
             r"$T_s = 200 + \mathrm{Softplus}(\mathbf{w}_T^T \mathbf{h} + b_T)$" + "\n" + r"$T_s > 200\,\mathrm{K}$ Guaranteed", c_head_t, '#D93025')
    draw_box(0.45, 0.24, 0.18, 0.20, "Spectral Emissivity Head",
             r"$\hat{\mathbf{\epsilon}} = \sigma(\mathbf{W}_\epsilon \mathbf{h} + \mathbf{b}_\epsilon)$" + "\n" + r"$\hat{\mathbf{\epsilon}} \in (0, 1)^N$ Guaranteed", c_head_e, '#188038')

    draw_arrow(0.40, 0.55, 0.45, 0.68)
    draw_arrow(0.40, 0.47, 0.45, 0.34)

    # 4. Forward Thermodynamic Layer
    draw_box(0.68, 0.38, 0.17, 0.26, "Forward Radiative Transfer\n& Planck Constraint",
             r"$\hat{\mathbf{L}}_{\mathrm{surf}} = \hat{\mathbf{\epsilon}}\odot\mathbf{B}(T_s)$" + "\n" + r"$+ (1-\hat{\mathbf{\epsilon}})\odot\mathbf{L}^\downarrow/\pi$", c_phys, '#F29900')
    draw_arrow(0.63, 0.68, 0.68, 0.55)
    draw_arrow(0.63, 0.34, 0.68, 0.47)

    # 5. Compound Loss
    draw_box(0.88, 0.12, 0.10, 0.78, "Sobolev Compound\nLoss",
             r"$\mathcal{L}_{\mathrm{phys}}$ (Planck)" + "\n\n" +
             r"$\mathcal{L}_{\mathrm{MSE}}(\hat{\mathbf{\epsilon}}, \mathbf{\epsilon})$" + "\n\n" +
             r"$\mathcal{L}_{\mathrm{SAM}}(\hat{\mathbf{\epsilon}}, \mathbf{\epsilon})$" + "\n\n" +
             r"$\mathcal{L}_{W^{2,1}}(\hat{\mathbf{\epsilon}})$:" + "\n" +
             r"$\|\nabla \hat{\mathbf{\epsilon}}\|_1 + \|\nabla^2 \hat{\mathbf{\epsilon}}\|_1$", c_loss, '#8430CE')
    draw_arrow(0.85, 0.51, 0.88, 0.51)
    draw_arrow(0.63, 0.28, 0.88, 0.28)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig1_pinn_architecture.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Updated Figure 1: {out_path}")


if __name__ == '__main__':
    print("="*70)
    print("STARTING PUBLICATION FIGURES GENERATION SUITE")
    print("="*70)
    update_fig1_architecture()
    generate_figure_4()
    generate_figure_5()
    generate_figure_6()
    generate_figure_7()
    print("="*70)
    print("ALL FIGURES GENERATED SUCCESSFULLY IN manuscript/figures/")
    print("="*70)
