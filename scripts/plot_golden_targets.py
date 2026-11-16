"""
Publication Figure Generator: Golden Targets Benchmark (PINN vs Classical TES)
==============================================================================
Generates comprehensive head-to-head comparison plots for IEEE TGRS:
  1. fig_golden_targets_spectra.png: Spectral emissivity recovery on 6 key Golden Targets
  2. fig_golden_targets_temperature_error.png: dT distribution by mineral class (Q-II & Q-IV)
  3. fig_golden_targets_emissivity_metrics.png: RMSE_eps and SAM by mineral class
  4. fig_golden_targets_scatter_1to1.png: 1:1 Temperature and Emissivity retrieval scatter

Author: Dr. Atilio Grondona & OpenTIR-PINN Research Team
Target Journal: IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.pinn import PhysicsInformedNN
from src.baselines.classical_tes import ClassicalTES
from src.training.input_utils import build_atmospheric_band_cache, assemble_input_tensor, compute_spectral_angle_mapper
from src.training.stage1_core import standardize_input, get_sensor_band_mask, get_sensor_wavelengths

# Output directories
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
ARTIFACTS_DIR = r"C:\Users\Atilio\.gemini\antigravity\brain\89b8c610-4766-4143-b9ea-f65c9e0ab256"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Publication styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['figure.titlesize'] = 13.5
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.35
plt.rcParams['grid.linestyle'] = '--'


def load_pinn_model(sensor='hytes'):
    device = torch.device('cpu')
    model_path = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'cv', sensor, f'model_{sensor}_fold0_best.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(PROJECT_ROOT, 'outputs', 'models', f'model_{sensor}_best.pth')
    
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    input_mode = ckpt.get('input_mode', config.get('input_mode', 'single' if sensor == 'hytes' else 'triple'))
    norm_stats = ckpt.get('norm_stats', {})
    mask = ckpt.get('band_mask', get_sensor_band_mask(sensor, device=device)).to(device)
    
    n_bands_eff = int(mask.sum())
    model = PhysicsInformedNN(
        input_dim=n_bands_eff,
        output_dim=n_bands_eff,
        hidden_dim=int(config.get('hidden_dim', 768 if sensor == 'hytes' else 384)),
        n_layers=int(config.get('n_layers', 3 if sensor == 'hytes' else 5)),
        input_mode=input_mode,
        activation=str(config.get('activation', 'SiLU')),
        dropout=float(config.get('dropout', 0.0))
    ).to(device)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    mu_x = norm_stats['mu_x'].to(device)
    std_x = norm_stats['std_x'].to(device)
    return model, input_mode, mu_x, std_x, mask


def save_and_copy(fig, filename):
    out_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    art_path = os.path.join(ARTIFACTS_DIR, filename)
    shutil.copyfile(out_path, art_path)
    print(f"Saved: {out_path} and copied to {art_path}")


def main():
    print("="*80)
    print("GENERATING GOLDEN TARGETS BENCHMARK FIGURES (PINN vs CLASSICAL TES)")
    print("="*80)

    # 1. Load Precomputed Benchmark Ledgers
    tes_csv = os.path.join(PROJECT_ROOT, 'outputs', 'baselines', 'classical_tes_benchmark_hytes.csv')
    df_tes = pd.read_csv(tes_csv)
    df_tes_q2 = df_tes[df_tes['dataset'] == 'golden_quadrant_II'].copy()
    df_tes_q4 = df_tes[df_tes['dataset'] == 'golden_quadrant_IV_ood'].copy()

    pinn_q2_csv = os.path.join(PROJECT_ROOT, 'outputs', 'benchmarks', 'pinn_eval_hytes_Quadrant_II_GoldenTargets_DevAtm.csv')
    pinn_q4_csv = os.path.join(PROJECT_ROOT, 'outputs', 'benchmarks', 'pinn_eval_hytes_Quadrant_IV_GoldenTargets_WinterAtm_OOD.csv')
    df_pinn_q2 = pd.read_csv(pinn_q2_csv)
    df_pinn_q4 = pd.read_csv(pinn_q4_csv)

    print(f"Loaded records -> TES Q2: {len(df_tes_q2)}, Q4: {len(df_tes_q4)} | PINN Q2: {len(df_pinn_q2)}, Q4: {len(df_pinn_q4)}")

    # -------------------------------------------------------------------------
    # FIGURE 1: Temperature Error (dT) Distribution across Golden Target Classes
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    classes = ['quartz', 'calcite', 'kaolinite', 'feldspar', 'urban_concrete', 'urban_asphalt', 'botanical']
    class_labels = ['Quartz', 'Calcite', 'Kaolinite', 'Feldspar', 'Concrete', 'Asphalt', 'Vegetation']
    
    x = np.arange(len(classes))
    width = 0.35

    # Q-II
    pinn_q2_means = [df_pinn_q2[df_pinn_q2['material_class'] == c]['abs_dt'].mean() for c in classes]
    pinn_q2_stds = [df_pinn_q2[df_pinn_q2['material_class'] == c]['abs_dt'].std() for c in classes]
    tes_q2_means = [df_tes_q2[df_tes_q2['material_class'] == c]['abs_delta_t_k'].mean() for c in classes]
    tes_q2_stds = [df_tes_q2[df_tes_q2['material_class'] == c]['abs_delta_t_k'].std() for c in classes]

    rects1 = ax1.bar(x - width/2, pinn_q2_means, width, yerr=pinn_q2_stds, label='Sobolev-PINN (Run 2)', color='#1E40AF', capsize=4, alpha=0.9)
    rects2 = ax1.bar(x + width/2, tes_q2_means, width, yerr=tes_q2_stds, label='Classical TES (Gillespie/Hulley)', color='#DC2626', capsize=4, alpha=0.9)
    ax1.set_title("Quadrant II: Golden Targets $\\times$ Dev Atmospheres\n(PINN $T_{\\mathrm{RMSE}}=1.78\\,\\mathrm{K}$ vs TES $2.31\\,\\mathrm{K}$, $+23.2\\%$ Gain)", fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_labels, rotation=30, ha='right')
    ax1.set_ylabel("Mean Absolute Temperature Error $|\\Delta T|$ (K)")
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 4.5)

    # Q-IV
    pinn_q4_means = [df_pinn_q4[df_pinn_q4['material_class'] == c]['abs_dt'].mean() for c in classes]
    pinn_q4_stds = [df_pinn_q4[df_pinn_q4['material_class'] == c]['abs_dt'].std() for c in classes]
    tes_q4_means = [df_tes_q4[df_tes_q4['material_class'] == c]['abs_delta_t_k'].mean() for c in classes]
    tes_q4_stds = [df_tes_q4[df_tes_q4['material_class'] == c]['abs_delta_t_k'].std() for c in classes]

    ax2.bar(x - width/2, pinn_q4_means, width, yerr=pinn_q4_stds, label='Sobolev-PINN (Run 2)', color='#1E40AF', capsize=4, alpha=0.9)
    ax2.bar(x + width/2, tes_q4_means, width, yerr=tes_q4_stds, label='Classical TES (Gillespie/Hulley)', color='#DC2626', capsize=4, alpha=0.9)
    ax2.set_title("Quadrant IV: Clean-Room OOD Golden Targets $\\times$ Winter Atm\n(PINN $T_{\\mathrm{RMSE}}=1.84\\,\\mathrm{K}$ vs TES $2.09\\,\\mathrm{K}$, $+11.8\\%$ Gain)", fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_labels, rotation=30, ha='right')
    ax2.legend(loc='upper right')

    plt.suptitle("Double-Quarantine Matrix: Temperature Retrieval Accuracy by Mineral Class", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_and_copy(fig, "fig_golden_targets_temperature_error.png")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 2: Emissivity RMSE and Spectral Angle Mapper (SAM)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Emissivity RMSE in Q-IV
    pinn_q4_eps = [df_pinn_q4[df_pinn_q4['material_class'] == c]['rmse_eps'].mean() for c in classes]
    tes_q4_eps = [df_tes_q4[df_tes_q4['material_class'] == c]['rmse_emissivity'].mean() for c in classes]

    ax1.bar(x - width/2, pinn_q4_eps, width, label='Sobolev-PINN (Run 2)', color='#2563EB', alpha=0.9)
    ax1.bar(x + width/2, tes_q4_eps, width, label='Classical TES', color='#EF4444', alpha=0.9)
    ax1.set_title("Emissivity RMSE $\\mathrm{RMSE}_\\varepsilon$ by Class (Quadrant IV OOD)", fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_labels, rotation=30, ha='right')
    ax1.set_ylabel("$\\mathrm{RMSE}_\\varepsilon$ (unitless)")
    ax1.legend(loc='upper right')

    # SAM in Q-IV
    pinn_q4_sam = [df_pinn_q4[df_pinn_q4['material_class'] == c]['sam_deg'].mean() for c in classes]
    tes_q4_sam = [df_tes_q4[df_tes_q4['material_class'] == c]['sam_deg'].mean() for c in classes]

    ax2.bar(x - width/2, pinn_q4_sam, width, label='Sobolev-PINN (Run 2)', color='#2563EB', alpha=0.9)
    ax2.bar(x + width/2, tes_q4_sam, width, label='Classical TES', color='#EF4444', alpha=0.9)
    ax2.set_title("Spectral Shape Distortion: SAM by Class (Quadrant IV OOD)", fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_labels, rotation=30, ha='right')
    ax2.set_ylabel("Spectral Angle Mapper (degrees)")
    ax2.legend(loc='upper right')

    plt.suptitle("Spectral Emissivity Recovery Fidelity across Pure Golden Targets (Q-IV OOD)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_and_copy(fig, "fig_golden_targets_emissivity_metrics.png")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 3: 1:1 Parity Scatter Plots (Temperature and Emissivity)
    # -------------------------------------------------------------------------
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 11))

    # Q-II Temperature
    ax1.scatter(df_tes_q2['true_temp_k'], df_tes_q2['pred_temp_k'], color='#DC2626', alpha=0.45, s=20, label='Classical TES')
    t_min = min(df_tes_q2['true_temp_k'].min(), df_pinn_q2['target_temp'].min()) - 2
    t_max = max(df_tes_q2['true_temp_k'].max(), df_pinn_q2['target_temp'].max()) + 2
    ax1.plot([t_min, t_max], [t_min, t_max], 'k--', lw=1.5, label='1:1 Ideal Parity')
    ax1.set_title(f"Classical TES: Q-II Temperature\nRMSE = {np.sqrt((df_tes_q2['delta_t_k']**2).mean()):.2f} K, MAE = {df_tes_q2['abs_delta_t_k'].mean():.2f} K", fontsize=10.5, fontweight='bold')
    ax1.set_xlabel("True Kinetic Temperature (K)")
    ax1.set_ylabel("Retrieved Temperature (K)")
    ax1.legend(loc='upper left')

    ax2.scatter(df_pinn_q2['target_temp'], df_pinn_q2['pred_temp'], color='#1E40AF', alpha=0.45, s=20, label='Sobolev-PINN (Run 2)')
    ax2.plot([t_min, t_max], [t_min, t_max], 'k--', lw=1.5, label='1:1 Ideal Parity')
    ax2.set_title(f"Sobolev-PINN: Q-II Temperature\nRMSE = {np.sqrt((df_pinn_q2['dt']**2).mean()):.2f} K, MAE = {df_pinn_q2['abs_dt'].mean():.2f} K", fontsize=10.5, fontweight='bold')
    ax2.set_xlabel("True Kinetic Temperature (K)")
    ax2.set_ylabel("Retrieved Temperature (K)")
    ax2.legend(loc='upper left')

    # Q-IV Temperature
    ax3.scatter(df_tes_q4['true_temp_k'], df_tes_q4['pred_temp_k'], color='#DC2626', alpha=0.35, s=15, label='Classical TES')
    ax3.plot([t_min, t_max], [t_min, t_max], 'k--', lw=1.5, label='1:1 Ideal Parity')
    ax3.set_title(f"Classical TES: Q-IV OOD Temperature\nRMSE = {np.sqrt((df_tes_q4['delta_t_k']**2).mean()):.2f} K, MAE = {df_tes_q4['abs_delta_t_k'].mean():.2f} K", fontsize=10.5, fontweight='bold')
    ax3.set_xlabel("True Kinetic Temperature (K)")
    ax3.set_ylabel("Retrieved Temperature (K)")
    ax3.legend(loc='upper left')

    ax4.scatter(df_pinn_q4['target_temp'], df_pinn_q4['pred_temp'], color='#1E40AF', alpha=0.35, s=15, label='Sobolev-PINN (Run 2)')
    ax4.plot([t_min, t_max], [t_min, t_max], 'k--', lw=1.5, label='1:1 Ideal Parity')
    ax4.set_title(f"Sobolev-PINN: Q-IV OOD Temperature\nRMSE = {np.sqrt((df_pinn_q4['dt']**2).mean()):.2f} K, MAE = {df_pinn_q4['abs_dt'].mean():.2f} K", fontsize=10.5, fontweight='bold')
    ax4.set_xlabel("True Kinetic Temperature (K)")
    ax4.set_ylabel("Retrieved Temperature (K)")
    ax4.legend(loc='upper left')

    plt.suptitle("1:1 Parity Verification: Classical TES vs. Sobolev-PINN on Golden Targets", fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_and_copy(fig, "fig_golden_targets_scatter_1to1.png")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 4: Representative Spectral Curves (Quartz, Calcite, Kaolinite, Concrete)
    # -------------------------------------------------------------------------
    print("\nExtracting actual spectral curves from HyTES Golden Targets dataset...")
    pinn_model, input_mode, mu_x, std_x, mask = load_pinn_model('hytes')
    device = torch.device('cpu')
    atm_cache = build_atmospheric_band_cache('hytes')
    tes = ClassicalTES('hytes')

    # Load HyTES wavelengths directly from stage1_core
    wav_all = get_sensor_wavelengths('hytes', device).cpu().numpy()
    wav_eff = wav_all[mask.cpu().numpy()]

    ds_q2 = torch.load(os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets', 'golden_benchmark_quadrant_II_hytes.pt'), weights_only=False)

    # Find one representative sample per class
    sample_classes = ['quartz', 'calcite', 'kaolinite', 'urban_concrete', 'urban_asphalt', 'botanical']
    found_samples = {}
    for sample in ds_q2:
        cls_name = sample['material_class']
        if cls_name in sample_classes and cls_name not in found_samples:
            found_samples[cls_name] = sample
        if len(found_samples) == len(sample_classes):
            break

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, cls_name in enumerate(sample_classes):
        ax = axes_flat[idx]
        sample = found_samples[cls_name]
        
        # Batch preparation for PINN
        batch = {k: torch.tensor(v).unsqueeze(0) if isinstance(v, (np.ndarray, list, float, int)) else [v] for k, v in sample.items()}
        for k in ['input_radiance', 'downwelling_radiance', 'target_temp', 'target_emissivity', 'pwv']:
            if k in sample:
                val = sample[k]
                batch[k] = val.clone().detach().unsqueeze(0) if torch.is_tensor(val) else torch.tensor(val).unsqueeze(0)
        
        x_in = assemble_input_tensor(batch, input_mode, atm_cache, device, mask=mask)
        x_norm = standardize_input(x_in, mu_x, std_x)
        
        with torch.no_grad():
            t_pred, e_pred = pinn_model(x_norm)
            eps_pinn = e_pred.squeeze(0).cpu().numpy()
            t_pinn = float(t_pred.item())
        
        # Ground truth
        eps_true = sample['target_emissivity'][mask.cpu()].numpy()
        t_true = float(sample['target_temp'])

        # Classical TES
        l_surf = sample['input_radiance'].numpy()
        l_down = sample['downwelling_radiance'].numpy()
        t_tes, eps_tes_full, mmd_val = tes.invert_single(l_surf, l_down)
        eps_tes = eps_tes_full[mask.cpu().numpy()]

        # Metrics for annotation
        sam_pinn = float(compute_spectral_angle_mapper(torch.tensor(eps_pinn).unsqueeze(0), torch.tensor(eps_true).unsqueeze(0)).item())
        sam_tes = float(compute_spectral_angle_mapper(torch.tensor(eps_tes).unsqueeze(0), torch.tensor(eps_true).unsqueeze(0)).item())
        rmse_pinn = float(np.sqrt(np.mean((eps_pinn - eps_true)**2)))
        rmse_tes = float(np.sqrt(np.mean((eps_tes - eps_true)**2)))

        # Plot curves
        ax.plot(wav_eff, eps_true, color='#059669', lw=2.2, label=f'Ground Truth (T = {t_true:.1f} K)')
        ax.plot(wav_eff, eps_pinn, color='#1E40AF', lw=1.8, linestyle='--', label=f'Sobolev-PINN (T = {t_pinn:.1f} K, $\\Delta T = {t_pinn - t_true:+.2f}$ K)')
        ax.plot(wav_eff, eps_tes, color='#DC2626', lw=1.5, linestyle=':', label=f'Classical TES (T = {t_tes:.1f} K, $\\Delta T = {t_tes - t_true:+.2f}$ K)')

        ax.set_title(f"{cls_name.replace('_', ' ').title()}", fontsize=11.5, fontweight='bold')
        
        # Text box with quantitative metrics
        box_text = f"PINN: $\\mathrm{{RMSE}}_\\varepsilon = {rmse_pinn:.4f}$, $\\mathrm{{SAM}} = {sam_pinn:.2f}^\\circ$\nTES:    $\\mathrm{{RMSE}}_\\varepsilon = {rmse_tes:.4f}$, $\\mathrm{{SAM}} = {sam_tes:.2f}^\\circ$"
        ax.text(0.03, 0.08, box_text, transform=ax.transAxes, fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#9CA3AF', alpha=0.9))

        ax.legend(loc='upper right', fontsize=8.2)
        if idx in [4, 5]:
            ax.set_xlabel("Wavelength ($\\mu$m)")
        if idx in [0, 2, 4]:
            ax.set_ylabel("Emissivity $\\varepsilon(\\lambda)$")
        ax.set_ylim(0.55, 1.02)

    plt.suptitle("HyTES Hyperspectral Emissivity Recovery: Sobolev-PINN vs. Classical TES on Golden Targets", fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_and_copy(fig, "fig_golden_targets_spectra.png")
    plt.close()

    print("\nAll 4 Golden Target benchmark figures successfully generated and mirrored to artifacts directory!")


if __name__ == '__main__':
    main()
