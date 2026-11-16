"""
Publication Figure Generator: ASTER 5-Band Golden Targets Benchmark
===================================================================
Compares:
  - High-Resolution Continuous Laboratory Truth (HyTES spectral resolution)
  - ASTER 5-Band Ground Truth Emissivity (convolved SRF)
  - Sobolev-PINN ASTER Retrieved Emissivity (Best Run 2 Model: Triple Ingestion, SiLU, Batch 8)
  - Canonical Classical TES Retrieved Emissivity (Gillespie et al. 1998)

Includes:
  1. fig_aster_golden_targets_spectra.png: 6-panel spectral comparison across diagnostic classes
  2. fig_aster_golden_targets_error_summary.png: Bar chart of dT, RMSE_eps, and SAM (PINN vs Classical TES)

Author: Dr. Atilio Grondona & OpenTIR-PINN Research Team
Target Journal: IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.pinn import PhysicsInformedNN
from src.baselines.classical_tes import ClassicalTES
from src.training.input_utils import build_atmospheric_band_cache, assemble_input_tensor, compute_spectral_angle_mapper
from src.training.stage1_core import (
    standardize_input,
    get_sensor_band_mask,
    get_sensor_wavelengths
)

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
plt.rcParams['legend.fontsize'] = 8.8
plt.rcParams['figure.titlesize'] = 13.5
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.35
plt.rcParams['grid.linestyle'] = '--'

ASTER_WAVELENGTHS = np.array([8.281, 8.633, 9.079, 10.662, 11.293])
ASTER_BAND_NAMES = ['B10\n8.28', 'B11\n8.63', 'B12\n9.08', 'B13\n10.66', 'B14\n11.29']


def load_aster_pinn_model():
    device = torch.device('cpu')
    model_path = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'cv', 'aster', 'model_aster_fold0_best.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'model_aster_fold0_best.pth')
    
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    input_mode = ckpt.get('input_mode', config.get('input_mode', 'triple'))
    norm_stats = ckpt.get('norm_stats', {})
    mask = ckpt.get('band_mask', get_sensor_band_mask('aster', device=device)).to(device)
    
    n_bands_eff = int(mask.sum())
    model = PhysicsInformedNN(
        input_dim=n_bands_eff,
        output_dim=n_bands_eff,
        hidden_dim=int(config.get('hidden_dim', 384)),
        n_layers=int(config.get('n_layers', 5)),
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
    print("=" * 80)
    print("GENERATING ASTER 5-BAND GOLDEN TARGETS BENCHMARK FIGURES")
    print("=" * 80)

    device = torch.device('cpu')
    atm_cache = build_atmospheric_band_cache('aster')
    tes = ClassicalTES('aster')
    pinn_model, input_mode, mu_x, std_x, mask = load_aster_pinn_model()

    # Load ASTER Golden Targets dataset (Quadrant II)
    q2_aster_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets', 'golden_benchmark_quadrant_II_aster.pt')
    q2_hytes_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets', 'golden_benchmark_quadrant_II_hytes.pt')
    ds_aster = torch.load(q2_aster_path, weights_only=False)
    ds_hytes = torch.load(q2_hytes_path, weights_only=False)

    wav_hytes = get_sensor_wavelengths('hytes', device).numpy()
    mask_hytes = get_sensor_band_mask('hytes', device).numpy()
    wav_hytes_clean = wav_hytes[mask_hytes]

    # Map spectrum_id to continuous HyTES spectrum for high-res reference
    hytes_by_spec_id = {s['spectrum_id']: s for s in ds_hytes}

    sample_classes = ['quartz', 'calcite', 'kaolinite', 'urban_concrete', 'urban_asphalt', 'botanical']
    class_titles = {
        'quartz': 'Quartz (SiO2) - Reststrahlen Doublet',
        'calcite': 'Calcite (CaCO3) - Carbonate Bending',
        'kaolinite': 'Kaolinite - Clay Hydroxyl Feature',
        'urban_concrete': 'Urban Concrete - Low Contrast Flat Spectrum',
        'urban_asphalt': 'Urban Asphalt - Low Contrast Hydrocarbon Matrix',
        'botanical': 'Vegetation / Botanical - High Emissivity Plateau'
    }

    found_samples = {}
    for sample in ds_aster:
        c = sample['material_class']
        if c in sample_classes and c not in found_samples:
            found_samples[c] = sample
        if len(found_samples) == len(sample_classes):
            break

    # -------------------------------------------------------------------------
    # FIGURE 1: ASTER 5-Band Spectral Curves (PINN vs Classical TES vs Truth)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, cls_name in enumerate(sample_classes):
        ax = axes_flat[idx]
        sample = found_samples[cls_name]
        spec_id = sample['spectrum_id']

        # Continuous high-resolution reference from HyTES
        if spec_id in hytes_by_spec_id:
            hytes_samp = hytes_by_spec_id[spec_id]
            eps_continuous = hytes_samp['target_emissivity'][mask_hytes].numpy()
            ax.plot(wav_hytes_clean, eps_continuous, color='#9CA3AF', lw=1.2, linestyle='-', alpha=0.75,
                    label='Continuous Lab Spectrum (Reference)')

        # Ground Truth ASTER 5 bands
        eps_true_aster = sample['target_emissivity'].numpy()
        t_true = float(sample['target_temp'])

        # Batch preparation for PINN
        batch = {}
        for k, v in sample.items():
            if torch.is_tensor(v):
                batch[k] = v.unsqueeze(0).to(device)
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(v).unsqueeze(0).to(device)
            elif isinstance(v, (int, float)):
                batch[k] = torch.tensor([v]).to(device)
            else:
                batch[k] = [v]
        
        x_in = assemble_input_tensor(batch, input_mode, atm_cache, device, mask=mask)
        x_norm = standardize_input(x_in, mu_x, std_x)
        
        with torch.no_grad():
            t_pred, e_pred = pinn_model(x_norm)
            eps_pinn = e_pred.squeeze(0).numpy()
            t_pinn = float(t_pred.item())

        # Classical TES Prediction
        l_surf = sample['input_radiance'].numpy()
        l_down = sample['downwelling_radiance'].numpy()
        t_tes, eps_tes, mmd_val = tes.invert_single(l_surf, l_down)

        # Compute metrics
        dt_pinn = t_pinn - t_true
        dt_tes = t_tes - t_true
        rmse_pinn = float(np.sqrt(np.mean((eps_pinn - eps_true_aster)**2)))
        rmse_tes = float(np.sqrt(np.mean((eps_tes - eps_true_aster)**2)))
        sam_pinn = float(compute_spectral_angle_mapper(torch.tensor(eps_pinn).unsqueeze(0), torch.tensor(eps_true_aster).unsqueeze(0)).item())
        sam_tes = float(compute_spectral_angle_mapper(torch.tensor(eps_tes).unsqueeze(0), torch.tensor(eps_true_aster).unsqueeze(0)).item())

        # Plot ASTER 5-Band curves with markers
        ax.plot(ASTER_WAVELENGTHS, eps_true_aster, 's-', color='#059669', lw=2.2, ms=6.5,
                label=f'ASTER True (T = {t_true:.1f} K)')
        ax.plot(ASTER_WAVELENGTHS, eps_pinn, 'o--', color='#1E40AF', lw=1.8, ms=6.0,
                label=f'Sobolev-PINN (T = {t_pinn:.1f} K, $\\Delta T = {dt_pinn:+.2f}$ K)')
        ax.plot(ASTER_WAVELENGTHS, eps_tes, '^:', color='#DC2626', lw=1.6, ms=6.5,
                label=f'Classical TES (T = {t_tes:.1f} K, $\\Delta T = {dt_tes:+.2f}$ K)')

        ax.set_title(class_titles.get(cls_name, cls_name.title()), fontsize=11, fontweight='bold')

        # Quantitative text box
        box_text = (f"Sobolev-PINN: $\\mathrm{{RMSE}}_\\varepsilon = {rmse_pinn:.4f}$ | $\\mathrm{{SAM}} = {sam_pinn:.2f}^\\circ$\n"
                    f"Classical TES: $\\mathrm{{RMSE}}_\\varepsilon = {rmse_tes:.4f}$ | $\\mathrm{{SAM}} = {sam_tes:.2f}^\\circ$")
        ax.text(0.03, 0.08, box_text, transform=ax.transAxes, fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#D1D5DB', alpha=0.92))

        ax.legend(loc='upper right', fontsize=8.0)
        ax.set_ylim(0.55, 1.03)
        if idx in [4, 5]:
            ax.set_xlabel("Wavelength ($\\mu$m)")
        if idx in [0, 2, 4]:
            ax.set_ylabel("Emissivity $\\varepsilon(\\lambda)$")

    plt.suptitle("ASTER 5-Band Emissivity Retrieval: Sobolev-PINN vs. Canonical Classical TES on Golden Targets", fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_and_copy(fig, "fig_aster_golden_targets_spectra.png")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 2: Quantitative Error Breakdown (ASTER Q-II vs Q-IV)
    # -------------------------------------------------------------------------
    tes_aster_csv = os.path.join(PROJECT_ROOT, 'outputs', 'baselines', 'classical_tes_benchmark_aster.csv')
    df_tes_ast = pd.read_csv(tes_aster_csv)
    df_tes_q2 = df_tes_ast[df_tes_ast['dataset'] == 'golden_quadrant_II'].copy()
    df_tes_q4 = df_tes_ast[df_tes_ast['dataset'] == 'golden_quadrant_IV_ood'].copy()

    pinn_q2_csv = os.path.join(PROJECT_ROOT, 'outputs', 'benchmarks', 'pinn_eval_aster_Quadrant_II_GoldenTargets_DevAtm.csv')
    pinn_q4_csv = os.path.join(PROJECT_ROOT, 'outputs', 'benchmarks', 'pinn_eval_aster_Quadrant_IV_GoldenTargets_WinterAtm_OOD.csv')
    df_pinn_q2 = pd.read_csv(pinn_q2_csv)
    df_pinn_q4 = pd.read_csv(pinn_q4_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    classes = ['quartz', 'calcite', 'kaolinite', 'feldspar', 'urban_concrete', 'urban_asphalt', 'botanical']
    class_labels = ['Quartz', 'Calcite', 'Kaolinite', 'Feldspar', 'Concrete', 'Asphalt', 'Vegetation']
    x = np.arange(len(classes))
    width = 0.35

    # Subplot 1: Temperature MAE
    pinn_q2_dt = [df_pinn_q2[df_pinn_q2['material_class'] == c]['abs_dt'].mean() for c in classes]
    tes_q2_dt = [df_tes_q2[df_tes_q2['material_class'] == c]['abs_delta_t_k'].mean() for c in classes]

    ax1.bar(x - width/2, pinn_q2_dt, width, label='Sobolev-PINN (Run 2)', color='#1E40AF', alpha=0.9)
    ax1.bar(x + width/2, tes_q2_dt, width, label='Classical TES (Gillespie et al. 1998)', color='#DC2626', alpha=0.9)
    ax1.set_title("ASTER Temperature Retrieval Error $|\\Delta T|$ (Quadrant II)", fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_labels, rotation=30, ha='right')
    ax1.set_ylabel("Mean Absolute Error $|\\Delta T|$ (K)")
    ax1.legend(loc='upper right')

    # Subplot 2: Emissivity RMSE
    pinn_q2_eps = [df_pinn_q2[df_pinn_q2['material_class'] == c]['rmse_eps'].mean() for c in classes]
    tes_q2_eps = [df_tes_q2[df_tes_q2['material_class'] == c]['rmse_emissivity'].mean() for c in classes]

    ax2.bar(x - width/2, pinn_q2_eps, width, label='Sobolev-PINN (Run 2)', color='#1E40AF', alpha=0.9)
    ax2.bar(x + width/2, tes_q2_eps, width, label='Classical TES (Gillespie et al. 1998)', color='#DC2626', alpha=0.9)
    ax2.set_title("ASTER Emissivity RMSE $\\mathrm{RMSE}_\\varepsilon$ (Quadrant II)", fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_labels, rotation=30, ha='right')
    ax2.set_ylabel("$\\mathrm{RMSE}_\\varepsilon$ (unitless)")
    ax2.legend(loc='upper right')

    plt.suptitle("ASTER Multispectral Performance: Sobolev-PINN vs. Classical TES by Mineral Class", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_and_copy(fig, "fig_aster_golden_targets_error_summary.png")
    plt.close()

    print("\nAll ASTER Golden Target benchmark figures successfully generated and mirrored!")


if __name__ == '__main__':
    main()
