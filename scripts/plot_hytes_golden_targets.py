"""
Publication Figure Generator: HyTES Hyperspectral Golden Targets Benchmark
==========================================================================
Strictly analogous to the ASTER layout (fig_aster_golden_targets_spectra.png):
  - 6-Panel Diagnostic Suite (Quartz, Calcite, Kaolinite, Concrete, Asphalt, Botanical)
  - Ground Truth Laboratory Emissivity (Green solid line)
  - Sobolev-PINN Retrieved Emissivity (Blue dashed line, Phase C Run 2 Best Model)
  - Classical TES Retrieved Emissivity (Red dotted line, Hulley et al. 2012 operational formulation)
  - Clean metrics box: dT, RMSE_eps, SAM
  - Accompanying Error Breakdown Bar Chart by Mineral Class (dT & RMSE_eps)

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


def load_hytes_pinn_model():
    device = torch.device('cpu')
    model_path = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'cv', 'hytes', 'model_hytes_fold0_best.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'model_hytes_fold0_best.pth')
    
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    input_mode = ckpt.get('input_mode', config.get('input_mode', 'single'))
    norm_stats = ckpt.get('norm_stats', {})
    mask = ckpt.get('band_mask', get_sensor_band_mask('hytes', device=device)).to(device)
    
    n_bands_eff = int(mask.sum())
    model = PhysicsInformedNN(
        input_dim=n_bands_eff,
        output_dim=n_bands_eff,
        hidden_dim=int(config.get('hidden_dim', 768)),
        n_layers=int(config.get('n_layers', 3)),
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
    print("GENERATING HYTES 223-BAND GOLDEN TARGETS BENCHMARK FIGURES")
    print("=" * 80)

    device = torch.device('cpu')
    atm_cache = build_atmospheric_band_cache('hytes')
    tes = ClassicalTES('hytes')
    pinn_model, input_mode, mu_x, std_x, mask = load_hytes_pinn_model()

    wav_all = get_sensor_wavelengths('hytes', device).numpy()
    mask_np = mask.cpu().numpy()
    wav_eff = wav_all[mask_np]

    # Load HyTES Golden Targets dataset (Quadrant II)
    q2_hytes_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets', 'golden_benchmark_quadrant_II_hytes.pt')
    ds_hytes = torch.load(q2_hytes_path, weights_only=False)

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
    for sample in ds_hytes:
        c = sample['material_class']
        if c in sample_classes and c not in found_samples:
            found_samples[c] = sample
        if len(found_samples) == len(sample_classes):
            break

    # -------------------------------------------------------------------------
    # FIGURE 1: HyTES Hyperspectral Curves (Analogous to ASTER Layout)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    # The operational thermal window where both PINN and Classical TES operate physically
    # 8.0 to 11.5 um avoids deep water-vapor wings at 7.5 um and CO2 band at 12.0 um
    win_display = (wav_eff >= 8.10) & (wav_eff <= 11.55)
    wav_disp = wav_eff[win_display]

    for idx, cls_name in enumerate(sample_classes):
        ax = axes_flat[idx]
        sample = found_samples[cls_name]

        # Ground truth
        eps_true_full = sample['target_emissivity'][mask.cpu()].numpy()
        eps_true = eps_true_full[win_display]
        t_true = float(sample['target_temp'])

        # PINN Prediction
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
            eps_pinn_full = e_pred.squeeze(0).numpy()
            eps_pinn = eps_pinn_full[win_display]
            t_pinn = float(t_pred.item())

        # Classical TES (Operational Hulley et al. 2012)
        l_surf = sample['input_radiance'].numpy()
        l_down = sample['downwelling_radiance'].numpy()
        t_tes, eps_tes_raw, mmd_val = tes.invert_single(l_surf, l_down)
        eps_tes_eff = eps_tes_raw[mask_np]
        
        # Operational filter: apply a gentle 3-channel boxcar smoothing to Classical TES raw hyperspectral channel noise
        # (simulating the operational noise-reduction filter of Hulley et al. 2012)
        eps_tes_smoothed = np.convolve(eps_tes_eff, np.ones(3)/3.0, mode='same')
        eps_tes = eps_tes_smoothed[win_display]

        # Metrics calculated across the thermal atmospheric window
        dt_pinn = t_pinn - t_true
        dt_tes = t_tes - t_true
        rmse_pinn = float(np.sqrt(np.mean((eps_pinn - eps_true)**2)))
        rmse_tes = float(np.sqrt(np.mean((eps_tes - eps_true)**2)))
        sam_pinn = float(compute_spectral_angle_mapper(torch.tensor(eps_pinn).unsqueeze(0), torch.tensor(eps_true).unsqueeze(0)).item())
        sam_tes = float(compute_spectral_angle_mapper(torch.tensor(eps_tes).unsqueeze(0), torch.tensor(eps_true).unsqueeze(0)).item())

        # Plot curves analogous to ASTER
        ax.plot(wav_disp, eps_true, color='#059669', lw=2.2,
                label=f'HyTES True (T = {t_true:.1f} K)')
        ax.plot(wav_disp, eps_pinn, color='#1E40AF', lw=1.8, linestyle='--',
                label=f'Sobolev-PINN (T = {t_pinn:.1f} K, $\\Delta T = {dt_pinn:+.2f}$ K)')
        ax.plot(wav_disp, eps_tes, color='#DC2626', lw=1.5, linestyle=':',
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

    plt.suptitle("HyTES Hyperspectral Emissivity Retrieval: Sobolev-PINN vs. Classical TES on Golden Targets", fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_and_copy(fig, "fig_hytes_golden_targets_spectra_analogous.png")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 2: HyTES Quantitative Error Breakdown (Analogous to ASTER)
    # -------------------------------------------------------------------------
    tes_hytes_csv = os.path.join(PROJECT_ROOT, 'outputs', 'baselines', 'classical_tes_benchmark_hytes.csv')
    df_tes_hy = pd.read_csv(tes_hytes_csv)
    df_tes_q2 = df_tes_hy[df_tes_hy['dataset'] == 'golden_quadrant_II'].copy()

    pinn_q2_csv = os.path.join(PROJECT_ROOT, 'outputs', 'benchmarks', 'pinn_eval_hytes_Quadrant_II_GoldenTargets_DevAtm.csv')
    df_pinn_q2 = pd.read_csv(pinn_q2_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    classes = ['quartz', 'calcite', 'kaolinite', 'feldspar', 'urban_concrete', 'urban_asphalt', 'botanical']
    class_labels = ['Quartz', 'Calcite', 'Kaolinite', 'Feldspar', 'Concrete', 'Asphalt', 'Vegetation']
    x = np.arange(len(classes))
    width = 0.35

    # Subplot 1: Temperature MAE
    pinn_q2_dt = [df_pinn_q2[df_pinn_q2['material_class'] == c]['abs_dt'].mean() for c in classes]
    tes_q2_dt = [df_tes_q2[df_tes_q2['material_class'] == c]['abs_delta_t_k'].mean() for c in classes]

    ax1.bar(x - width/2, pinn_q2_dt, width, label='Sobolev-PINN (Run 2)', color='#1E40AF', alpha=0.9)
    ax1.bar(x + width/2, tes_q2_dt, width, label='Classical TES (Hulley et al. 2012)', color='#DC2626', alpha=0.9)
    ax1.set_title("HyTES Temperature Retrieval Error $|\\Delta T|$ (Quadrant II)", fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_labels, rotation=30, ha='right')
    ax1.set_ylabel("Mean Absolute Error $|\\Delta T|$ (K)")
    ax1.legend(loc='upper right')

    # Subplot 2: Emissivity RMSE
    pinn_q2_eps = [df_pinn_q2[df_pinn_q2['material_class'] == c]['rmse_eps'].mean() for c in classes]
    tes_q2_eps = [df_tes_q2[df_tes_q2['material_class'] == c]['rmse_emissivity'].mean() for c in classes]

    ax2.bar(x - width/2, pinn_q2_eps, width, label='Sobolev-PINN (Run 2)', color='#1E40AF', alpha=0.9)
    ax2.bar(x + width/2, tes_q2_eps, width, label='Classical TES (Hulley et al. 2012)', color='#DC2626', alpha=0.9)
    ax2.set_title("HyTES Emissivity RMSE $\\mathrm{RMSE}_\\varepsilon$ (Quadrant II)", fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_labels, rotation=30, ha='right')
    ax2.set_ylabel("$\\mathrm{RMSE}_\\varepsilon$ (unitless)")
    ax2.legend(loc='upper right')

    plt.suptitle("HyTES Hyperspectral Performance: Sobolev-PINN vs. Classical TES by Mineral Class", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_and_copy(fig, "fig_hytes_golden_targets_error_summary_analogous.png")
    plt.close()

    print("\nAll HyTES Golden Target analogous benchmark figures successfully generated and mirrored!")


if __name__ == '__main__':
    main()

