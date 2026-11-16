"""
Generate publication-quality figures for OpenTIR-PINN Zenodo Benchmark v1.1.0.

Figures generated:
1. fig1_golden_targets_emissivity.png: The 7 Quarantined Golden Targets (HyTES 256-band vs ASTER 5-band).
2. fig2_modtran_atmospheric_profiles.png: 6 Standard MODTRAN 5.2 Geographic Models (tau, L_up, L_down).
3. fig3_toa_radiance_and_planck_coupling.png: Physical Planck manifold and complete RTE boundary coupling.
4. fig4_sensor_spectral_response_functions.png: HyTES (256 channels) and ASTER (5 bands) SRFs.
5. fig5_dataset_split_and_augmentation.png: Double-Quarantine topology and data augmentation pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = r"C:\Users\Atilio\Desktop\PROJETOS\AITIR"
RELEASE_DIR = os.path.join(ROOT_DIR, "zenodo_release")
FIGURES_DIR = os.path.join(RELEASE_DIR, "docs", "figures")
DATA_DIR = os.path.join(RELEASE_DIR, "data")
ATM_DIR = os.path.join(DATA_DIR, "atmospheric_luts")
SRF_DIR = os.path.join(DATA_DIR, "srf")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Publication styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13.5

print("=" * 80)
print("GENERATING ZENODO v1.1.0 FIGURES")
print("=" * 80)

# Load NPZ datasets
hytes_npz = np.load(os.path.join(DATA_DIR, "hytes_256band_sobolev_pinn_dataset.npz"))
aster_npz = np.load(os.path.join(DATA_DIR, "aster_5band_sobolev_pinn_dataset.npz"))

hytes_wav = hytes_npz['wavelengths']
aster_wav = aster_npz['wavelengths']

# -------------------------------------------------------------------------
# FIGURE 1: Golden Targets Laboratory Spectral Emissivity (HyTES vs ASTER)
# -------------------------------------------------------------------------
print("\n[1/5] Generating Figure 1: Golden Targets Spectral Emissivity...")

colors = [
    "#4B5563",  # concrete (slate gray)
    "#1F2937",  # asphalt (dark charcoal)
    "#059669",  # calcite (emerald green)
    "#2563EB",  # kaolinite (royal blue)
    "#DC2626",  # quartz (crimson red)
    "#9333EA",  # feldspar (purple)
    "#854D0E"   # non-photosynthetic vegetation, white-fir bark (warm bark brown)
]

labels_clean = [
    "Urban Concrete (Cement Matrix)",
    "Urban Asphalt (Bituminous Binder)",
    "Calcite (CaCO₃, Carbonate Doublet)",
    "Kaolinite (Al₂Si₂O₅(OH)₄, Clay Minimum)",
    "Quartz (α-SiO₂, Silica Reststrahlen)",
    "Feldspar (Albite Plagioclase)",
    "NPV: White-Fir Bark (Abies concolor)"
]

# Extract unique spectra from Q2
q2_emis_h = []
q2_emis_a = []
golden_keys_order = ['urban_concrete', 'urban_asphalt', 'calcite', 'kaolinite', 'quartz', 'feldspar', 'npv_bark']

for gk in golden_keys_order:
    for idx, name in enumerate(hytes_npz['val_q2_golden_name']):
        if name == gk:
            q2_emis_h.append(hytes_npz['val_q2_emissivity'][idx])
            break
    for idx, name in enumerate(aster_npz['val_q2_golden_name']):
        if name == gk:
            q2_emis_a.append(aster_npz['val_q2_emissivity'][idx])
            break

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.0), sharey=True, dpi=300)

for i in range(len(labels_clean)):
    ax1.plot(hytes_wav, q2_emis_h[i], label=labels_clean[i], color=colors[i], lw=2.0)
    ax2.plot(aster_wav, q2_emis_a[i], 'o--', label=labels_clean[i], color=colors[i], lw=1.8, ms=7)

ax1.set_title("(a) HyTES Hyperspectral Sampling (256 Contiguous Channels)", fontweight='bold')
ax1.set_xlabel(r"Wavelength ($\mu$m)")
ax1.set_ylabel(r"Spectral Emissivity $\epsilon(\lambda)$")
ax1.set_xlim(7.45, 12.05)
ax1.set_ylim(0.15, 1.02)
ax1.grid(True, linestyle='--', alpha=0.35)

ax1.legend(loc='lower left', framealpha=0.92, edgecolor='#D1D5DB', fancybox=True, fontsize=8.5)

# Diagnostic annotations
ax1.annotate("Quartz Reststrahlen\nDoublet (8.3 & 9.3 $\mu$m)", 
             xy=(8.48, 0.891), xytext=(7.60, 0.65),
             arrowprops=dict(facecolor='#DC2626', edgecolor='#DC2626', arrowstyle='->', lw=1.5), 
             fontsize=8.5, fontweight='bold', color='#DC2626',
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#DC2626", alpha=0.9, lw=0.8))

ax1.annotate(r"Calcite $\nu_2$ Bending" + "\n" + r"Doublet (11.3 $\mu$m)", 
             xy=(11.32, 0.932), xytext=(9.9, 0.82),
             arrowprops=dict(facecolor='#059669', edgecolor='#059669', arrowstyle='->', lw=1.5), 
             fontsize=8.5, fontweight='bold', color='#059669',
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#059669", alpha=0.9, lw=0.8))

ax1.annotate(r"Feldspar Asymmetric" + "\n" + r"Depression (10.0 $\mu$m)", 
             xy=(10.05, 0.38), xytext=(8.8, 0.25),
             arrowprops=dict(facecolor='#9333EA', edgecolor='#9333EA', arrowstyle='->', lw=1.5), 
             fontsize=8.5, fontweight='bold', color='#9333EA',
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#9333EA", alpha=0.9, lw=0.8))

ax2.set_title("(b) ASTER Multispectral Sampling (5 Discrete TIR Bands)", fontweight='bold')
ax2.set_xlabel(r"Wavelength ($\mu$m)")
ax2.set_xlim(8.0, 11.6)
ax2.grid(True, linestyle='--', alpha=0.35)

for b_idx, (w, b_name) in enumerate(zip(aster_wav, ["B10", "B11", "B12", "B13", "B14"])):
    ax2.text(w, 0.18, b_name, ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#4B5563')

plt.suptitle("OpenTIR-PINN Benchmark: 7 Quarantined Golden Target Emissivity Signatures (v1.1.0)", y=0.98, fontsize=14, fontweight='bold')
plt.tight_layout()
fig1_path = os.path.join(FIGURES_DIR, "fig1_golden_targets_emissivity.png")
fig.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  -> Saved {fig1_path}")

# -------------------------------------------------------------------------
# FIGURE 2: MODTRAN 5.2 Atmospheric Profiles Across 6 Geographic Models
# -------------------------------------------------------------------------
print("\n[2/5] Generating Figure 2: MODTRAN 5.2 Atmospheric Profiles...")

# Pick standard 1.0x water vapor profile for the 6 standard atmospheric models
atm_models = [
    ("profile_atm1_h2o1.00.csv", "1. Tropical (PWV = 4.11 g/cm²)", "#DC2626"),
    ("profile_atm2_h2o1.00.csv", "2. Mid-Lat Summer (PWV = 2.92 g/cm²)", "#D97706"),
    ("profile_atm3_h2o1.00.csv", "3. Mid-Lat Winter (PWV = 0.85 g/cm², Q-IV OOD)", "#2563EB"),
    ("profile_atm4_h2o1.00.csv", "4. Sub-Arctic Summer (PWV = 2.08 g/cm²)", "#059669"),
    ("profile_atm5_h2o1.00.csv", "5. Sub-Arctic Winter (PWV = 0.42 g/cm²)", "#7C3AED"),
    ("profile_atm6_h2o1.00.csv", "6. 1976 US Standard (PWV = 1.42 g/cm²)", "#4B5563")
]

fig, (ax_tau, ax_up, ax_down) = plt.subplots(1, 3, figsize=(17, 5.0), dpi=300)

for fname, plabel, pcolor in atm_models:
    fpath = os.path.join(ATM_DIR, fname)
    df = pd.read_csv(fpath)
    df_window = df[(df['wavelength_um'] >= 7.5) & (df['wavelength_um'] <= 12.5)]
    
    lw = 2.2 if "Winter" in plabel else 1.5
    ls = '--' if "Q-IV OOD" in plabel else '-'
    ax_tau.plot(df_window['wavelength_um'], df_window['transmittance'], label=plabel, color=pcolor, lw=lw, linestyle=ls)
    ax_up.plot(df_window['wavelength_um'], df_window['path_radiance'], label=plabel, color=pcolor, lw=lw, linestyle=ls)
    ax_down.plot(df_window['wavelength_um'], df_window['downwelling_radiance'], label=plabel, color=pcolor, lw=lw, linestyle=ls)

ax_tau.set_title("(a) Atmospheric Transmittance $\\tau(\\lambda)$", fontweight='bold')
ax_tau.set_xlabel(r"Wavelength ($\mu$m)")
ax_tau.set_ylabel("Total Path Transmittance [0, 1]")
ax_tau.set_ylim(0.0, 1.02)
ax_tau.set_xlim(7.5, 12.5)
ax_tau.grid(True, linestyle='--', alpha=0.35)
ax_tau.legend(loc='lower right', fontsize=8.0, framealpha=0.92)

# Highlight ozone band
ax_tau.axvspan(9.4, 9.8, color='gray', alpha=0.15)
ax_tau.text(9.6, 0.08, "Ozone (O₃)\nAbsorption", ha='center', fontsize=8, color='#4B5563')

ax_up.set_title("(b) Upwelling Path Radiance $L^\\uparrow(\\lambda)$", fontweight='bold')
ax_up.set_xlabel(r"Wavelength ($\mu$m)")
ax_up.set_ylabel(r"Path Radiance ($\mathrm{W}/(\mathrm{m}^2\cdot\mathrm{sr}\cdot\mu\mathrm{m})$)")
ax_up.set_xlim(7.5, 12.5)
ax_up.grid(True, linestyle='--', alpha=0.35)

ax_down.set_title(r"(c) Downwelling Sky Irradiance $L^\downarrow(\lambda) / \pi$", fontweight='bold')
ax_down.set_xlabel(r"Wavelength ($\mu$m)")
ax_down.set_ylabel(r"Sky Irradiance / $\pi$ ($\mathrm{W}/(\mathrm{m}^2\cdot\mathrm{sr}\cdot\mu\mathrm{m})$)")
ax_down.set_xlim(7.5, 12.5)
ax_down.grid(True, linestyle='--', alpha=0.35)

plt.suptitle("MODTRAN 5.2 Radiative Transfer Library: 6 Standard Atmospheric Regimes (v1.1.0)", y=0.99, fontsize=14, fontweight='bold')
plt.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, "fig2_modtran_atmospheric_profiles.png")
fig.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  -> Saved {fig2_path}")

# -------------------------------------------------------------------------
# FIGURE 3: Radiative Transfer Equation Forward Coupling & Planck Spectrum
# -------------------------------------------------------------------------
print("\n[3/5] Generating Figure 3: RTE Radiance Coupling & Planck Manifold...")

sys.path.insert(0, RELEASE_DIR)
from physics_engine import planck_blackbody_radiance, forward_rte_solve

fig, (ax_planck, ax_toa) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

wav_dense = np.linspace(7.5, 12.5, 500)
temps = [285.0, 300.0, 315.0, 330.0]
temp_colors = ["#2563EB", "#059669", "#D97706", "#DC2626"]

# Subplot 1: Planck Blackbody Spectrum
for t, c in zip(temps, temp_colors):
    b = planck_blackbody_radiance(wav_dense, t)
    ax_planck.plot(wav_dense, b, label=f"$T_s = {t:.0f}\ \mathrm{{K}}$", color=c, lw=2.2)

ax_planck.set_title(r"(a) Continuous Planck Blackbody Radiance $B(\lambda, T_s)$", fontweight='bold')
ax_planck.set_xlabel(r"Wavelength ($\mu$m)")
ax_planck.set_ylabel(r"Spectral Radiance ($\mathrm{W}/(\mathrm{m}^2\cdot\mathrm{sr}\cdot\mu\mathrm{m})$)")
ax_planck.set_xlim(7.5, 12.5)
ax_planck.grid(True, linestyle='--', alpha=0.35)
ax_planck.legend(loc='upper right', framealpha=0.9)

# Subplot 2: Forward RTE Coupling for Quartz
df_atm = pd.read_csv(os.path.join(ATM_DIR, "profile_atm2_h2o1.00.csv"))
tau_interp = np.interp(wav_dense, df_atm['wavelength_um'], df_atm['transmittance'])
l_up_interp = np.interp(wav_dense, df_atm['wavelength_um'], df_atm['path_radiance'])
l_down_interp = np.interp(wav_dense, df_atm['wavelength_um'], df_atm['downwelling_radiance'])
quartz_emis_interp = np.interp(wav_dense, hytes_wav, q2_emis_h[4])  # quartz

for t, c in zip(temps, temp_colors):
    b_surf = planck_blackbody_radiance(wav_dense, t)
    l_surf = (quartz_emis_interp * b_surf) + ((1.0 - quartz_emis_interp) * l_down_interp)
    l_toa = (tau_interp * l_surf) + l_up_interp
    ax_toa.plot(wav_dense, l_surf, label=f"$L_{{\mathrm{{surf}}}}$ at {t:.0f} K", color=c, lw=2.0)
    ax_toa.plot(wav_dense, l_toa, linestyle=':', color=c, lw=1.6, alpha=0.85)

ax_toa.plot([], [], 'k-', label=r"$L_{\mathrm{surf}} = \epsilon B + (1-\epsilon) L^\downarrow / \pi$ (Solid)", lw=1.8)
ax_toa.plot([], [], 'k:', label=r"$L_{\mathrm{TOA}} = \tau L_{\mathrm{surf}} + L^\uparrow$ (Dotted)", lw=1.8)

ax_toa.set_title(r"(b) Full RTE Surface & TOA Coupling (Quartz Target)", fontweight='bold')
ax_toa.set_xlabel(r"Wavelength ($\mu$m)")
ax_toa.set_ylabel(r"Coupled Radiance ($\mathrm{W}/(\mathrm{m}^2\cdot\mathrm{sr}\cdot\mu\mathrm{m})$)")
ax_toa.set_xlim(7.5, 12.5)
ax_toa.grid(True, linestyle='--', alpha=0.35)
ax_toa.legend(loc='upper right', framealpha=0.9, fontsize=8.5)

plt.suptitle("Thermodynamic Radiance Coupling & Dual-Input Radiative Transfer Physics", y=0.98, fontsize=14, fontweight='bold')
plt.tight_layout()
fig3_path = os.path.join(FIGURES_DIR, "fig3_toa_radiance_and_planck_coupling.png")
fig.savefig(fig3_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  -> Saved {fig3_path}")

# -------------------------------------------------------------------------
# FIGURE 4: Sensor Spectral Response Functions (HyTES vs ASTER)
# -------------------------------------------------------------------------
print("\n[4/5] Generating Figure 4: Sensor Spectral Response Functions...")

fig, (ax_hytes, ax_aster) = plt.subplots(2, 1, figsize=(15, 7.8), sharex=True, dpi=300)

# HyTES SRFs (sample every 6th channel)
hytes_srf_csv = pd.read_csv(os.path.join(SRF_DIR, "hytes", "hytes_channel_parameters.csv"))
wav_srf_grid = np.linspace(7.2, 12.5, 1200)

cmap = plt.get_cmap('plasma')
for idx in range(0, 256, 5):
    row = hytes_srf_csv.iloc[idx]
    c_um = row['center_wavelength']
    fwhm_um = row['fwhm']
    sigma = fwhm_um / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gauss = np.exp(-0.5 * ((wav_srf_grid - c_um) / sigma) ** 2)
    color_val = cmap(idx / 256.0)
    ax_hytes.plot(wav_srf_grid, gauss, color=color_val, lw=1.2, alpha=0.85)

ax_hytes.set_title("HyTES: 256 Contiguous Imaging Channels ($7.49 - 12.00\ \mu\mathrm{m}$, FWHM $\\approx 25-42\ \mathrm{nm}$)", fontweight='bold')
ax_hytes.set_ylabel("Normalized Response")
ax_hytes.set_ylim(0, 1.05)
ax_hytes.grid(True, linestyle='--', alpha=0.3)

# ASTER SRFs (Bands 10 to 14)
aster_colors = ["#2563EB", "#059669", "#D97706", "#DC2626", "#7C3AED"]
aster_specs = pd.read_csv(os.path.join(SRF_DIR, "aster", "aster_5bands_specifications.csv"))

for b_idx, (b_num, b_col) in enumerate(zip(range(10, 15), aster_colors)):
    srf_file = os.path.join(SRF_DIR, "aster", aster_specs.iloc[b_idx]['srf_curve_file'])
    if os.path.exists(srf_file):
        df_srf = pd.read_csv(srf_file)
        ax_aster.plot(df_srf['wavelength_microns'], df_srf['relative_spectral_response'],
                      label=f"Band {b_num} ({aster_wav[b_idx]:.2f} $\mu$m, FWHM={aster_specs.iloc[b_idx]['fwhm_microns']:.2f} $\mu$m)",
                      color=b_col, lw=2.5)
        ax_aster.fill_between(df_srf['wavelength_microns'], 0, df_srf['relative_spectral_response'], color=b_col, alpha=0.15)

ax_aster.set_title("ASTER: 5 Discrete Multispectral TIR Bands (Bands 10–14, FWHM $\\approx 0.34 - 0.70\ \mu\mathrm{m}$)", fontweight='bold')
ax_aster.set_xlabel(r"Wavelength ($\mu$m)")
ax_aster.set_ylabel("Normalized Response")
ax_aster.set_xlim(7.2, 12.5)
ax_aster.set_ylim(0, 1.05)
ax_aster.grid(True, linestyle='--', alpha=0.3)
ax_aster.legend(loc='upper right', framealpha=0.9, fontsize=9)

plt.suptitle("Sensor Spectral Response Functions (SRF) Comparison: Hyperspectral vs. Multispectral", y=0.98, fontsize=14, fontweight='bold')
plt.tight_layout()
fig4_path = os.path.join(FIGURES_DIR, "fig4_sensor_spectral_response_functions.png")
fig.savefig(fig4_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  -> Saved {fig4_path}")

# -------------------------------------------------------------------------
# FIGURE 5: Dataset Split, Double-Quarantine Matrix & Data Augmentation Pipeline
# -------------------------------------------------------------------------
print("\n[5/5] Generating Figure 5: Dataset Split & Augmentation Architecture...")

fig, (ax_matrix, ax_pipe) = plt.subplots(1, 2, figsize=(16, 6.2), dpi=300)

# Subplot 1: Double-Quarantine Matrix
ax_matrix.set_xlim(0, 10)
ax_matrix.set_ylim(0, 10)
ax_matrix.axis('off')

# Quadrant Boxes
# Q1: Dev Materials x Dev Atm (Training Pool)
rect_q1 = patches.FancyBboxPatch((0.5, 5.2), 4.2, 4.0, boxstyle="round,pad=0.2", facecolor="#EFF6FF", edgecolor="#3B82F6", lw=2)
ax_matrix.add_patch(rect_q1)
ax_matrix.text(2.6, 8.5, "QUADRANT I: Training Pool\n(Development Regime)", ha='center', va='center', fontweight='bold', color="#1E40AF", fontsize=11)
ax_matrix.text(2.6, 6.8, "• N = 5,000 synthetic pairs\n• 245 curated laboratory endmembers\n• Atm 1, 2, 4, 5, 6 (Tropical, Summer, Sub-Arctic)\n• 5-Fold Cross-Validation Splitting", ha='center', va='center', color="#1E3A8A", fontsize=9.5)

# Q2: Golden Targets x Dev Atm (Golden Benchmark In-Dist)
rect_q2 = patches.FancyBboxPatch((5.3, 5.2), 4.2, 4.0, boxstyle="round,pad=0.2", facecolor="#ECFDF5", edgecolor="#10B981", lw=2)
ax_matrix.add_patch(rect_q2)
ax_matrix.text(7.4, 8.5, "QUADRANT II: Golden Suite\n(In-Distribution Atmospheres)", ha='center', va='center', fontweight='bold', color="#065F46", fontsize=11)
ax_matrix.text(7.4, 6.8, "• N = 308 evaluation pairs\n• 7 Quarantined Golden Targets (OOD Materials)\n• Development Atmospheres (1, 2, 4, 5, 6)\n• Evaluates Pure Material Generalization", ha='center', va='center', color="#064E3B", fontsize=9.5)

# Q3: Dev Materials x Arctic Winter
rect_q3 = patches.FancyBboxPatch((0.5, 0.6), 4.2, 4.0, boxstyle="round,pad=0.2", facecolor="#F3F4F6", edgecolor="#9CA3AF", lw=1.5, linestyle="--")
ax_matrix.add_patch(rect_q3)
ax_matrix.text(2.6, 3.8, "QUADRANT III: Atmospheric OOD\n(Training Materials in Winter)", ha='center', va='center', fontweight='bold', color="#4B5563", fontsize=11)
ax_matrix.text(2.6, 2.2, "• Dev endmembers in Arctic/Winter\n• Atm 3 (Mid-Lat Winter, PWV < 1.0 g/cm²)\n• Diagnostic step for transferability", ha='center', va='center', color="#374151", fontsize=9.5)

# Q4: Golden Targets x Arctic Winter (Strict Double-Quarantine)
rect_q4 = patches.FancyBboxPatch((5.3, 0.6), 4.2, 4.0, boxstyle="round,pad=0.2", facecolor="#FEF2F2", edgecolor="#EF4444", lw=2.5)
ax_matrix.add_patch(rect_q4)
ax_matrix.text(7.4, 3.8, "QUADRANT IV: Double Quarantine\n(OOD Materials & OOD Winter Atm)", ha='center', va='center', fontweight='bold', color="#991B1B", fontsize=11)
ax_matrix.text(7.4, 2.2, "• N = 770 evaluation pairs (110 per target)\n• 7 Quarantined Targets + Atm 3 Winter\n• Zero-Shot Dual Extrapolation Stress Test\n• Primary Benchmark vs. NASA Classical TES", ha='center', va='center', color="#7F1D1D", fontsize=9.5)

ax_matrix.set_title("(a) Double-Quarantine Matrix Benchmark Topology", fontweight='bold', fontsize=12)

# Subplot 2: Data Augmentation & Synthesis Pipeline
ax_pipe.set_xlim(0, 10)
ax_pipe.set_ylim(0, 10)
ax_pipe.axis('off')

steps = [
    ("Step 1: Endmember Selection & Quarantine", "245 training spectra selected from USGS/ECOSTRESS;\n7 Golden Targets + 6 sibling spectra quarantined 100%.", "#3B82F6"),
    ("Step 2: Thermodynamic Temperature Sweep", "Surface temperature T_s ~ U(280.0 K, 330.0 K) with\nboundary constraint |T_s - T_air| <= 15.0 K.", "#10B981"),
    ("Step 3: Atmospheric Coupling & Water Vapor Perturbation", "6 MODTRAN 5.2 models x 5 water vapor scaling factors\n(0.50x to 2.00x), sweeping PWV from 0.21 to 8.23 g/cm².", "#F59E0B"),
    ("Step 4: Dual-Input RTE Radiative Transfer Integration", "Surface radiance L_surf = e*B(T_s) + (1-e)*L_down/pi;\nTop-of-Atmosphere radiance L_TOA = tau*L_surf + L_up.", "#8B5CF6"),
    ("Step 5: Stratified 5-Fold Cross-Validation Splitting", "Training pool partitioned into 5 balanced folds (split_v1.1.json)\npreserving material superclass proportions.", "#EC4899")
]

y_pos = 8.8
for title, desc, color in steps:
    rect = patches.FancyBboxPatch((0.5, y_pos - 1.3), 9.0, 1.4, boxstyle="round,pad=0.15", facecolor="white", edgecolor=color, lw=1.8)
    ax_pipe.add_patch(rect)
    ax_pipe.text(0.8, y_pos - 0.2, title, fontweight='bold', color=color, fontsize=10.5)
    ax_pipe.text(0.8, y_pos - 0.75, desc, color="#374151", fontsize=9.0)
    
    if y_pos > 2.5:
        ax_pipe.annotate('', xy=(5.0, y_pos - 1.4), xytext=(5.0, y_pos - 1.7),
                        arrowprops=dict(facecolor=color, edgecolor=color, arrowstyle="->", lw=1.5))
    y_pos -= 1.8

ax_pipe.set_title("(b) Physics-Grounded Data Augmentation & Splitting Flow", fontweight='bold', fontsize=12)

plt.suptitle("OpenTIR-PINN Benchmark (v1.1.0): Architectural Topology, Splitting & Augmentation", y=0.98, fontsize=14, fontweight='bold')
plt.tight_layout()
fig5_path = os.path.join(FIGURES_DIR, "fig5_dataset_split_and_augmentation.png")
fig.savefig(fig5_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  -> Saved {fig5_path}")

print("=" * 80)
print("ALL 5 ZENODO v1.1.0 FIGURES GENERATED SUCCESSFULLY!")
print("=" * 80)

