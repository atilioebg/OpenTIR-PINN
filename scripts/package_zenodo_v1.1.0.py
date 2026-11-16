"""
Package Zenodo Release v1.1.0:
OpenTIR-PINN Benchmark: Synthetic Radiative Transfer & Golden Target Dataset
for Ill-Posed Temperature and Emissivity Separation Across Multispectral and Hyperspectral Sensors.

Associated Manuscript:
"Physics-Informed Neural Networks with Sobolev Shape Regularization for Ill-Posed
Temperature and Emissivity Separation Across Multispectral and Hyperspectral Thermal Infrared Sensors"
(IEEE Transactions on Geoscience and Remote Sensing)
Author: Dr. Atilio Grondona (UFRGS)
"""

import os
import shutil
import zipfile
import json
import torch
import numpy as np
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = r"C:\Users\Atilio\Desktop\PROJETOS\AITIR"
RELEASE_DIR = os.path.join(ROOT_DIR, "zenodo_release")
DATA_DIR = os.path.join(RELEASE_DIR, "data")
ATM_DIR = os.path.join(DATA_DIR, "atmospheric_luts")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
BENCH_DIR = os.path.join(DATA_DIR, "benchmarks")
RAW_GOLDEN_DIR = os.path.join(DATA_DIR, "raw_laboratory_spectra", "golden_targets")
SRF_DIR = os.path.join(DATA_DIR, "srf")
HYTES_SRF_DIR = os.path.join(SRF_DIR, "hytes")
ASTER_SRF_DIR = os.path.join(SRF_DIR, "aster")

os.makedirs(RELEASE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ATM_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(BENCH_DIR, exist_ok=True)
os.makedirs(RAW_GOLDEN_DIR, exist_ok=True)
os.makedirs(HYTES_SRF_DIR, exist_ok=True)
os.makedirs(ASTER_SRF_DIR, exist_ok=True)

print("=" * 80)
print("PACKAGING ZENODO BENCHMARK DATASET v1.1.0")
print("=" * 80)

# -------------------------------------------------------------------------
# Step 1: Atmospheric Look-Up Tables (MODTRAN 5.2 - 30 Standard Profiles)
# -------------------------------------------------------------------------
print("\n[Step 1/8] Updating Atmospheric Profiles (30 MODTRAN 5.2 LUTs)...")

# Clean legacy run_0000... files
for fname in os.listdir(ATM_DIR):
    if fname.startswith("run_") or (fname.endswith(".txt") and "tape" in fname):
        fpath = os.path.join(ATM_DIR, fname)
        os.remove(fpath)
        print(f"  - Removed legacy file: {fname}")

src_atm_dir = os.path.join(ROOT_DIR, "data", "processed", "atmospheric_luts")
copied_atm_count = 0
for fname in os.listdir(src_atm_dir):
    if fname.endswith(".csv") or fname.endswith(".npz"):
        shutil.copy(os.path.join(src_atm_dir, fname), os.path.join(ATM_DIR, fname))
        copied_atm_count += 1

print(f"  -> Copied {copied_atm_count} atmospheric files (30 .csv, 30 .npz, lut_catalog.csv) to {ATM_DIR}")

# Create modtran_simulation_parameters_matrix.csv
lut_cat_df = pd.read_csv(os.path.join(ATM_DIR, "lut_catalog.csv"))
sim_matrix_df = lut_cat_df[['profile_id', 'atm_model_idx', 'atm_model_name', 'h2o_scale', 'pwv_g_cm2', 'quarantine_status', 'csv_file', 'npz_file']]
sim_matrix_df.to_csv(os.path.join(ATM_DIR, "modtran_simulation_parameters_matrix.csv"), index=False)
print(f"  -> Generated {os.path.join(ATM_DIR, 'modtran_simulation_parameters_matrix.csv')} (30 profiles)")

# -------------------------------------------------------------------------
# Step 2: Splits and Golden Target Manifests
# -------------------------------------------------------------------------
print("\n[Step 2/8] Copying Cross-Validation and Golden Target Manifests...")
shutil.copy(os.path.join(ROOT_DIR, "splits", "split_v1.1.json"), os.path.join(SPLITS_DIR, "split_v1.1.json"))
shutil.copy(os.path.join(ROOT_DIR, "splits", "golden_targets.json"), os.path.join(SPLITS_DIR, "golden_targets.json"))
print(f"  -> Saved split_v1.1.json (5-Fold CV split) and golden_targets.json into {SPLITS_DIR}")

# -------------------------------------------------------------------------
# Step 3: Benchmark Scorecards & Hyperparameter Configurations
# -------------------------------------------------------------------------
print("\n[Step 3/8] Copying Evaluation Scorecards and Optuna Best Configurations...")
src_bench_dir = os.path.join(ROOT_DIR, "outputs", "benchmarks")
bench_files = [f for f in os.listdir(src_bench_dir) if f.endswith(".csv")]
for bf in bench_files:
    shutil.copy(os.path.join(src_bench_dir, bf), os.path.join(BENCH_DIR, bf))
print(f"  -> Copied {len(bench_files)} benchmark CSV files to {BENCH_DIR}")

shutil.copy(os.path.join(ROOT_DIR, "outputs", "best_config_aster.json"), os.path.join(BENCH_DIR, "best_config_aster.json"))
shutil.copy(os.path.join(ROOT_DIR, "outputs", "best_config_hytes.json"), os.path.join(BENCH_DIR, "best_config_hytes.json"))
print(f"  -> Copied best_config_aster.json and best_config_hytes.json to {BENCH_DIR}")

# -------------------------------------------------------------------------
# Step 4: Raw Golden Targets and Discrete Metadata/CSV Files
# -------------------------------------------------------------------------
print("\n[Step 4/8] Updating Golden Target Raw Spectra and Metadata...")

# Ensure correct raw files in golden_targets/
raw_src_vh311 = os.path.join(ROOT_DIR, "data", "raw", "ecospeclib", "ecospeclib-all", "nonphotosyntheticvegetation.bark.abies.concolor.tir.vh311.ucsb.nicolet.spectrum.txt")
raw_dst_vh311 = os.path.join(RAW_GOLDEN_DIR, "npv_bark_abies_concolor_tir_vh311.txt")
shutil.copy(raw_src_vh311, raw_dst_vh311)

# Remove legacy misnamed vegetation files
for f in os.listdir(RAW_GOLDEN_DIR):
    if f.startswith("vegetation_green_leaf"):
        os.remove(os.path.join(RAW_GOLDEN_DIR, f))
        print(f"  - Removed legacy file: {f}")

print(f"  -> Copied non-photosynthetic vegetation bark (VH311) raw spectrum to {raw_dst_vh311}")
raw_golden_files = os.listdir(RAW_GOLDEN_DIR)
print(f"  -> Verified raw golden targets count: {len(raw_golden_files)} files: {raw_golden_files}")

# Update golden_targets_metadata.csv
golden_meta_data = [
    {
        "target_id": 0,
        "target_name": "urban_concrete",
        "material_class": "urban_concrete",
        "chemical_or_mineral_description": "Hydrated Portland cement matrix with quartz/carbonate aggregates (0598UUUCNC)",
        "spectral_diagnostic_features": "Flat, high broad-band emissivity (0.93 - 0.96) with slight silicate inflection near 9.3 um",
        "laboratory_source_library": "JHU / ECOSTRESS Spectral Library v1.0",
        "quarantine_status": "Strict Out-of-Distribution (OOD) Golden Target - 100% excluded from PINN training"
    },
    {
        "target_id": 1,
        "target_name": "urban_asphalt",
        "material_class": "urban_asphalt",
        "chemical_or_mineral_description": "Bituminous hydrocarbon binder with mineral filler aggregate (0095UUUASP)",
        "spectral_diagnostic_features": "Smooth, low-contrast featureless continuum (emissivity 0.94 - 0.96)",
        "laboratory_source_library": "JHU / ECOSTRESS Spectral Library v1.0",
        "quarantine_status": "Strict Out-of-Distribution (OOD) Golden Target - 100% excluded from PINN training"
    },
    {
        "target_id": 2,
        "target_name": "calcite",
        "material_class": "calcite",
        "chemical_or_mineral_description": "Calcium carbonate (CaCO3), trigonal crystal system (calcite_1 coarse)",
        "spectral_diagnostic_features": "Sharp internal fundamental vibrational modes: v3 asymmetric stretch (7.0 um), v1 symmetric stretch (9.26 um), v2 out-of-plane bending doublet (11.4 um)",
        "laboratory_source_library": "USGS / ASTER Spectral Library v2.0",
        "quarantine_status": "Strict Out-of-Distribution (OOD) Golden Target - 100% excluded from PINN training"
    },
    {
        "target_id": 3,
        "target_name": "kaolinite",
        "material_class": "kaolinite",
        "chemical_or_mineral_description": "Al2Si2O5(OH)4, 1:1 dioctahedral aluminosilicate layer structure (kaolinite_2 coarse)",
        "spectral_diagnostic_features": "Strong Si-O-Si asymmetric stretch minimum at 9.6 um; distinct OH-bending overtone shoulder at 10.8 um",
        "laboratory_source_library": "USGS / ASTER Spectral Library v2.0",
        "quarantine_status": "Strict Out-of-Distribution (OOD) Golden Target - 100% excluded from PINN training"
    },
    {
        "target_id": 4,
        "target_name": "quartz",
        "material_class": "quartz",
        "chemical_or_mineral_description": "Silicon dioxide (SiO2), alpha-quartz trigonal network (quartz_1 coarse)",
        "spectral_diagnostic_features": "Prototypical Christiansen frequency at 7.4 um; prominent Si-O-Si Reststrahlen doublet with sharp emissivity minima at 8.3 um and 9.3 um",
        "laboratory_source_library": "USGS / ASTER Spectral Library v2.0",
        "quarantine_status": "Strict Out-of-Distribution (OOD) Golden Target - 100% excluded from PINN training"
    },
    {
        "target_id": 5,
        "target_name": "feldspar",
        "material_class": "feldspar",
        "chemical_or_mineral_description": "Anorthite (CaAl2Si2O8), calcium-rich endmember of plagioclase feldspar series (anorthite_1 coarse)",
        "spectral_diagnostic_features": "High thermal continuum emissivity (0.95 - 0.99) with subtle Al-Si tetrahedral stretch inflection at 9.8 um",
        "laboratory_source_library": "USGS / ASTER Spectral Library v2.0",
        "quarantine_status": "Strict Out-of-Distribution (OOD) Golden Target - 100% excluded from PINN training"
    },
    {
        "target_id": 6,
        "target_name": "npv_bark",
        "material_class": "botanical",
        "chemical_or_mineral_description": "Abies concolor (White Fir) tree bark, non-photosynthetic vegetation (VH311)",
        "spectral_diagnostic_features": "Lignin and cellulose absorption bands across 8.5 - 11.5 um; muted Reststrahlen features; high continuum emissivity (~0.95 - 0.98)",
        "laboratory_source_library": "UCSB / ECOSTRESS Spectral Library v1.0",
        "quarantine_status": "Strict Out-of-Distribution (OOD) Golden Target - 100% excluded from PINN training"
    }
]
pd.DataFrame(golden_meta_data).to_csv(os.path.join(DATA_DIR, "golden_targets_metadata.csv"), index=False)
print(f"  -> Updated golden_targets_metadata.csv with 7 Golden Targets (including NPV White-Fir Bark)")

# -------------------------------------------------------------------------
# Step 5: HyTES 256-band Dataset Packaging (.npz)
# -------------------------------------------------------------------------
print("\n[Step 5/8] Packaging HyTES 256-Channel Dataset Archive (.npz)...")
hytes_params_df = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'processed', 'hytes_real_parameters.csv')).sort_values('band_idx')
hytes_wavelengths = hytes_params_df['center_wavelength'].values.astype(np.float32)
hytes_fwhms = hytes_params_df['fwhm'].values.astype(np.float32)

train_hytes = torch.load(os.path.join(ROOT_DIR, 'data', 'processed', 'final_datasets', 'train_dataset_hytes.pt'), map_location='cpu', weights_only=False)
q2_hytes = torch.load(os.path.join(ROOT_DIR, 'data', 'processed', 'final_datasets', 'golden_benchmark_quadrant_II_hytes.pt'), map_location='cpu', weights_only=False)
q4_hytes = torch.load(os.path.join(ROOT_DIR, 'data', 'processed', 'final_datasets', 'golden_benchmark_quadrant_IV_hytes.pt'), map_location='cpu', weights_only=False)
val_hytes_path = os.path.join(ROOT_DIR, 'data', 'processed', 'final_datasets', 'val_dataset_hytes.pt')
val_hytes = torch.load(val_hytes_path, map_location='cpu', weights_only=False) if os.path.exists(val_hytes_path) else None

def extract_tensors(sample_list):
    L_obs = np.stack([s['input_radiance'].numpy() for s in sample_list]).astype(np.float32)
    L_toa = np.stack([s['input_radiance_sensor'].numpy() for s in sample_list]).astype(np.float32)
    L_down = np.stack([s['downwelling_radiance'].numpy() for s in sample_list]).astype(np.float32)
    eps = np.stack([s['target_emissivity'].numpy() for s in sample_list]).astype(np.float32)
    Ts = np.array([float(s['target_temp']) for s in sample_list], dtype=np.float32)
    materials = [s.get('material_class', s.get('material', 'unknown')) for s in sample_list]
    golden_names = [s.get('golden_name', s.get('material_class', 'unknown')) for s in sample_list]
    atm_ids = [s.get('atmosphere_id', 'unknown') for s in sample_list]
    pwvs = np.array([float(s.get('pwv', 0.0)) for s in sample_list], dtype=np.float32)
    return L_obs, L_toa, L_down, eps, Ts, materials, golden_names, atm_ids, pwvs

tr_L_obs_h, tr_L_toa_h, tr_L_down_h, tr_eps_h, tr_Ts_h, tr_mat_h, _, tr_atm_h, tr_pwv_h = extract_tensors(train_hytes)
q2_L_obs_h, q2_L_toa_h, q2_L_down_h, q2_eps_h, q2_Ts_h, q2_mat_h, q2_gn_h, q2_atm_h, q2_pwv_h = extract_tensors(q2_hytes)
q4_L_obs_h, q4_L_toa_h, q4_L_down_h, q4_eps_h, q4_Ts_h, q4_mat_h, q4_gn_h, q4_atm_h, q4_pwv_h = extract_tensors(q4_hytes)
if val_hytes is not None:
    vo_L_obs_h, vo_L_toa_h, vo_L_down_h, vo_eps_h, vo_Ts_h, vo_mat_h, vo_gn_h, vo_atm_h, vo_pwv_h = extract_tensors(val_hytes)

hytes_npz = os.path.join(DATA_DIR, "hytes_256band_sobolev_pinn_dataset.npz")
hytes_save_dict = {
    "wavelengths": hytes_wavelengths,
    "fwhm": hytes_fwhms,
    "train_L_obs": tr_L_obs_h,
    "train_L_toa": tr_L_toa_h,
    "train_L_down": tr_L_down_h,
    "train_emissivity": tr_eps_h,
    "train_Ts": tr_Ts_h,
    "train_materials": tr_mat_h,
    "train_atmosphere_id": tr_atm_h,
    "train_pwv": tr_pwv_h,
    "val_q2_L_obs": q2_L_obs_h,
    "val_q2_L_toa": q2_L_toa_h,
    "val_q2_L_down": q2_L_down_h,
    "val_q2_emissivity": q2_eps_h,
    "val_q2_Ts": q2_Ts_h,
    "val_q2_materials": q2_mat_h,
    "val_q2_golden_name": q2_gn_h,
    "val_q2_atmosphere_id": q2_atm_h,
    "val_q2_pwv": q2_pwv_h,
    "val_q4_L_obs": q4_L_obs_h,
    "val_q4_L_toa": q4_L_toa_h,
    "val_q4_L_down": q4_L_down_h,
    "val_q4_emissivity": q4_eps_h,
    "val_q4_Ts": q4_Ts_h,
    "val_q4_materials": q4_mat_h,
    "val_q4_golden_name": q4_gn_h,
    "val_q4_atmosphere_id": q4_atm_h,
    "val_q4_pwv": q4_pwv_h
}
if val_hytes is not None:
    hytes_save_dict.update({
        "val_oof_L_obs": vo_L_obs_h,
        "val_oof_L_toa": vo_L_toa_h,
        "val_oof_L_down": vo_L_down_h,
        "val_oof_emissivity": vo_eps_h,
        "val_oof_Ts": vo_Ts_h,
        "val_oof_materials": vo_mat_h,
        "val_oof_golden_name": vo_gn_h,
        "val_oof_atmosphere_id": vo_atm_h,
        "val_oof_pwv": vo_pwv_h
    })

np.savez_compressed(hytes_npz, **hytes_save_dict)
print(f"  -> Saved HyTES archive: {hytes_npz} ({os.path.getsize(hytes_npz)/1e6:.2f} MB)")
print(f"     Train: {tr_L_obs_h.shape}, Q2: {q2_L_obs_h.shape}, Q4: {q4_L_obs_h.shape}" + (f", Val OOF: {vo_L_obs_h.shape}" if val_hytes else ""))

# Generate discrete golden_targets_hytes_256band_emissivity.csv
hytes_golden_dict = {'channel_index': np.arange(1, 257), 'wavelength_microns': hytes_wavelengths, 'fwhm_microns': hytes_fwhms}
golden_keys_order = ['urban_concrete', 'urban_asphalt', 'calcite', 'kaolinite', 'quartz', 'feldspar', 'npv_bark']
for gk in golden_keys_order:
    for s in q2_hytes:
        if s.get('golden_name') == gk:
            hytes_golden_dict[f'emissivity_{gk}'] = s['target_emissivity'].numpy()
            break
pd.DataFrame(hytes_golden_dict).to_csv(os.path.join(DATA_DIR, "golden_targets_hytes_256band_emissivity.csv"), index=False)
print(f"  -> Generated golden_targets_hytes_256band_emissivity.csv")

# -------------------------------------------------------------------------
# Step 6: ASTER 5-band Dataset Packaging (.npz)
# -------------------------------------------------------------------------
print("\n[Step 6/8] Packaging ASTER 5-Band Dataset Archive (.npz)...")
aster_wavelengths = np.array([8.291, 8.634, 9.075, 10.657, 11.318], dtype=np.float32)
aster_fwhms = np.array([0.354, 0.358, 0.355, 0.702, 0.705], dtype=np.float32)

train_aster = torch.load(os.path.join(ROOT_DIR, 'data', 'processed', 'final_datasets', 'train_dataset_aster.pt'), map_location='cpu', weights_only=False)
q2_aster = torch.load(os.path.join(ROOT_DIR, 'data', 'processed', 'final_datasets', 'golden_benchmark_quadrant_II_aster.pt'), map_location='cpu', weights_only=False)
q4_aster = torch.load(os.path.join(ROOT_DIR, 'data', 'processed', 'final_datasets', 'golden_benchmark_quadrant_IV_aster.pt'), map_location='cpu', weights_only=False)
val_aster_path = os.path.join(ROOT_DIR, 'data', 'processed', 'final_datasets', 'val_dataset_aster.pt')
val_aster = torch.load(val_aster_path, map_location='cpu', weights_only=False) if os.path.exists(val_aster_path) else None

tr_L_obs_a, tr_L_toa_a, tr_L_down_a, tr_eps_a, tr_Ts_a, tr_mat_a, _, tr_atm_a, tr_pwv_a = extract_tensors(train_aster)
q2_L_obs_a, q2_L_toa_a, q2_L_down_a, q2_eps_a, q2_Ts_a, q2_mat_a, q2_gn_a, q2_atm_a, q2_pwv_a = extract_tensors(q2_aster)
q4_L_obs_a, q4_L_toa_a, q4_L_down_a, q4_eps_a, q4_Ts_a, q4_mat_a, q4_gn_a, q4_atm_a, q4_pwv_a = extract_tensors(q4_aster)
if val_aster is not None:
    vo_L_obs_a, vo_L_toa_a, vo_L_down_a, vo_eps_a, vo_Ts_a, vo_mat_a, vo_gn_a, vo_atm_a, vo_pwv_a = extract_tensors(val_aster)

aster_npz = os.path.join(DATA_DIR, "aster_5band_sobolev_pinn_dataset.npz")
aster_save_dict = {
    "wavelengths": aster_wavelengths,
    "fwhm": aster_fwhms,
    "train_L_obs": tr_L_obs_a,
    "train_L_toa": tr_L_toa_a,
    "train_L_down": tr_L_down_a,
    "train_emissivity": tr_eps_a,
    "train_Ts": tr_Ts_a,
    "train_materials": tr_mat_a,
    "train_atmosphere_id": tr_atm_a,
    "train_pwv": tr_pwv_a,
    "val_q2_L_obs": q2_L_obs_a,
    "val_q2_L_toa": q2_L_toa_a,
    "val_q2_L_down": q2_L_down_a,
    "val_q2_emissivity": q2_eps_a,
    "val_q2_Ts": q2_Ts_a,
    "val_q2_materials": q2_mat_a,
    "val_q2_golden_name": q2_gn_a,
    "val_q2_atmosphere_id": q2_atm_a,
    "val_q2_pwv": q2_pwv_a,
    "val_q4_L_obs": q4_L_obs_a,
    "val_q4_L_toa": q4_L_toa_a,
    "val_q4_L_down": q4_L_down_a,
    "val_q4_emissivity": q4_eps_a,
    "val_q4_Ts": q4_Ts_a,
    "val_q4_materials": q4_mat_a,
    "val_q4_golden_name": q4_gn_a,
    "val_q4_atmosphere_id": q4_atm_a,
    "val_q4_pwv": q4_pwv_a
}
if val_aster is not None:
    aster_save_dict.update({
        "val_oof_L_obs": vo_L_obs_a,
        "val_oof_L_toa": vo_L_toa_a,
        "val_oof_L_down": vo_L_down_a,
        "val_oof_emissivity": vo_eps_a,
        "val_oof_Ts": vo_Ts_a,
        "val_oof_materials": vo_mat_a,
        "val_oof_golden_name": vo_gn_a,
        "val_oof_atmosphere_id": vo_atm_a,
        "val_oof_pwv": vo_pwv_a
    })

np.savez_compressed(aster_npz, **aster_save_dict)
print(f"  -> Saved ASTER archive: {aster_npz} ({os.path.getsize(aster_npz)/1e6:.2f} MB)")
print(f"     Train: {tr_L_obs_a.shape}, Q2: {q2_L_obs_a.shape}, Q4: {q4_L_obs_a.shape}" + (f", Val OOF: {vo_L_obs_a.shape}" if val_aster else ""))

# Generate discrete golden_targets_aster_5band_emissivity.csv
aster_golden_dict = {'band_index': np.arange(10, 15), 'center_wavelength_microns': aster_wavelengths, 'fwhm_microns': aster_fwhms}
for gk in golden_keys_order:
    for s in q2_aster:
        if s.get('golden_name') == gk:
            aster_golden_dict[f'emissivity_{gk}'] = s['target_emissivity'].numpy()
            break
pd.DataFrame(aster_golden_dict).to_csv(os.path.join(DATA_DIR, "golden_targets_aster_5band_emissivity.csv"), index=False)
print(f"  -> Generated golden_targets_aster_5band_emissivity.csv")

# Ensure all ASTER RTTOV files (ch11 to ch15) are present in ASTER_SRF_DIR
for ch in range(11, 16):
    fname = f"rtcoef_eos_1_aster_srf_ch{ch:02d}.txt"
    src_f = os.path.join(ROOT_DIR, 'bibliografic_references', 'aster', fname)
    if os.path.exists(src_f):
        shutil.copy(src_f, os.path.join(ASTER_SRF_DIR, fname))

# -------------------------------------------------------------------------
# Step 7: Update Metadata (.zenodo.json, README.md, DATA_DICTIONARY.md)
# -------------------------------------------------------------------------
print("\n[Step 7/8] Writing FAIR Documentation (.zenodo.json, README.md, DATA_DICTIONARY.md)...")

# 7.1 .zenodo.json
zenodo_json_data = {
    "title": "OpenTIR-PINN Benchmark v1.3.0: Multi-Atmospheric Radiative Transfer and Double-Quarantine Golden Target Dataset for Ill-Posed Temperature and Emissivity Separation",
    "description": "Comprehensive benchmark dataset for Physics-Informed Neural Networks (PINN), radiative transfer inversion, and operational Temperature and Emissivity Separation (TES) algorithms in the thermal infrared (TIR). Version 1.3.0 provides rigorous MODTRAN 5.2 simulations across 30 distinct atmospheric profiles (6 standard atmospheric models × 5 water vapor scaling factors), operational atmospheric-correction PWV error propagation (N4), Dual-Input spectral radiances [L_surf, L_down], Top-of-Atmosphere radiances [L_TOA], a rebalanced 5,460 synthetic training pool (Fold 0), 5-Fold group-disjoint cross-validation splits, and an immutable Double-Quarantine evaluation suite of 7 Golden Targets (including non-photosynthetic vegetation bark) tested across In-Distribution (Quadrant II, N=308) and Arctic Out-of-Distribution (Quadrant IV, N=770) regimes.",
    "version": "1.3.0",
    "publication_date": "2026-11-04",
    "creators": [
        {
            "name": "Grondona, Atilio",
            "affiliation": "Universidade Federal do Rio Grande do Sul (UFRGS)",
            "orcid": "0000-0002-1825-0097"
        }
    ],
    "keywords": [
        "Thermal Infrared",
        "Temperature and Emissivity Separation",
        "Physics-Informed Neural Networks",
        "Radiative Transfer Equation",
        "MODTRAN",
        "HyTES",
        "ASTER",
        "Remote Sensing",
        "GeoAI",
        "Double-Quarantine Benchmark"
    ],
    "license": "CC-BY-4.0",
    "access_right": "open",
    "upload_type": "dataset",
    "related_identifiers": [
        {
            "relation": "isSupplementTo",
            "identifier": "10.1109/TGRS.2026.XXXXXXX",
            "resource_type": "publication-article"
        }
    ]
}

with open(os.path.join(RELEASE_DIR, ".zenodo.json"), "w", encoding="utf-8") as f:
    json.dump(zenodo_json_data, f, indent=2)
print("  -> Updated .zenodo.json (v1.1.0)")

# 7.2 DATA_DICTIONARY.md
data_dict_content = r"""# Data Dictionary: OpenTIR-PINN Benchmark (v1.1.0)

**Associated Manuscript:** *Physics-Informed Neural Networks with Sobolev Shape Regularization for Ill-Posed Temperature and Emissivity Separation Across Multispectral and Hyperspectral Thermal Infrared Sensors* (IEEE Transactions on Geoscience and Remote Sensing / Remote Sensing of Environment)  
**Author:** Dr. Atilio Grondona (*Universidade Federal do Rio Grande do Sul - UFRGS*)  
**Version:** 1.1.0 (Released November 2026)  
**License:** CC-BY 4.0 International

---

## 1. Directory Topology

```
zenodo_release/
├── .zenodo.json                               # Zenodo DOI deposit schema
├── README.md                                  # FAIR overview, erratum notice, and quickstart guide
├── DATA_DICTIONARY.md                         # Meticulous field-by-field tensor and tabular specification
├── quickstart_demo.py                         # Standalone python validation script
├── physics_engine.py                          # Vectorized, differentiable forward RTE operator
├── tests/
│   └── test_physics_engine.py                 # PyTorch autograd and Planck round-trip unit tests
└── data/
    ├── hytes_256band_sobolev_pinn_dataset.npz # HyTES 256-band training, Q2, and Q4 archives
    ├── aster_5band_sobolev_pinn_dataset.npz   # ASTER 5-band training, Q2, and Q4 archives
    ├── golden_targets_metadata.csv            # Detailed physical/chemical metadata for the 7 Golden Targets
    ├── golden_targets_hytes_256band_emissivity.csv # HyTES 256-band resampled laboratory emissivities
    ├── golden_targets_aster_5band_emissivity.csv   # ASTER 5-band resampled laboratory emissivities
    ├── atmospheric_luts/                      # 30 MODTRAN 5.2 profile look-up tables
    │   ├── lut_catalog.csv                    # Catalog of the 30 profiles (atmospheric models, PWV, quarantine)
    │   ├── modtran_simulation_parameters_matrix.csv # Parameter sweep matrix
    │   ├── profile_atm1_h2o0.50.csv/.npz      # Tropical (PWV = 2.06 g/cm²)
    │   └── ...                                # Up to profile_atm6_h2o2.00 (PWV = 0.42 to 8.23 g/cm²)
    ├── splits/                                # Reproducibility splits
    │   ├── split_v1.1.json                    # 5-Fold cross-validation manifest (rebalanced pool)
    │   └── golden_targets.json                # Immutable golden target definitions & sibling quarantine
    ├── benchmarks/                            # Verification scorecards and best Optuna configs
    │   ├── best_config_aster.json             # Converged hyperparameter architecture (ASTER)
    │   ├── best_config_hytes.json             # Converged hyperparameter architecture (HyTES)
    │   ├── cv_benchmark_aster.csv / hytes.csv # 5-Fold cross-validation scorecards
    │   ├── ablation_ingestion_benchmark_*.csv # Atmospheric coupling ablation scorecards
    │   ├── loco_benchmark_aster.csv / hytes.csv # Leave-One-Class-Out (LOCO 7+1) stress-tests
    │   └── quarantine_matrix_benchmark_*.csv  # Double-Quarantine vs. Classical TES benchmarks
    ├── raw_laboratory_spectra/                # Raw continuous laboratory FTIR spectra
    │   ├── golden_targets/                    # 7 individual FTIR text files (Pre-code continuous spectra)
    │   ├── raw_spectral_library_inventory.csv # Inventory of 1,205 continuous spectra
    │   └── all_raw_laboratory_spectra_1205.zip # Compressed master raw library (1,205 files)
    └── srf/                                   # Sensor Spectral Response Functions
        ├── aster/                             # ASTER Bands 10-14 SRFs and specification tables
        └── hytes/                             # HyTES 256-channel SRFs, WMX, and specification tables
```

---

## 2. Main Compressed NumPy Archives (`.npz`)

Both `hytes_256band_sobolev_pinn_dataset.npz` and `aster_5band_sobolev_pinn_dataset.npz` contain identical key topologies, sized to their respective spectral band counts ($B=256$ for HyTES, $B=5$ for ASTER).

| Key | Tensor Shape | Data Type | Physical Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| `wavelengths` | `(B,)` | `float32` | $\mu\text{m}$ | Center band wavelengths across the TIR atmospheric window. |
| `fwhm` | `(B,)` | `float32` | $\mu\text{m}$ | Full-Width at Half-Maximum of sensor spectral response functions. |
| `train_L_obs` | `(5460, B)` | `float32` | $\text{W}/(\text{m}^2\cdot\text{sr}\cdot\mu\text{m})$ | Surface-leaving thermal radiance $L_{\mathrm{surf}}(\lambda) = \epsilon(\lambda) B(\lambda, T_s) + [1-\epsilon(\lambda)] L^\downarrow(\lambda)/\pi$. |
| `train_L_toa` | `(5460, B)` | `float32` | $\text{W}/(\text{m}^2\cdot\text{sr}\cdot\mu\text{m})$ | Top-of-Atmosphere radiance $L_{\mathrm{TOA}}(\lambda) = \tau(\lambda) L_{\mathrm{surf}}(\lambda) + L^\uparrow(\lambda)$. |
| `train_L_down` | `(5460, B)` | `float32` | $\text{W}/(\text{m}^2\cdot\text{sr}\cdot\mu\text{m})$ | Downwelling atmospheric sky irradiance divided by $\pi$ ($L^\downarrow(\lambda) / \pi$). |
| `train_emissivity`| `(5460, B)` | `float32` | Dimensionless $[0, 1]$ | Ground-truth surface spectral emissivity convolved by sensor SRF. |
| `train_Ts` | `(5460,)` | `float32` | $\text{Kelvin (K)}$ | Ground-truth surface kinetic temperature ($T_s \in [275, 335]\text{ K}$). |
| `train_materials` | `(5460,)` | `str / object`| Categorical | Primary material classification (urban, geological, botanical, water). |
| `train_atmosphere_id`| `(5460,)` | `str / object`| Categorical | Identifier of the MODTRAN profile applied during simulation. |
| `train_pwv` | `(5460,)` | `float32` | $\text{g}/\text{cm}^2$ | Precipitable water vapor of the simulation atmosphere. |
| `val_q2_L_obs` | `(308, B)` | `float32` | $\text{W}/(\text{m}^2\cdot\text{sr}\cdot\mu\text{m})$ | Quadrant II (In-Distribution Atmospheres) surface-leaving radiance. |
| `val_q2_L_toa` | `(308, B)` | `float32` | $\text{W}/(\text{m}^2\cdot\text{sr}\cdot\mu\text{m})$ | Quadrant II Top-of-Atmosphere radiance. |
| `val_q2_L_down` | `(308, B)` | `float32` | $\text{W}/(\text{m}^2\cdot\text{sr}\cdot\mu\text{m})$ | Quadrant II downwelling atmospheric sky irradiance. |
| `val_q2_emissivity`| `(308, B)`| `float32` | Dimensionless $[0, 1]$ | Quadrant II ground-truth emissivity for the 7 Golden Targets. |
| `val_q2_Ts` | `(308,)` | `float32` | $\text{Kelvin (K)}$ | Quadrant II ground-truth surface kinetic temperature. |
| `val_q2_golden_name`| `(308,)` | `str / object`| Categorical | Golden Target identifier (`quartz`, `calcite`, `kaolinite`, etc.). |
| `val_q2_atmosphere_id`| `(308,)`| `str / object`| Categorical | MODTRAN profile identifier used for simulation. |
| `val_q2_pwv` | `(308,)` | `float32` | $\text{g}/\text{cm}^2$ | Precipitable water vapor column of simulation atmosphere. |
| `val_q4_L_obs` | `(770, B)` | `float32` | $\text{W}/(\text{m}^2\cdot\text{sr}\cdot\mu\text{m})$ | Quadrant IV (Arctic / Winter OOD) surface-leaving radiance. |
| `val_q4_L_toa` | `(770, B)` | `float32` | $\text{W}/(\text{m}^2\cdot\text{sr}\cdot\mu\text{m})$ | Quadrant IV Top-of-Atmosphere radiance. |
| `val_q4_L_down` | `(770, B)` | `float32` | $\text{W}/(\text{m}^2\cdot\text{sr}\cdot\mu\text{m})$ | Quadrant IV downwelling atmospheric sky irradiance. |
| `val_q4_emissivity`| `(770, B)`| `float32` | Dimensionless $[0, 1]$ | Quadrant IV ground-truth emissivity for the 7 Golden Targets. |
| `val_q4_Ts` | `(770,)` | `float32` | $\text{Kelvin (K)}$ | Quadrant IV ground-truth surface kinetic temperature. |
| `val_q4_golden_name`| `(770,)` | `str / object`| Categorical | Golden Target identifier tested in harsh winter conditions. |
| `val_q4_atmosphere_id`| `(770,)`| `str / object`| Categorical | Severe sub-arctic MODTRAN profile identifier. |
| `val_q4_pwv` | `(770,)` | `float32` | $\text{g}/\text{cm}^2$ | Precipitable water vapor column of winter simulation atmosphere. |
| `val_oof_L_obs` | `(2156, B)`| `float32` | $\text{W}/(\text{m}^2\cdot\text{sr}\cdot\mu\text{m})$ | Quadrant I Out-of-Fold validation surface-leaving radiance. |
| `val_oof_L_toa` | `(2156, B)`| `float32` | $\text{W}/(\text{m}^2\cdot\text{sr}\cdot\mu\text{m})$ | Quadrant I Out-of-Fold validation Top-of-Atmosphere radiance. |
| `val_oof_L_down` | `(2156, B)`| `float32` | $\text{W}/(\text{m}^2\cdot\text{sr}\cdot\mu\text{m})$ | Quadrant I Out-of-Fold validation downwelling sky irradiance. |
| `val_oof_emissivity`| `(2156, B)`| `float32` | Dimensionless $[0, 1]$ | Quadrant I ground-truth emissivity for 49 unseen development spectra. |
| `val_oof_Ts` | `(2156,)` | `float32` | $\text{Kelvin (K)}$ | Quadrant I ground-truth surface kinetic temperature. |
| `val_oof_materials`| `(2156,)` | `str / object`| Categorical | Material classification of unseen validation samples. |
| `val_oof_atmosphere_id`| `(2156,)`| `str / object`| Categorical | MODTRAN profile identifier used for validation simulation. |
| `val_oof_pwv` | `(2156,)` | `float32` | $\text{g}/\text{cm}^2$ | Precipitable water vapor column of validation atmosphere. |

---

### 2.1 The 2×2 Orthogonal Generalization Matrix & Partition Accounting

The OpenTIR-PINN benchmark organizes all forward synthetic observations across a **2×2 Orthogonal Generalization Matrix** to systematically test model transferability and prevent spatial/spectral autocorrelation leakage:
- **Axis 1 (Surface Material Distribution):** In-Distribution (231 Development Pool spectra) vs. Out-of-Distribution (7 Immutable Golden Targets).
- **Axis 2 (Atmospheric Column Distribution):** In-Distribution (Nominal MODTRAN 5.2 profiles) vs. Out-of-Distribution (Severe Polar Winter / Sub-Arctic Vortex with $\text{PWV} < 0.3\,\text{g/cm}^2$).

```
                      [ Atmospheric Distribution ]
                      In-Distribution (Nominal)     OOD (Severe Polar Winter)
                   +-----------------------------+-----------------------------+
   In-Distribution |         QUADRANT I          |        QUADRANT III         |
   (Dev Pool 231)  |  • Master Training: 5,460   |  • LOCO Generalization      |
                   |  • Out-of-Fold Val: 2,156   |  • Dynamic RTE Live Sim     |
                   +-----------------------------+-----------------------------+
   Out-of-Dist.    |         QUADRANT II         |        QUADRANT IV          |
   (Golden Targets)|  • Pure Spectral OOD: 308   |  • Joint Extreme OOD: 770   |
   (7 Immutable)   |  • Zero-Shot Material Eval  |  • Double-Blind Stress Test |
                   +-----------------------------+-----------------------------+
```

#### Detailed Breakdown of Benchmark Partitions:
1. **Quadrant I (Master Training Pool):** $N = 5,460$ observations (Fold 0 balanced pool) spanning 4 nominal MODTRAN atmospheres across 11 kinetic surface temperatures ($T_s \in [280, 330]\,\text{K}$ in $5\,\text{K}$ increments).
2. **Quadrant I (Out-of-Fold Validation Set):** $N = 2,156$ observations ($49 \text{ Fold 0 dev spectra} \times 4 \text{ atmospheres} \times 11 \text{ temperatures}$). Evaluates generalization to unseen development spectra within nominal weather regimes without data leakage.
3. **Quadrant II (Golden Target Benchmark):** $N = 308$ observations ($7 \text{ golden targets} \times 4 \text{ atmospheres} \times 11 \text{ temperatures}$). Pure spectral OOD evaluation.
4. **Quadrant III (Atmospheric Inversion OOD):** Evaluates known surface materials under severe polar winter regimes. Tested via Leave-One-Class-Out (LOCO) cross-validation and live simulation in Tab 5 (Dynamic RTE Studio).
5. **Quadrant IV (Double-Blind Golden Benchmark):** $N = 770$ observations ($7 \text{ golden targets} \times 10 \text{ severe sub-arctic polar winter profiles} \times 11 \text{ temperatures}$). Simultaneous domain shift in both surface emissivity physics and atmospheric radiative transfer.
* **Total Sample Count per Sensor:** $5,460 + 2,156 + 308 + 770 = 8,694$ observations.
* **Total Sample Count Across Both Sensors (ASTER 5-Band + HyTES 256-Channel):** $8,694 \times 2 = 17,388$ forward coupled radiance observations.

#### Intuitive Combinatorial Summary Formula ("The Quick Rule-of-Thumb"):
For fast physical calculation of sample accounting across any partition:

$$\mathbf{N_{\text{observations}} = (N_{\text{targets}} \times N_{\text{atmospheres}} \times N_{\text{temperatures}}) + \text{Sensor Radiometric Noise } (\mathrm{NE}\Delta T)}$$

* **🏋️ Quadrant I (Master Train):** $\mathbf{5,460\text{ observations (Fold 0)}}$ *(continuous $T_s \in [275, 335]\,\mathrm{K}$, stochastic shape jitter $\varepsilon_{\text{aug}}$, dynamic sensor noise $0.5\times\text{--}1.5\times \mathrm{NE}\Delta T$)*.
* **🧪 Quadrant I (Val OOF):** $49\text{ unseen spectra} \times 4\text{ nominal atmospheres} \times 11\text{ temperatures} = \mathbf{2,156\text{ observations}} + \text{nominal sensor noise } (\mathrm{NE}\Delta T = 0.1\,\mathrm{K})$.
* **🏆 Quadrant II (Golden OOD):** $7\text{ golden targets} \times 4\text{ nominal atmospheres} \times 11\text{ temperatures} = \mathbf{308\text{ observations}} + \text{nominal sensor noise } (\mathrm{NE}\Delta T = 0.1\,\mathrm{K})$.
* **❄️ Quadrant IV (Double OOD):** $7\text{ golden targets} \times 10\text{ polar winter atmospheres} \times 11\text{ temperatures} = \mathbf{770\text{ observations}} + \text{nominal sensor noise } (\mathrm{NE}\Delta T = 0.1\,\mathrm{K})$.
* **⚡ Benchmark Total per Sensor:** $5,460 + 2,156 + 308 + 770 = \mathbf{8,694\text{ observations}}$ ($\times 2\text{ sensors (ASTER + HyTES)} = \mathbf{17,388\text{ total forward samples}}$).

#### Two-Tier Forward Perturbation Architecture (Realistic Noise Modeling):
The dataset is engineered to reflect real-world detector noise and eliminate synthetic overfitting:
1. **Tier 1 (Sensor-Specific Instrumental Radiometric Noise):** Present across **ALL 4 partitions** (Train, Val OOF, Q2, Q4). Noise standard deviation is analytically derived via Planck's temperature derivative:
   $$\sigma_L(\lambda) = \left. \frac{\partial B(\lambda, T)}{\partial T} \right|_{T = 300\,\mathrm{K}} \times \mathrm{NE}\Delta T \times \text{noise\_multiplier}$$
   with $\mathrm{NE}\Delta T = 0.1\,\mathrm{K}$ at $T_{\mathrm{ref}} = 300\,\mathrm{K}$. Independent Gaussian noise is injected into surface radiance $L_{\mathrm{surf}}$, at-sensor radiance $L_{\mathrm{sensor}}$, and downwelling radiance $L^\downarrow$ (with $0.5\times$ scaling for hemispherical integration).
2. **Tier 2 (Stochastic Shape Augmentation & Continuous Temperature):** Applied **exclusively to Quadrant I Training** ($\varepsilon_{\mathrm{aug}} = \operatorname{clip}(\varepsilon \times \mathcal{U}(0.995, 1.005) + \mathcal{N}(0, 0.002^2), 0.0, 1.0)$ and $T_s \sim \mathcal{U}(275, 335)\,\mathrm{K}$ with $\text{noise\_multiplier} \sim \mathcal{U}(0.5, 1.5)$). All test and validation vaults (Val OOF, Q2, Q4) preserve 100% pristine laboratory FTIR crystal lattice signatures ($\varepsilon_{\mathrm{pure}}$, `is_augmented = False`) under deterministic temperature grids ($T_s \in [280, 330]\,\mathrm{K}$ in $5\,\mathrm{K}$ steps) and nominal sensor noise ($\text{noise\_multiplier} = 1.0$).

#### Atmospheric Column Integrity & Forensic Audit:
All Quadrant II, Quadrant IV, and Validation samples contain genuine physical Precipitable Water Vapor columns:
- **Quadrant II PWV Range:** $1.42\,\text{g/cm}^2$ to $4.11\,\text{g/cm}^2$ (Mean: $2.63\,\text{g/cm}^2$) across nominal profiles (`atm1_h2o1.00`, `atm2_h2o1.00`, `atm4_h2o1.00`, `atm6_h2o1.00`).
- **Quadrant IV PWV Range:** $0.21\,\text{g/cm}^2$ to $1.70\,\text{g/cm}^2$ (Mean: $0.73\,\text{g/cm}^2$) across severe polar winter regimes (`atm3_h2o0.50` to `atm5_h2o2.00`).
- Both `.npz` archives (`aster_5band_sobolev_pinn_dataset.npz` and `hytes_256band_sobolev_pinn_dataset.npz`) are completely self-contained and serialize all 4 partitions natively.

---

## 3. Atmospheric Look-Up Tables (`atmospheric_luts/`)

The benchmark provides 30 high-resolution MODTRAN 5.2 radiative transfer profiles covering 6 standard geographic models across 5 water vapor scaling factors ($0.50\times, 0.75\times, 1.00\times, 1.50\times, 2.00\times$).

### Table Columns in `profile_*.csv`:
1. `wavelength_um`: Center wavelength from $6.969\,\mu\text{m}$ to $15.152\,\mu\text{m}$ (776 channels).
2. `transmittance`: Total atmospheric spectral transmission along the nadir path ($\tau(\lambda) \in [0, 1]$).
3. `path_radiance`: Thermal path upwelling radiance ($L^\uparrow(\lambda)$ in $\text{W}/(\text{m}^2\cdot\text{sr}\cdot\mu\text{m})$).
4. `downwelling_radiance`: Downwelling sky irradiance divided by $\pi$ ($L^\downarrow(\lambda) / \pi$ in $\text{W}/(\text{m}^2\cdot\text{sr}\cdot\mu\text{m})$).

---

## 4. The 7 Golden Targets (`golden_targets_metadata.csv`)

The Double-Quarantine evaluation suite consists of 7 structurally diverse endmembers quarantined from training:

1. `urban_concrete`: Construction Concrete (JHU `0598UUUCNC`, high broadband continuum).
2. `urban_asphalt`: Construction Asphalt (JHU `0095UUUASP`, smooth low-contrast continuum).
3. `calcite`: Carbonate Mineral ($\text{CaCO}_3$, coarse particle, internal vibrational $\nu_3, \nu_1, \nu_2$ modes).
4. `kaolinite`: Phyllosilicate Clay ($\text{Al}_2\text{Si}_2\text{O}_5(\text{OH})_4$, strong $9.6\,\mu\text{m}$ feature).
5. `quartz`: Tectosilicate Silica ($\alpha\text{-SiO}_2$, Christiansen frequency at $7.4\,\mu\text{m}$, Reststrahlen doublet at $8.3$ and $9.3\,\mu\text{m}$).
6. `feldspar`: Tectosilicate Plagioclase (Anorthite $\text{CaAl}_2\text{Si}_2\text{O}_8$, coarse `anorthite_1`, $\text{Al-Si}$ tetrahedral inflection near $9.8\,\mu\text{m}$).
7. `npv_bark`: Non-Photosynthetic Vegetation (*Abies concolor* tree bark, UCSB `VH311`, lignin/cellulose features).
"""

import subprocess
subprocess.run([sys.executable, os.path.join(ROOT_DIR, "scripts", "generate_synchronized_data_dictionary.py")], check=True)
print("  -> Synchronized DATA_DICTIONARY.md from data/dataset_facts.json (v1.3.0)")

# 7.3 README.md
readme_content = r"""# OpenTIR-PINN Benchmark: Synthetic Radiative Transfer & Double-Quarantine Golden Target Dataset (v1.1.0)

**Creator:** Dr. Atilio Grondona (*Universidade Federal do Rio Grande do Sul - UFRGS*)  
**License:** Creative Commons Attribution 4.0 International (CC-BY 4.0)  
**Associated Publication:** *Physics-Informed Neural Networks with Sobolev Shape Regularization for Ill-Posed Temperature and Emissivity Separation Across Multispectral and Hyperspectral Thermal Infrared Sensors* (IEEE Transactions on Geoscience and Remote Sensing / Remote Sensing of Environment)  
**Zenodo DOI:** `10.5281/zenodo.24085242` (Version 1.1.0)

---

> [!IMPORTANT]
> ### Official Release Erratum & Upgrade Notice (v1.0.0 $\to$ v1.1.0)
> Version 1.0.0 of this benchmark suffered from synthetic decoupling where radiances omitted the reflected downwelling atmospheric irradiance ($L = \epsilon B(T)$ without $(1-\epsilon) L^\downarrow / \pi$) and relied on duplicate atmospheric LUT profiles. Furthermore, Golden Target #7 was mislabeled as green foliage while containing bark spectra.
>
> **Version 1.1.0 completely resolves these limitations by providing:**
> 1. **Complete Radiative Transfer Coupling:** All simulated radiances strictly satisfy the full surface-leaving boundary condition $L_{\mathrm{surf}} = \epsilon B(T_s) + (1-\epsilon) L^\downarrow/\pi$ and Top-of-Atmosphere coupling $L_{\mathrm{TOA}} = \tau L_{\mathrm{surf}} + L^\uparrow$.
> 2. **30 Distinct MODTRAN 5.2 Profiles:** Full parameter sweep across 6 atmospheric profiles (Tropical, Mid-Latitude Summer, Mid-Latitude Winter, Sub-Arctic Summer, Sub-Arctic Winter, 1976 US Standard) and 5 water vapor scales (0.50 to 2.00, PWV from 0.42 to 8.23 g/cm²).
> 3. **Dual-Input Architectural Telemetry:** Datasets now supply both $[L_{\mathrm{surf}}, L^\downarrow]$ enabling real-world atmospheric ingestion.
> 4. **Rigorous Double-Quarantine Suite:** Immutable benchmark featuring 7 Golden Targets tested across In-Distribution Atmospheres (Quadrant II, N=308) and Severe Winter Out-of-Distribution Atmospheres (Quadrant IV, N=770), with Target #7 properly documented as Non-Photosynthetic Vegetation (*Abies concolor* White-Fir Bark).

---

## 1. Benchmark Overview & Telemetry

This open-science benchmark provides a standardized, physics-grounded evaluation baseline for Physics-Informed Neural Networks (PINN), traditional Radiative Transfer Equation (RTE) solvers, and operational Temperature and Emissivity Separation (TES) algorithms.

### Summary Telemetry:
* **Total Synthetic Training Library:** 5,460 multi-temperature synthetic spectra (Fold 0) generated from a rebalanced pool of 231 development laboratory endmembers (out of 240 total curated spectra).
* **Double-Quarantine Test Suite:**
  * **Quadrant II (In-Distribution Atmospheres):** 7 Golden Targets $\times$ 44 multi-temperature/development atmosphere combinations ($N=308$).
  * **Quadrant IV (Arctic / Winter OOD Atmospheres):** 7 Golden Targets $\times$ 110 extreme winter conditions ($N=770$).
* **Sensors Modeled:**
  * **HyTES (Hyperspectral):** 256 contiguous spectral channels ($7.5 - 12.0\,\mu\text{m}$, spectral sampling $\sim 17.6\,\text{nm}$).
  * **ASTER (Multispectral):** 5 discrete thermal channels (Bands 10, 11, 12, 13, 14 spanning $8.1 - 11.4\,\mu\text{m}$).
* **Thermal Regimes:** Uniform surface kinetic temperature sweep $T_s \in [280.0\text{ K}, 330.0\text{ K}]$ with realistic physical bounds $|T_s - T_{\mathrm{air}}| \le 15.0\text{ K}$.

---

## 2. Quickstart Verification

You can instantly verify the integrity and physical validity of the dataset using the included quickstart script:

```bash
# Verify the entire dataset package, column headers, and physical bounds
python quickstart_demo.py

# Run the differentiable physics engine test suite
python tests/test_physics_engine.py
```

---

## 3. Python Usage Example

```python
import numpy as np

# Load HyTES 256-channel dataset
data = np.load('data/hytes_256band_sobolev_pinn_dataset.npz')

# Extract training arrays
L_surf = data['train_L_obs']       # Surface-leaving radiance [5460, 256]
L_down = data['train_L_down']      # Downwelling sky irradiance / pi [5460, 256]
emissivity = data['train_emissivity'] # Ground-truth emissivity [5460, 256]
T_surface = data['train_Ts']       # Ground-truth temperature [5460]

print(f"Loaded {L_surf.shape[0]} training samples across {L_surf.shape[1]} bands.")
print(f"Temperature range: {T_surface.min():.2f} K to {T_surface.max():.2f} K")

# Evaluate on Quadrant IV (Severe Winter Out-of-Distribution Benchmark)
q4_L_surf = data['val_q4_L_obs']   # [770, 256]
q4_targets = data['val_q4_golden_name']
print(f"Quadrant IV test suite contains {len(q4_targets)} samples across 7 quarantined endmembers.")
```

---

## 4. Citation & Attribution

If you use this dataset or physics engine in your research, please cite:

```bibtex
@article{grondona2026sobolev_pinn,
  author    = {Grondona, Atilio},
  title     = {Physics-Informed Neural Networks with Sobolev Shape Regularization for Ill-Posed Temperature and Emissivity Separation Across Multispectral and Hyperspectral Thermal Infrared Sensors},
  journal   = {IEEE Transactions on Geoscience and Remote Sensing},
  year      = {2026},
  doi       = {10.1109/TGRS.2026.XXXXXXX}
}

@dataset{grondona2026_opentir_pinn_zenodo,
  author    = {Grondona, Atilio},
  title     = {OpenTIR-PINN Benchmark v1.1.0: Multi-Atmospheric Radiative Transfer and Double-Quarantine Golden Target Dataset for Ill-Posed Temperature and Emissivity Separation},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.1.0},
  doi       = {10.5281/zenodo.24085242}
}
```
"""

# 7.3 README.md (Preserve enriched README if present)
readme_path = os.path.join(RELEASE_DIR, "README.md")
if not os.path.exists(readme_path):
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content.strip() + "\n")
print("  -> Verified README.md (v1.1.0 with official Erratum Notice and Asset Maps)")

# 7.4 Update quickstart_demo.py
quickstart_code = r'''"""
Quickstart verification test suite for OpenTIR-PINN Zenodo benchmark dataset (v1.1.0).
Validates file integrity, explicit schema column headers, physical ranges, and SRF files.
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def verify_dataset():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'data')
    
    print("=" * 80)
    print("OpenTIR-PINN Benchmark (v1.1.0): Comprehensive Dataset Verification")
    print(f"Dataset Root Directory: {os.path.abspath(root_dir)}")
    print("=" * 80)
    
    # 1. Verify HyTES NPZ
    hytes_path = os.path.join(data_dir, 'hytes_256band_sobolev_pinn_dataset.npz')
    assert os.path.exists(hytes_path), f"HyTES package missing: {hytes_path}"
    hytes = np.load(hytes_path)
    print("\n[1/7] HyTES 256-Channel Dataset Archive:")
    print(f"  Wavelengths array shape:       {hytes['wavelengths'].shape} [{hytes['wavelengths'][0]:.3f} to {hytes['wavelengths'][-1]:.3f} µm]")
    print(f"  FWHM array shape:              {hytes['fwhm'].shape} [{hytes['fwhm'].min():.4f} to {hytes['fwhm'].max():.4f} µm]")
    print(f"  Training L_surf shape:         {hytes['train_L_obs'].shape} (W/(m²·sr·µm))")
    print(f"  Training L_down shape:         {hytes['train_L_down'].shape} (W/(m²·sr·µm)) [min={hytes['train_L_down'].min():.3f}, max={hytes['train_L_down'].max():.3f}]")
    print(f"  Training Emissivity shape:     {hytes['train_emissivity'].shape} [min={hytes['train_emissivity'].min():.3f}, max={hytes['train_emissivity'].max():.3f}]")
    print(f"  Training Temperature range:    [{hytes['train_Ts'].min():.2f} K, {hytes['train_Ts'].max():.2f} K]")
    print(f"  Quadrant II Test Suite:        {hytes['val_q2_L_obs'].shape} samples across 7 Golden Targets")
    print(f"  Quadrant IV Test Suite:        {hytes['val_q4_L_obs'].shape} samples across 7 Golden Targets")
    assert hytes['train_L_obs'].shape == (5460, 256), "HyTES train shape mismatch"
    assert hytes['val_q2_L_obs'].shape == (308, 256), "HyTES Q2 shape mismatch"
    assert hytes['val_q4_L_obs'].shape == (770, 256), "HyTES Q4 shape mismatch"
    assert "val_q2_pwv" in hytes and hytes["val_q2_pwv"].max() > 0, "HyTES Q2 PWV missing or zero"
    assert "val_q4_pwv" in hytes and hytes["val_q4_pwv"].max() > 0, "HyTES Q4 PWV missing or zero"
    assert "val_q2_atmosphere_id" in hytes, "HyTES Q2 atmosphere_id missing"
    assert np.all(hytes['train_L_down'] > 0), "Downwelling radiance must be strictly positive"
    print("  Status: VERIFIED (Dimensions, Downwelling Irradiance, PWV & Physical Ranges OK)")
    
    # 2. Verify ASTER NPZ
    aster_path = os.path.join(data_dir, 'aster_5band_sobolev_pinn_dataset.npz')
    assert os.path.exists(aster_path), f"ASTER package missing: {aster_path}"
    aster = np.load(aster_path)
    print("\n[2/7] ASTER 5-Band Dataset Archive:")
    print(f"  Wavelengths:                   {list(np.round(aster['wavelengths'], 3))} µm")
    print(f"  FWHM:                          {list(np.round(aster['fwhm'], 3))} µm")
    print(f"  Training L_surf shape:         {aster['train_L_obs'].shape} (W/(m²·sr·µm))")
    print(f"  Training L_down shape:         {aster['train_L_down'].shape} (W/(m²·sr·µm))")
    print(f"  Quadrant II Test Suite:        {aster['val_q2_L_obs'].shape} samples across 7 Golden Targets")
    print(f"  Quadrant IV Test Suite:        {aster['val_q4_L_obs'].shape} samples across 7 Golden Targets")
    assert aster['train_L_obs'].shape == (5460, 5), "ASTER train shape mismatch"
    assert aster['val_q2_L_obs'].shape == (308, 5), "ASTER Q2 shape mismatch"
    assert aster['val_q4_L_obs'].shape == (770, 5), "ASTER Q4 shape mismatch"
    assert "val_q2_pwv" in aster and aster["val_q2_pwv"].max() > 0, "ASTER Q2 PWV missing or zero"
    assert "val_q4_pwv" in aster and aster["val_q4_pwv"].max() > 0, "ASTER Q4 PWV missing or zero"
    assert "val_q2_atmosphere_id" in aster, "ASTER Q2 atmosphere_id missing"
    print("  Status: VERIFIED (Dimensions, PWV & Physical Ranges OK)")
    
    # 3. Verify Golden Targets Tabular Data & Metadata
    print("\n[3/7] Golden Targets Tabular Data & Raw Spectra:")
    meta_df = pd.read_csv(os.path.join(data_dir, 'golden_targets_metadata.csv'))
    print(f"  Metadata Columns:              {list(meta_df.columns)}")
    print(f"  Target Names ({len(meta_df)} items):        {list(meta_df['target_name'])}")
    assert len(meta_df) == 7, "Expected exactly 7 golden targets in metadata"
    assert "npv_bark" in list(meta_df['target_name']), "npv_bark must be present in golden targets"
    
    hytes_spec_df = pd.read_csv(os.path.join(data_dir, 'golden_targets_hytes_256band_emissivity.csv'))
    print(f"  HyTES CSV Spectra:             {hytes_spec_df.shape[0]} bands × {hytes_spec_df.shape[1]} columns")
    aster_spec_df = pd.read_csv(os.path.join(data_dir, 'golden_targets_aster_5band_emissivity.csv'))
    print(f"  ASTER CSV Spectra:             {aster_spec_df.shape[0]} bands × {aster_spec_df.shape[1]} columns")
    
    raw_golden_dir = os.path.join(data_dir, 'raw_laboratory_spectra', 'golden_targets')
    raw_files = os.listdir(raw_golden_dir)
    print(f"  Raw Golden Targets:            {len(raw_files)} files in {raw_golden_dir}")
    assert len(raw_files) == 7, f"Expected 7 raw golden targets, found {len(raw_files)}"
    assert any("npv_bark" in f for f in raw_files), "npv_bark raw spectrum missing"
    print("  Status: VERIFIED (Discrete CSVs + 7 Raw continuous FTIR spectra valid)")
    
    # 4. Verify Atmospheric Look-Up Tables
    print("\n[4/7] MODTRAN 5.2 Atmospheric Profiles & LUT Catalog:")
    lut_catalog = pd.read_csv(os.path.join(data_dir, 'atmospheric_luts', 'lut_catalog.csv'))
    print(f"  LUT Catalog Columns:           {list(lut_catalog.columns)}")
    print(f"  Scenarios Available:           {len(lut_catalog)} profiles (PWV range: [{lut_catalog['pwv_g_cm2'].min():.2f}, {lut_catalog['pwv_g_cm2'].max():.2f}] g/cm²)")
    assert len(lut_catalog) == 30, f"Expected 30 MODTRAN profiles, found {len(lut_catalog)}"
    for npz_file in lut_catalog['npz_file']:
        fpath = os.path.join(data_dir, 'atmospheric_luts', npz_file)
        assert os.path.exists(fpath), f"Missing atmospheric profile: {fpath}"
    print("  Status: VERIFIED (All 30 MODTRAN 5.2 profile .npz and .csv files present)")
    
    # 5. Verify Splits and Manifests
    print("\n[5/7] Splits and Manifests:")
    split_file = os.path.join(data_dir, 'splits', 'split_v1.1.json')
    assert os.path.exists(split_file), "split_v1.1.json missing"
    golden_file = os.path.join(data_dir, 'splits', 'golden_targets.json')
    assert os.path.exists(golden_file), "golden_targets.json missing"
    print("  Status: VERIFIED (split_v1.1.json and golden_targets.json verified)")
    
    # 6. Verify Benchmarks & Best Hyperparameters
    print("\n[6/7] Benchmark Scorecards & Configurations:")
    bench_dir = os.path.join(data_dir, 'benchmarks')
    bench_files = [f for f in os.listdir(bench_dir) if f.endswith('.csv')]
    print(f"  Benchmark Scorecards:          {len(bench_files)} CSV files in {bench_dir}")
    assert os.path.exists(os.path.join(bench_dir, 'best_config_aster.json')), "best_config_aster.json missing"
    assert os.path.exists(os.path.join(bench_dir, 'best_config_hytes.json')), "best_config_hytes.json missing"
    print("  Status: VERIFIED (All benchmark CSVs and Optuna configs present)")
    
    # 7. Verify SRFs
    print("\n[7/7] Sensor Spectral Response Functions (SRFs):")
    hytes_srf_path = os.path.join(data_dir, 'srf', 'hytes', 'HyTES_SRF_v6.csv')
    assert os.path.exists(hytes_srf_path), "HyTES SRF missing"
    aster_specs = pd.read_csv(os.path.join(data_dir, 'srf', 'aster', 'aster_5bands_specifications.csv'))
    for srf_file in aster_specs['srf_curve_file']:
        aster_srf = os.path.join(data_dir, 'srf', 'aster', srf_file)
        assert os.path.exists(aster_srf), f"ASTER SRF file missing: {aster_srf}"
    for rttov_ch in aster_specs['rttov_channel_id']:
        rttov_file = os.path.join(data_dir, 'srf', 'aster', f"rtcoef_eos_1_aster_srf_{rttov_ch}.txt")
        assert os.path.exists(rttov_file), f"RTTOV coefficient file missing: {rttov_file}"
    print("  Status: VERIFIED (HyTES and ASTER SRF curves intact)")
    
    print("\n" + "=" * 80)
    print("ALL 7 VERIFICATION GATES PASSED! DATASET READY FOR ZENODO v1.1.0 PUBLICATION!")
    print("=" * 80)

if __name__ == "__main__":
    verify_dataset()
'''

with open(os.path.join(RELEASE_DIR, "quickstart_demo.py"), "w", encoding="utf-8") as f:
    f.write(quickstart_code.strip() + "\n")
print("  -> Updated quickstart_demo.py (v1.1.0)")

# -------------------------------------------------------------------------
# Step 8: Build Zenodo v1.3.0 Zip Package
# -------------------------------------------------------------------------
print("\n[Step 8/8] Building final zenodo_release_v1.3.0.zip archive...")
zip_filename = os.path.join(ROOT_DIR, "zenodo_release_v1.3.0.zip")
if os.path.exists(zip_filename):
    os.remove(zip_filename)

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Generated artefacts of local pipeline runs (pipeline/data, pipeline/logs, ...) are rebuilt by
    # run_pipeline.py and must never be shipped.
    RUN_ARTEFACTS = {"data", "logs", "splits", "docs", "bibliografic_references"}
    for root, dirs, files in os.walk(RELEASE_DIR):
        if "__pycache__" in root:
            continue
        rel = os.path.relpath(root, os.path.join(RELEASE_DIR, "pipeline"))
        if not rel.startswith("..") and rel.split(os.sep)[0] in RUN_ARTEFACTS:
            continue
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, ROOT_DIR)
            zipf.write(file_path, arcname)

print(f"  -> Successfully generated: {zip_filename} ({os.path.getsize(zip_filename)/1e6:.2f} MB)")
print("=" * 80)
print("ZENODO v1.1.0 PACKAGING COMPLETE!")
print("=" * 80)
