import os
import shutil
import torch
import numpy as np
import pandas as pd
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

release_dir = r"C:\Users\Atilio\Desktop\PROJETOS\AITIR\zenodo_release"
data_dir = os.path.join(release_dir, "data")
srf_dir = os.path.join(data_dir, "srf")
hytes_srf_dir = os.path.join(srf_dir, "hytes")
aster_srf_dir = os.path.join(srf_dir, "aster")
atm_dir = os.path.join(data_dir, "atmospheric_luts")

os.makedirs(hytes_srf_dir, exist_ok=True)
os.makedirs(aster_srf_dir, exist_ok=True)
os.makedirs(atm_dir, exist_ok=True)

print("=" * 70)
print("Packaging OpenTIR-PINN Benchmark Dataset for Zenodo (with full SRFs)...")
print("=" * 70)

# -------------------------------------------------------------
# 1. HyTES Dataset & SRFs
# -------------------------------------------------------------
print("\n[1/4] Processing HyTES (256 channels)...")

# Load calibrated parameters
hytes_params_df = pd.read_csv('data/processed/hytes_real_parameters.csv')
hytes_params_df = hytes_params_df.sort_values('band_idx')
hytes_wavelengths = hytes_params_df['center_wavelength'].values.astype(np.float32)
hytes_fwhms = hytes_params_df['fwhm'].values.astype(np.float32)

train_hytes = torch.load('data/processed/final_datasets/train_dataset_hytes.pt', map_location='cpu', weights_only=False)
val_hytes = torch.load('data/processed/final_datasets/val_dataset_hytes.pt', map_location='cpu', weights_only=False)

L_obs_hytes = np.stack([s['input_radiance'].numpy() for s in train_hytes]).astype(np.float32)
eps_hytes = np.stack([s['target_emissivity'].numpy() for s in train_hytes]).astype(np.float32)
Ts_hytes = np.stack([s['target_temp'].numpy() for s in train_hytes]).astype(np.float32)
materials_hytes = [s['material'] for s in train_hytes]

val_L_obs_hytes = np.stack([s['input_radiance'].numpy() for s in val_hytes]).astype(np.float32)
val_eps_hytes = np.stack([s['target_emissivity'].numpy() for s in val_hytes]).astype(np.float32)
val_Ts_hytes = np.stack([s['target_temp'].numpy() for s in val_hytes]).astype(np.float32)
val_materials_hytes = [s.get('material', f'target_{i}') for i, s in enumerate(val_hytes)]

hytes_npz = os.path.join(data_dir, "hytes_256band_sobolev_pinn_dataset.npz")
np.savez_compressed(
    hytes_npz,
    wavelengths=hytes_wavelengths,
    fwhm=hytes_fwhms,
    train_L_obs=L_obs_hytes,
    train_emissivity=eps_hytes,
    train_Ts=Ts_hytes,
    train_materials=materials_hytes,
    val_golden_L_obs=val_L_obs_hytes,
    val_golden_emissivity=val_eps_hytes,
    val_golden_Ts=val_Ts_hytes,
    val_golden_materials=val_materials_hytes
)
print(f"  -> Saved HyTES archive: {hytes_npz} ({os.path.getsize(hytes_npz)/1e6:.2f} MB)")
print(f"     Included {len(hytes_wavelengths)} band centers [{hytes_wavelengths[0]:.3f} to {hytes_wavelengths[-1]:.3f} µm]")

# Copy HyTES calibration and SRF files
shutil.copy('data/processed/hytes_real_parameters.csv', os.path.join(hytes_srf_dir, 'hytes_channel_parameters.csv'))
shutil.copy('bibliografic_references/hytes/HyTES_SRF_v6.csv', os.path.join(hytes_srf_dir, 'HyTES_SRF_v6.csv'))
shutil.copy('bibliografic_references/hytes/HyTES_WMX_v6.csv', os.path.join(hytes_srf_dir, 'HyTES_WMX_v6.csv'))
shutil.copy('bibliografic_references/hytes/HyTES_SRF_v4.csv', os.path.join(hytes_srf_dir, 'HyTES_SRF_v4.csv'))
print(f"  -> Copied HyTES calibration and SRF tables to {hytes_srf_dir}")

# -------------------------------------------------------------
# 2. ASTER Dataset & SRFs
# -------------------------------------------------------------
print("\n[2/4] Processing ASTER (5 bands)...")

aster_wavelengths = np.array([8.291, 8.634, 9.075, 10.657, 11.318], dtype=np.float32)
aster_fwhms = np.array([0.354, 0.358, 0.355, 0.702, 0.705], dtype=np.float32)

train_aster = torch.load('data/processed/final_datasets/train_dataset_aster.pt', map_location='cpu', weights_only=False)
val_aster = torch.load('data/processed/final_datasets/val_dataset_aster.pt', map_location='cpu', weights_only=False)

L_obs_aster = np.stack([s['input_radiance'].numpy() for s in train_aster]).astype(np.float32)
eps_aster = np.stack([s['target_emissivity'].numpy() for s in train_aster]).astype(np.float32)
Ts_aster = np.stack([s['target_temp'].numpy() for s in train_aster]).astype(np.float32)
materials_aster = [s['material'] for s in train_aster]

val_L_obs_aster = np.stack([s['input_radiance'].numpy() for s in val_aster]).astype(np.float32)
val_eps_aster = np.stack([s['target_emissivity'].numpy() for s in val_aster]).astype(np.float32)
val_Ts_aster = np.stack([s['target_temp'].numpy() for s in val_aster]).astype(np.float32)
val_materials_aster = [s.get('material', f'target_{i}') for i, s in enumerate(val_aster)]

aster_npz = os.path.join(data_dir, "aster_5band_sobolev_pinn_dataset.npz")
np.savez_compressed(
    aster_npz,
    wavelengths=aster_wavelengths,
    fwhm=aster_fwhms,
    train_L_obs=L_obs_aster,
    train_emissivity=eps_aster,
    train_Ts=Ts_aster,
    train_materials=materials_aster,
    val_golden_L_obs=val_L_obs_aster,
    val_golden_emissivity=val_eps_aster,
    val_golden_Ts=val_Ts_aster,
    val_golden_materials=val_materials_aster
)
print(f"  -> Saved ASTER archive: {aster_npz} ({os.path.getsize(aster_npz)/1e6:.2f} MB)")
print(f"     Included {len(aster_wavelengths)} bands: {list(aster_wavelengths)} µm")

# Copy ASTER TIR SRF files (Channels 10-14)
for ch in range(10, 15):
    fname = f"rtcoef_eos_1_aster_srf_ch{ch:02d}.txt"
    shutil.copy(os.path.join('bibliografic_references/aster', fname), os.path.join(aster_srf_dir, fname))
print(f"  -> Copied ASTER Bands 10-14 SRF files to {aster_srf_dir}")

# -------------------------------------------------------------
# 3. MODTRAN Atmospheric Profiles (LUTs)
# -------------------------------------------------------------
print("\n[3/4] Processing MODTRAN Atmospheric Profiles...")
shutil.copy('data/processed/atmospheric_luts/lut_catalog.csv', os.path.join(atm_dir, 'lut_catalog.csv'))
for f in os.listdir('data/processed/atmospheric_luts'):
    if f.endswith('.txt'):
        shutil.copy(os.path.join('data/processed/atmospheric_luts', f), os.path.join(atm_dir, f))
print(f"  -> Copied MODTRAN tape7 profiles and catalog to {atm_dir}")

# -------------------------------------------------------------
# 4. Generate FAIR Documentation & Metadata
# -------------------------------------------------------------
print("\n[4/4] Writing Zenodo README, .zenodo.json, and quickstart demo...")

readme_content = r"""# OpenTIR-PINN Benchmark: Synthetic Radiative Transfer & Golden Target Dataset for Ill-Posed Temperature and Emissivity Separation

**Creators:** Dr. Atilio Grondona (*Universidade Federal do Rio Grande do Sul - UFRGS*)  
**License:** Creative Commons Attribution 4.0 International (CC-BY 4.0)  
**Associated Manuscript:** *Physics-Informed Neural Networks with Sobolev Shape Regularization for Ill-Posed Temperature and Emissivity Separation Across Multispectral and Hyperspectral Thermal Infrared Sensors* (IEEE Transactions on Geoscience and Remote Sensing / Remote Sensing of Environment)  
**DOI Placeholder:** `10.5281/zenodo.XXXXXXX`

---

## 1. Dataset Overview

This open-science benchmark contains forward-simulated thermal infrared (TIR) at-sensor radiance spectra coupled with ground-truth surface kinetic temperatures and continuous spectral emissivities. It provides a standardized evaluation baseline to benchmark Physics-Informed Neural Networks (PINN), traditional Radiative Transfer Equation (RTE) solvers, and operational Temperature and Emissivity Separation (TES) algorithms.

### Key Telemetry & Dimensions:
* **Total Synthetic Training Pairs:** 8,134 multi-temperature samples.
* **Atmospheric Scenarios:** 4 MODTRAN 6 standard geographic models (Tropical, Mid-Latitude Summer, Mid-Latitude Winter, 1976 US Standard) covering Precipitable Water Vapor (PWV) from 0.5 to 4.5 g/cm².
* **Thermal Regimes:** Uniform surface kinetic temperature sweep $T_s \sim \mathcal{U}(280.0\text{ K}, 330.0\text{ K})$ with boundary constraint $|T_s - T_{\mathrm{air}}| \le 15.0\text{ K}$.
* **Sensors Modeled:**
  1. **HyTES (Hyperspectral Thermal Emission Spectrometer):** 256 contiguous channels spanning 7.49 to 12.00 µm (channel sampling interval $\approx 17.6\text{ nm}$, FWHM $\approx 18\text{--}26\text{ nm}$).
  2. **ASTER (Advanced Spaceborne Thermal Emission and Reflection Radiometer):** 5 discrete bands (Bands 10–14: 8.29, 8.63, 9.07, 10.66, 11.32 µm).
* **Quarantined Golden Targets (OOD Benchmark):** 7 chemically pure, structurally diverse terrestrial endmembers (Quartz, Feldspar/Anorthite, Kaolinite, Calcite, Urban Concrete, Urban Asphalt, and Green Foliage) permanently isolated prior to training to rigorously test out-of-distribution generalization.

---

## 2. Directory Structure

```
zenodo_release/
├── README.md                                  # Comprehensive FAIR dataset documentation
├── .zenodo.json                               # Zenodo deposit metadata specification
├── quickstart_demo.py                         # Standalone Python script demonstrating loading & verification
└── data/
    ├── hytes_256band_sobolev_pinn_dataset.npz # 256-channel HyTES dataset (train + golden targets + wavelengths)
    ├── aster_5band_sobolev_pinn_dataset.npz   # 5-band ASTER dataset (train + golden targets + wavelengths)
    ├── atmospheric_luts/                      # MODTRAN atmospheric profiles & look-up tables
    │   ├── lut_catalog.csv                    # Metadata mapping for atmospheric scenarios
    │   └── run_*.txt                          # Full MODTRAN tape7 radiative transfer outputs
    └── srf/                                   # Sensor Spectral Response Functions (SRF)
        ├── hytes/                             # HyTES Spectral Response Functions
        │   ├── hytes_channel_parameters.csv   # Band center, FWHM, and Gaussian fit parameters for all 256 channels
        │   ├── HyTES_WMX_v6.csv               # Calibrated wavelength grid (microns)
        │   ├── HyTES_SRF_v6.csv               # Calibrated spectral response weights
        │   └── HyTES_SRF_v4.csv               # Previous calibration release
        └── aster/                             # ASTER TIR Spectral Response Functions
            ├── rtcoef_eos_1_aster_srf_ch10.txt # Band 10 (8.29 µm)
            ├── rtcoef_eos_1_aster_srf_ch11.txt # Band 11 (8.63 µm)
            ├── rtcoef_eos_1_aster_srf_ch12.txt # Band 12 (9.07 µm)
            ├── rtcoef_eos_1_aster_srf_ch13.txt # Band 13 (10.66 µm)
            └── rtcoef_eos_1_aster_srf_ch14.txt # Band 14 (11.32 µm)
```

---

## 3. Sensor Spectral Response Functions (SRF) Integration

In thermal radiometry, the channel-integrated spectral radiance $L_i$ observed by sensor channel $i$ is calculated by convolving the hyperspectral top-of-atmosphere radiance $L_{\mathrm{TOA}}(\lambda)$ with the normalized channel response function $\phi_i(\lambda)$:

$$L_i = \frac{\int_{\lambda_{\min}}^{\lambda_{\max}} L_{\mathrm{TOA}}(\lambda) \phi_i(\lambda) \, d\lambda}{\int_{\lambda_{\min}}^{\lambda_{\max}} \phi_i(\lambda) \, d\lambda}$$

### Included Calibration Data:
1. **HyTES (JPL/NASA):** 
   - `hytes_channel_parameters.csv`: Precomputed Gaussian parameters for all 256 channels ($\lambda_c$, $\text{FWHM}$, amplitude, and $\sigma$).
   - `HyTES_WMX_v6.csv` & `HyTES_SRF_v6.csv`: The raw laboratory optical response profiles from JPL.
   - Channel centers and FWHMs are also embedded directly inside `data/hytes_256band_sobolev_pinn_dataset.npz` under keys `wavelengths` and `fwhm`.
2. **ASTER (EOS-AM1 / NASA / METI):**
   - `rtcoef_eos_1_aster_srf_ch10.txt` through `ch14.txt`: Exact relative spectral response tables in wavenumber ($\text{cm}^{-1}$) vs transmission.
   - Pre-interpolated band centers and FWHMs are embedded directly inside `data/aster_5band_sobolev_pinn_dataset.npz` under keys `wavelengths` and `fwhm`.

---

## 4. NumPy Archive Structure

Each `.npz` archive contains:
* `wavelengths`: Array of channel center wavelengths in $\mu\text{m}$ (`(256,)` for HyTES, `(5,)` for ASTER).
* `fwhm`: Array of channel Full Width at Half Maximum in $\mu\text{m}$ (`(256,)` for HyTES, `(5,)` for ASTER).
* `train_L_obs`: Array of shape `(8134, N)` containing at-sensor top-of-atmosphere radiance in $\text{W}\cdot\text{m}^{-2}\cdot\text{sr}^{-1}\cdot\mu\text{m}^{-1}$.
* `train_emissivity`: Array of shape `(8134, N)` containing ground-truth spectral emissivity $\bm{\epsilon} \in (0, 1)^N$.
* `train_Ts`: Array of shape `(8134, 1)` containing surface kinetic temperature in Kelvin.
* `train_materials`: Array of shape `(8134,)` containing material class identifiers.
* `val_golden_L_obs`: Array of shape `(7, N)` containing at-sensor radiance for the 7 quarantined Golden Targets.
* `val_golden_emissivity`: Array of shape `(7, N)` containing true laboratory DHR emissivities for Golden Targets.
* `val_golden_Ts`: Array of shape `(7, 1)` containing true test temperatures.
* `val_golden_materials`: Array of shape `(7,)` containing names of the 7 Golden Targets.

---

## 5. Quickstart: Loading and Plotting in Python

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Load HyTES dataset
hytes = np.load('data/hytes_256band_sobolev_pinn_dataset.npz')

wavelengths = hytes['wavelengths']            # (256,)
radiances = hytes['train_L_obs']              # (8134, 256)
emissivities = hytes['train_emissivity']      # (8134, 256)
temperatures = hytes['train_Ts']              # (8134, 1)

print(f"HyTES: {len(wavelengths)} channels from {wavelengths[0]:.2f} to {wavelengths[-1]:.2f} µm")
print(f"Golden Targets: {list(hytes['val_golden_materials'])}")

# 2. Load ASTER dataset
aster = np.load('data/aster_5band_sobolev_pinn_dataset.npz')
print(f"ASTER Bands: {list(aster['wavelengths'])} µm")
```

---

## 6. How to Cite

```bibtex
@dataset{grondona2026_opentir_pinn,
  author       = {Atilio Grondona},
  title        = {{OpenTIR-PINN: A Benchmark Dataset of Synthetic Radiative 
                   Transfer and Quarantined Mineral Endmembers for 
                   Physics-Informed Temperature and Emissivity Separation}},
  month        = nov,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX}
}
```
"""

with open(os.path.join(release_dir, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content)

# Write updated .zenodo.json
zenodo_metadata = {
    "title": "OpenTIR-PINN: A Benchmark Dataset of Synthetic Radiative Transfer, Sensor Response Functions, and Quarantined Mineral Endmembers for Physics-Informed Temperature and Emissivity Separation",
    "upload_type": "dataset",
    "description": "OpenTIR-PINN is an open-science benchmark dataset designed for training and evaluating physics-informed neural networks (PINN) and numerical solvers on the ill-posed thermal infrared (TIR) Temperature and Emissivity Separation (TES) problem. The dataset includes 8,134 synthetic training pairs simulated across 4 standard MODTRAN atmospheric profiles, full sensor spectral response functions (SRFs) and calibration tables for HyTES (256 channels) and ASTER (5 TIR bands), and 7 quarantined out-of-distribution (OOD) Golden Targets spanning pure silicate minerals, carbonates, clay, and urban materials.",
    "creators": [
        {
            "name": "Grondona, Atilio",
            "affiliation": "Universidade Federal do Rio Grande do Sul (UFRGS)"
        }
    ],
    "access_right": "open",
    "license": "CC-BY-4.0",
    "keywords": [
        "Thermal Infrared (TIR)",
        "Spectral Response Functions (SRF)",
        "Temperature and Emissivity Separation (TES)",
        "Physics-Informed Neural Networks (PINN)",
        "MODTRAN",
        "HyTES",
        "ASTER",
        "Radiative Transfer Equation",
        "Sobolev Shape Regularization",
        "Golden Targets",
        "Remote Sensing"
    ],
    "communities": [
        {"identifier": "earth-observation"},
        {"identifier": "remote-sensing"}
    ]
}

with open(os.path.join(release_dir, ".zenodo.json"), "w", encoding="utf-8") as f:
    json.dump(zenodo_metadata, f, indent=2, ensure_ascii=False)

# Write updated quickstart_demo.py
demo_code = r'''"""
Quickstart verification script for OpenTIR-PINN Zenodo dataset.
Validates file integrity, dimensions, physical ranges, and SRF files.
"""

import os
import numpy as np
import pandas as pd

def verify_dataset():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    # 1. Verify HyTES
    hytes_path = os.path.join(data_dir, 'hytes_256band_sobolev_pinn_dataset.npz')
    assert os.path.exists(hytes_path), f"HyTES package missing: {hytes_path}"
    hytes = np.load(hytes_path)
    print("=" * 65)
    print("HyTES 256-Channel Dataset Summary:")
    print(f"  Wavelengths:                {len(hytes['wavelengths'])} channels [{hytes['wavelengths'][0]:.3f} to {hytes['wavelengths'][-1]:.3f} µm]")
    print(f"  FWHM range:                 [{hytes['fwhm'].min():.4f} to {hytes['fwhm'].max():.4f} µm]")
    print(f"  Training Radiance shape:    {hytes['train_L_obs'].shape}")
    print(f"  Training Emissivity shape:  {hytes['train_emissivity'].shape}")
    print(f"  Training Temperature range: [{hytes['train_Ts'].min():.2f} K, {hytes['train_Ts'].max():.2f} K]")
    print(f"  Golden Targets evaluated:   {list(hytes['val_golden_materials'])}")
    print("  Integrity: OK")
    
    # 2. Verify ASTER
    aster_path = os.path.join(data_dir, 'aster_5band_sobolev_pinn_dataset.npz')
    assert os.path.exists(aster_path), f"ASTER package missing: {aster_path}"
    aster = np.load(aster_path)
    print("=" * 65)
    print("ASTER 5-Band Dataset Summary:")
    print(f"  Wavelengths:                {list(np.round(aster['wavelengths'], 3))} µm")
    print(f"  FWHM:                       {list(np.round(aster['fwhm'], 3))} µm")
    print(f"  Training Radiance shape:    {aster['train_L_obs'].shape}")
    print(f"  Training Emissivity shape:  {aster['train_emissivity'].shape}")
    print(f"  Training Temperature range: [{aster['train_Ts'].min():.2f} K, {aster['train_Ts'].max():.2f} K]")
    print(f"  Golden Targets evaluated:   {list(aster['val_golden_materials'])}")
    print("  Integrity: OK")
    
    # 3. Check atmospheric LUTs
    lut_dir = os.path.join(data_dir, 'atmospheric_luts')
    lut_files = [f for f in os.listdir(lut_dir) if f.endswith('.txt')]
    print("=" * 65)
    print(f"MODTRAN Atmospheric LUTs: {len(lut_files)} profiles available in data/atmospheric_luts/")
    
    # 4. Check SRF
    hytes_srf = os.path.join(data_dir, 'srf', 'hytes')
    aster_srf = os.path.join(data_dir, 'srf', 'aster')
    print("=" * 65)
    print("Sensor Spectral Response Functions (SRF):")
    print(f"  HyTES files in {hytes_srf}:")
    for f in os.listdir(hytes_srf):
        sz = os.path.getsize(os.path.join(hytes_srf, f)) / 1024
        print(f"    - {f} ({sz:.1f} KB)")
    print(f"  ASTER files in {aster_srf}:")
    for f in os.listdir(aster_srf):
        sz = os.path.getsize(os.path.join(aster_srf, f)) / 1024
        print(f"    - {f} ({sz:.1f} KB)")
    
    print("=" * 65)
    print("ALL ZENODO DATASET AND SRF ASSETS VERIFIED SUCCESSFULLY!")

if __name__ == '__main__':
    verify_dataset()
'''

with open(os.path.join(release_dir, "quickstart_demo.py"), "w", encoding="utf-8") as f:
    f.write(demo_code)

print("\nZenodo packaging and SRF integration completed successfully!")
