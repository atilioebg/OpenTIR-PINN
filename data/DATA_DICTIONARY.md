# Data Dictionary: OpenTIR-PINN Benchmark (v1.3.0)

**Associated Manuscript:** *Physics-Informed Neural Networks with Sobolev Shape Regularization for Ill-Posed Temperature and Emissivity Separation Across Multispectral and Hyperspectral Thermal Infrared Sensors* (IEEE Transactions on Geoscience and Remote Sensing)  
**Author:** Dr. Atilio Grondona (*Universidade Federal do Rio Grande do Sul - UFRGS*)  
**Version:** 1.3.0 (Single Source of Truth Frozen Release)  
**License:** CC-BY 4.0 International  
**SSoT Manifest Hash:** `data/dataset_facts.json` (Schema v1.0)

---

## 1. Directory Topology

```
zenodo_release/
├── .zenodo.json                               # Zenodo DOI deposit schema
├── README.md                                  # FAIR overview, verification report, and quickstart guide
├── DATA_DICTIONARY.md                         # Authoritative field-by-field tensor and tabular specification
├── quickstart_demo.py                         # Standalone python verification and inference script
├── physics_engine.py                          # Vectorized, differentiable forward RTE operator
├── tests/
│   └── test_physics_engine.py                 # PyTorch autograd and Planck round-trip unit tests
└── data/
    ├── processed/
    │   └── final_datasets/                    # Serialized PyTorch tensor partitions (.pt)
    │       ├── train_fold0_aster.pt / hytes.pt
    │       ├── val_fold0_aster.pt / hytes.pt
    │       ├── golden_benchmark_quadrant_II_aster.pt / hytes.pt
    │       ├── golden_benchmark_quadrant_IV_aster.pt / hytes.pt
    │       └── ... (5 cross-validation folds)
    ├── golden_targets_metadata.csv            # Physical, mineralogical, and chemical metadata for the 7 Golden Targets
    ├── atmospheric_luts/                      # 30 MODTRAN 5.2 profile look-up tables
    │   ├── lut_catalog.csv                    # Catalog of the 30 profiles (models, PWV, saturation cap)
    │   ├── modtran_simulation_parameters_matrix.csv # Parameter sweep matrix
    │   └── profile_atm*.csv / .npz            # Transmittance, path radiance, and hemispheric downwelling radiance
    ├── splits/                                # Reproducibility splits and manifests
    │   ├── split_v1.1.json                    # 5-Fold cross-validation manifest (162 similarity groups)
    │   └── golden_targets.json                # Immutable golden target definitions & sibling quarantine
    ├── benchmarks/                            # Verification scorecards and best Optuna configs
    │   ├── best_config_aster.json             # Converged hyperparameter architecture (ASTER 384x5 SiLU)
    │   ├── best_config_hytes.json             # Converged hyperparameter architecture (HyTES 768x3 SiLU)
    │   ├── cv_benchmark_aster.csv / hytes.csv # 5-Fold cross-validation scorecards
    │   ├── loss_decomposition_master.csv      # Epoch-by-epoch loss trajectories across 100 epochs
    │   └── quarantine_matrix_benchmark_*.csv  # Double-Quarantine vs. Classical TES benchmarks
    ├── raw_laboratory_spectra/                # Raw continuous laboratory FTIR spectra
    │   ├── golden_targets/                    # 7 individual FTIR text files (Pre-code continuous spectra)
    │   ├── raw_spectral_library_inventory.csv # Inventory of all 1,205 continuous spectra
    │   └── all_raw_laboratory_spectra_1205.zip # Compressed master raw library (1,205 files)
    └── srf/                                   # Sensor Spectral Response Functions
        ├── aster/                             # ASTER Bands 10–14 SRFs and specification tables
        └── hytes/                             # HyTES 256-channel SRFs, WMX, and specification tables
```

---

## 2. Partition Accounting & The 2×2 Double-Quarantine Matrix

The benchmark organizes all forward coupled radiative observations across an orthogonal **2×2 Double-Quarantine Matrix** to rigorously test generalization across unseen materials and unseen weather regimes without data leakage:

```
                      [ Atmospheric Weather Regime ]
                      In-Distribution (Nominal Dev)   OOD (Severe Polar Winter / Sub-Arctic)
                   +-------------------------------+-------------------------------+
   In-Distribution |          QUADRANT I           |         QUADRANT III          |
   (231 Dev Pool)  |  • Fold 0 Train: 5,460        |  • Fold 0 Q-III: 5,390        |
                   |  • Fold 0 Val:   2,156        |  • All Folds Union: 25,410    |
                   |  • 5-Fold OOF:  10,164        |  • Systematic Atmospheric OOD |
                   +-------------------------------+-------------------------------+
   Out-of-Dist.    |          QUADRANT II          |          QUADRANT IV          |
   (7 Golden Pool) |  • Pure Spectral OOD: 308     |  • Joint Extreme OOD: 770     |
   (Quarantined)   |  • 7 Targets × 22 Atm × 2 Ts  |  • 7 Targets × 8 Atm × 11 Ts  |
                   +-------------------------------+-------------------------------+
```

### Canonical Partition Sample Counts (Fold 0 Baseline)
* **🏋️ Quadrant I (Training, Fold 0):** **5,460 samples** (182 training spectra × 30 realizations per spectrum; continuous $T_s \sim \mathcal{U}(275, 335)\text{ K}$; sensor noise multiplier $\mathcal{U}(0.5, 1.5)$; Model N4 PWV error $\sim \mathcal{U}(1.84, 4.1)\text{ mm}$).
* **🧪 Quadrant I (Validation Out-Of-Fold, Fold 0):** **2,156 samples** (49 held-out validation spectra). Across all 5 folds, the complete development pool comprises **7,616 samples** (231 unique spectra) with each spectrum validated exactly once (**10,164 Out-Of-Fold evaluation samples**).
* **🏆 Quadrant II (Golden Targets Material OOD):** **308 samples** (7 pristine golden targets × 22 development atmospheres × 2 test temperatures).
* **❄️ Quadrant III (Atmospheric Inversion OOD, Fold 0):** **5,390 samples** (49 dev spectra × 8 winter atmospheres × 11 temperature levels). The full 5-fold union comprises **25,410 samples**.
* **⚡ Quadrant IV (Joint Extreme OOD):** **770 samples** (7 golden targets × 8 polar winter atmospheres × 11 test temperatures: $280\text{--}330\text{ K}$ in $5\text{ K}$ increments).
* **Total Canonical Samples per Sensor (Fold 0):** **14,084 samples** (28,168 combined across ASTER and HyTES).

---

## 3. Data Schema for PyTorch Tensor Partitions (`.pt`)

Each `.pt` file serializes a list of dictionaries containing individual forward radiative observations:

| Field Name | Type | Shape | Units | Physical Description |
| :--- | :--- | :--- | :--- | :--- |
| `input_radiance` | `torch.FloatTensor` | `(B,)` | $\text{W}/(\text{m}^2\cdot\text{sr}\cdot\mu\text{m})$ | Perturbed surface-leaving radiance $L_{surf,in}(\lambda)$ after inverse atmospheric correction with N4 error. |
| `downwelling_radiance`| `torch.FloatTensor` | `(B,)` | $\text{W}/(\text{m}^2\cdot\text{sr}\cdot\mu\text{m})$ | Hemispheric sky irradiance divided by $\pi$ ($L_\downarrow = F_\downarrow/\pi$, 8-node Gauss-Legendre quadrature). |
| `transmittance` | `torch.FloatTensor` | `(B,)` | $[0, 1]$ | Perturbed atmospheric path transmittance $\tau_{in}(\lambda)$ interpolated with N4 PWV error. |
| `path_radiance` | `torch.FloatTensor` | `(B,)` | $\text{W}/(\text{m}^2\cdot\text{sr}\cdot\mu\text{m})$ | Upwelling thermal path radiance $L^\uparrow_{in}(\lambda)$ interpolated with N4 PWV error. |
| `target_temp` | `torch.FloatTensor` | `(1,)` | $\text{Kelvin (K)}$ | Ground-truth surface kinetic temperature $T_s$. |
| `target_emissivity` | `torch.FloatTensor` | `(B,)` | $[0, 1]$ | Ground-truth pristine laboratory surface spectral emissivity convolved by sensor SRF. |
| `spectrum_id` | `str` | scalar | — | Unique spectral library identifier (e.g., `SPEC_0015_quartz`). |
| `sample_id` | `str` | scalar | — | Laboratory sample code (e.g., `quartz_1`, `VH311`, `0598UUUCNC`). |
| `material_class` | `str` | scalar | — | High-level taxonomic category (`botanical`, `soil`, `sandstone`, `quartz`, `basalt`, etc.). |
| `atmosphere_id` | `str` | scalar | — | Profile identifier from MODTRAN 5.2 catalog (e.g., `atm1_h2o1.00`). |
| `pwv` | `float` | scalar | $\text{g}/\text{cm}^2$ | True atmospheric Precipitable Water Vapor column. |
| `pwv_est` | `float` | scalar | $\text{g}/\text{cm}^2$ | Perturbed PWV estimate contaminated by retrieval error (Model N4). |
| `pwv_est_clipped` | `bool` | scalar | — | Flag indicating whether perturbed PWV hit MODTRAN interpolation boundary. |
| `is_augmented` | `bool` | scalar | — | Training vs. validation indicator. Emissivity is strictly pristine across all partitions. |

---

## 4. Physical Perturbation & Noise Modeling

The OpenTIR-PINN benchmark v1.3.0 implements a physically grounded perturbation hierarchy:

### A. Radiometric Sensor Noise ($NE\Delta T$)
Independent zero-mean Gaussian detector noise is added to the at-sensor radiance:
$$\sigma_L(\lambda) = \left. \frac{\partial B(\lambda, T)}{\partial T} \right|_{T = 300\text{ K}} \times NE\Delta T \times \text{noise\_multiplier}$$
* **ASTER:** $NE\Delta T = 0.3\text{ K}$ at $300\text{ K}$ across all 5 bands.
* **HyTES:** $NE\Delta T = 0.2\text{ K}$ at $300\text{ K}$ across all 256 channels (223 unmasked active bands).
* **Multipliers:** Fixed at $1.0\times$ during evaluation; randomly sampled from $\mathcal{U}(0.5, 1.5)$ during training.

### B. Realistic Atmospheric-Correction Error (Model N4)
Atmospheric water vapor column estimation uncertainty is explicitly modeled:
$$\text{PWV}_{est} = \text{PWV} + \delta, \quad \delta \sim \mathcal{N}(0, s^2)$$
* **Training:** $s \sim \mathcal{U}(1.84, 4.1)\text{ mm}$.
* **Evaluation:** $s = 2.97\text{ mm}$ (fixed standard error).
* Atmospheric parameters $(\tau, L^\uparrow, L_\downarrow)$ are dynamically retrieved via quadratic Lagrange interpolation across the MODTRAN 5.2 look-up tables.

### C. Emissivity Integrity (Zero Label Corruption)
* **Pristine Signatures:** Emissivity micro-perturbations are completely disabled ($\text{eps\_scale} = [1.0, 1.0], \text{eps\_jitter\_sigma} = 0.0$). True laboratory Reststrahlen features and crystal lattice absorption bands are strictly preserved.

---

## 5. The 7 Immutable Golden Targets

Quarantined permanently in the clean-room vault to serve as zero-shot material generalization benchmarks:

| Target ID | Mineral / Material Name | Chemical Formula / Sample Code | $\varepsilon_{min}$ | $\varepsilon_{max}$ | Mean $\varepsilon$ | Diagnostic Reststrahlen Feature |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `urban_concrete` | Construction  Concrete | `0598UUUCNC` | 0.8465 | 0.9768 | 0.9284 | High broadband construction continuum |
| `urban_asphalt` | Construction Asphalt | `0095UUUASP` | 0.9324 | 0.9799 | 0.9624 | Smooth low-contrast hydrocarbon continuum |
| `calcite` | Calcite CaCO3 | `calcite_1` | 0.8986 | 0.9679 | 0.9490 | Carbonate in-plane bending $\nu_3, \nu_1, \nu_2$ modes |
| `kaolinite` | Kaolinite Al2Si2O5(OH)4 | `kaolinite_2` | 0.4727 | 0.9794 | 0.8037 | Deep phyllosilicate Al-OH doublet (min at $8.79\,\mu\text{m}$) |
| `feldspar` | Anorthite CaAl2Si2O8 | `anorthite_1` | 0.9567 | 0.9969 | 0.9743 | Tectosilicate plagioclase asymmetric wing |
| `quartz` | Quartz SiO2 | `quartz_1` | 0.8828 | 0.9945 | 0.9594 | Sharp quartz Reststrahlen doublet ($8.3\,\mu\text{m}$ & $9.3\,\mu\text{m}$) |
| `npv_bark` | Abies concolor bark | `VH311` | 0.9539 | 0.9778 | 0.9661 | Non-photosynthetic vegetation lignin/cellulose |

---

## 6. Atmospheric Look-Up Tables (`atmospheric_luts/`)

Generated via MODTRAN 5.2 with DISORT 8-stream radiative transfer, rural aerosol ($5\text{ km}$ visibility), $\text{CO}_2 = 300\text{ ppmv}$, and surface reflectance $0.06$:
* **Development Profiles (22 profiles):** Tropical, Midlatitude Summer, Subarctic Summer, and 1976 US Standard models across 5 water scalings ($0.50\times$ to $2.00\times$). PWV: $0.7082$ to $6.7669\text{ g/cm}^2$.
* **Winter / Polar OOD Profiles (10 profiles):** Midlatitude Winter and Subarctic Winter models across 5 water scalings. PWV: $0.2081$ to $1.3724\text{ g/cm}^2$.
* **Hemispheric Sky Radiance:** Calculated via 8-node Gauss-Legendre quadrature integration ($L_\downarrow = F_\downarrow / \pi$).

---

## 7. Raw Laboratory Spectra Schema (`data/raw_laboratory_spectra/`)

Archives the source laboratory FTIR measurements before code processing or SRF convolution:
* **`golden_targets/`:** Individual continuous laboratory spectra (~2,150 channels, $1.25\text{--}15.0\,\mu\text{m}$).
* **`raw_spectral_library_inventory.csv`:** Metadata catalog of all 1,205 candidate raw continuous laboratory spectra.
* **`all_raw_laboratory_spectra_1205.zip`:** Master compressed ZIP archive ($11.73\text{ MB}$) for sensor cross-convolution.
