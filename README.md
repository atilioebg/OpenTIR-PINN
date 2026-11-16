# OpenTIR-PINN: Physics-Informed Neural Networks with Sobolev Shape Regularization for Thermal Infrared Temperature and Emissivity Separation

[![Target: Remote Sensing of Environment](https://img.shields.io/badge/Target-Remote%20Sensing%20of%20Environment-00629B.svg)](https://www.sciencedirect.com/journal/remote-sensing-of-environment)
[![Zenodo DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24085242-blue.svg)](https://doi.org/10.5281/zenodo.24085242)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Official Code & Benchmark Repository for the Research Manuscript:**  
> **"Physics-Informed Neural Networks with Sobolev Shape Regularization for Thermal Infrared Temperature and Emissivity Separation Across Multispectral and Hyperspectral Sensors"**  
> **Authors:** Atilio Grondona¹⁴*, Fernanda Zunkowski Grondona¹, Bijeesh Kozhikkodan Veettil², Luis Morales-Salinas³  
> ¹ *Research and Development Department, AgroRisco AI, Porto Alegre, RS 91760-630, Brazil*  
> ² *Department of Environmental Technology and Management, Faculty of Applied Technology, Van Lang University, Ho Chi Minh City, Vietnam*  
> ³ *Laboratory for Research in Environmental Sciences (LARES), Faculty of Agricultural Sciences, University of Chile, Santiago, Chile*  
> ⁴ *Federal University of Rio Grande do Sul (UFRGS), Porto Alegre, Brazil*  
> \* *Corresponding Author: atilio.grondona@ufrgs.br*

---

## 📌 Executive Summary

Retrieving Land Surface Temperature (LST) and Land Surface Emissivity (LSE) from thermal infrared (TIR) radiances is a classically **ill-posed inverse problem**: an instrument observing in $N$ discrete spectral channels measures $N$ radiometric equations but faces $N + 1$ physical unknowns (one kinetic temperature $T_s$ and $N$ channel-integrated directional emissivities $\varepsilon_i$).

Operational heuristic algorithms, notably the **Temperature and Emissivity Separation (TES)** algorithm (Gillespie et al., 1998; Hulley et al., 2016), resolve this Planck degeneracy using an empirical power-law regression linking minimum spectral emissivity to spectral contrast (Min–Max Difference, MMD). While computationally expedient, this empirical heuristic:
1. **Amplifies Channel Noise:** Stochastically inflates spectral contrast on flat natural emitters, triggering severe high-frequency sawtooth oscillations across hundreds of narrow hyperspectral channels (up to 18-fold noise amplification).
2. **Compounds Atmospheric Errors:** Propagates precipitable water vapor (PWV) compensation errors directly into spurious surface temperature and emissivity biases.
3. **Breaks Down on Anomalous Mineralogies:** Overfits to typical rock and soil spectra, failing on materials exhibiting deep Reststrahlen doublets (e.g., quartz and kaolinite).

**OpenTIR-PINN** replaces empirical contrast regressions with a physics-informed deep learning framework that embeds fundamental radiative transfer and functional spectral smoothness directly into neural optimization:
* **Bounded Thermodynamic Output Heads:** Employs shifted Softplus ($T_s > 200$\,K) and channel-wise Sigmoid ($\varepsilon \in (0, 1)$) activation heads, ensuring physical energy conservation by construction.
* **Discrete Sobolev Regularization:** Penalizes first-order spectral slopes ($W^{1,1}$) and second-order curvatures ($W^{2,1}$) weighted by inter-channel wavelength spacing, suppressing high-frequency noise while preserving diagnostic mineral absorption doublets.
* **Forward Planck Radiative Consistency:** Evaluates Kirchhoff's surface radiative transfer equation in brightness-temperature space during backpropagation, enforcing thermodynamic consistency as an explicit convergence criterion.
* **Dual-Sensor Regime Parity:** Evaluated across both underdetermined multispectral sensors (**ASTER**, 5 TIR bands) and high-dimensional hyperspectral spectrometers (**HyTES**, 223 active TIR channels).

---

## 📊 The OpenTIR-PINN Benchmark Suite (v1.3.0)

The accompanying benchmark dataset is archived and citable on Zenodo under [DOI: 10.5281/zenodo.24085242](https://doi.org/10.5281/zenodo.24085242).

| Benchmark Dimension | Specification | Grounding Reference |
|---|---|---|
| **Curated Spectral Library** | 240 high-resolution laboratory/field spectra across 11 material classes | ECOSTRESS / USGS Spectral Libraries |
| **Similarity Grouping** | 162 disjoint spectral similarity groups (preventing split leakage) | `splits/split_v1.1.json` |
| **Quarantined Golden Targets** | 7 pristine endmembers (Quartz, Kaolinite, Calcite, Feldspar, Concrete, Asphalt, Conifer Bark) | Sequestered outside all training folds |
| **Atmospheric Radiative Transfer** | 30 MODTRAN 5.2 DISORT 8-stream profiles (PWV: $0.35$ to $5.40\,\mathrm{g\,cm^{-2}}$) | Tropical, Mid-Lat, Sub-Arctic, US Standard |
| **Quadrature Formulation** | 8-node Gauss–Legendre hemispheric downwelling integration ($L^{\downarrow}$) | Converged to within 0.08% of 16-node reference |
| **Sensors Evaluated** | ASTER (5 bands, $8.125\text{--}11.65\,\mu\mathrm{m}$) and HyTES (223 channels, $7.50\text{--}12.00\,\mu\mathrm{m}$) | Full SRF convolution and opaque-band masking |
| **Evaluation Matrix** | $2 \times 2$ Double-Quarantine Matrix (Q-I In-Domain, Q-II Material OOD, Q-III Atmospheric OOD, Q-IV Joint Novelty) | 36,652 unique physical evaluation samples |
| **Verification Gate** | 32-test automated audit suite (416 individual checks: 402 PASS, 1 WARN, 0 FAIL, 13 INFO) | `scripts/validate_dataset_integrity.py` |

---

## 🚀 Key Experimental Findings

1. **Hyperspectral Decisive Advantage (HyTES):** Relying solely on surface-leaving radiance ($\hat{\mathbf{L}}_{\mathrm{surf}}$) without auxiliary atmospheric telemetry, the PINN reduces LST root-mean-square error by **44.0%** (1.186 vs. 2.119 K), cuts channel-averaged emissivity error by **68.4%**, halves water-vapor error propagation, and eliminates the 18-fold high-frequency noise amplification of classical TES.
2. **Multispectral Operational Parity (ASTER):** Five broad bands lack the mathematical degrees of freedom to decouple surface emission from atmospheric attenuation without auxiliary telemetry ($\hat{\bm{\tau}}$, $\hat{\mathbf{L}}^{\downarrow}$). Under strict input parity, the PINN achieves operational parity in temperature (1.528 vs. 1.781 K) while reconstructing emissivity shapes with superior angular fidelity (SAM error reduced by 56.8%).
3. **Failure Boundary Transparency:** On anomalous mineralogies exhibiting extreme absorption depths beyond the training manifold (exemplified by kaolinite), the neural shrinkage prior induces cold temperature biases ($-4$ to $-12$\,K). Model ensemble spread captures these out-of-domain targets on HyTES ($77\%\text{--}100\%$ coverage within $2\sigma$), but fails on ASTER, establishing the operational necessity of radiance-derived novelty filters.
4. **Computational Ground-Segment Scalability:** Batched inference processes $>51,000$ pixels/s on ASTER and $>23,000$ pixels/s on HyTES on a single laptop CPU core (scaling to $>153,000$ and $>81,000$ pixels/s with 10 threads), establishing that physics-informed neural retrieval is computationally ready for satellite ground segments without requiring GPU hardware.

---

## 🖥️ Interactive Dataset Explorer Dashboard

The repository includes a standalone interactive Streamlit web dashboard for visual dataset inspection, radiative transfer auditing, and evaluation matrix exploration:

```bash
# Launch the interactive web dashboard
streamlit run scripts/dataset_explorer_dashboard.py
```

### Dashboard Capabilities:
* **Planck Inversion & Forward Radiative Transfer:** Interactively simulate surface-leaving radiance across temperatures (275–335 K) and atmospheric columns.
* **Sensor Geometry & Spectral Response:** Inspect channel center wavelengths, SRF convolution curves, and opaque water-vapor band masks for ASTER and HyTES.
* **4-Quadrant Stress Matrix Navigator:** Interactively filter and compare Q-I (in-distribution), Q-II (quarantined golden targets), Q-III (winter atmospheres), and Q-IV (joint stress) sample distributions.
* **Spectral Similarity Clustering:** Audit the 162 group-disjoint clusters and verify zero data leakage across 5-fold splits.
* **Automated Audit Suite Telemetry:** Review the 32 automated verification test results, acceptance criteria, and contingency tables.

---

## 📦 Quickstart & Installation

### 1. Environment Setup

Clone the repository and install the production dependencies:

```bash
git clone https://github.com/atilioebg/OpenTIR-PINN.git
cd OpenTIR-PINN

# Install required packages
pip install -r requirements.txt
```

### 2. Verify Dataset Integrity (32-Test Audit)

Execute the automated verification suite to validate all 416 physical, numerical, and split-integrity checks:

```bash
python scripts/validate_dataset_integrity.py
```

### 3. Run Pipeline or Training Benchmarks

To execute training, cross-validation, or evaluation:

```bash
# Execute the cross-validation training workflow
python src/training/orchestrate_training_pipeline.py

# Run leave-one-class-out (LOCO) generalization benchmarks
python src/training/train_loco_suite.py
```

---

## 📂 Repository Architecture

```text
AITIR/
├── data/                    # Open Source Benchmark specifications & Single Source of Truth
│   ├── dataset_facts.json   # Machine-readable SSOT manifest (v1.3.0)
│   └── DATA_DICTIONARY.md   # Formal dataset feature schema & sensor specifications
├── docs/                    # User documentation & pipeline guides
│   ├── DASHBOARD_USER_GUIDE.md # Comprehensive guide for the Streamlit dashboard
│   ├── PIPELINE_README.md      # Data synthesis & forward radiative transfer documentation
│   └── dataset_validation_protocol.md # Specification of the 32 audit test families
├── scripts/                 # Reproducibility, evaluation, and dashboard runners
│   ├── dataset_explorer_dashboard.py # Interactive Streamlit exploration application
│   ├── validate_dataset_integrity.py # 32-test automated audit suite
│   ├── build_dataset_facts.py        # SSOT manifest generator
│   └── run_pipeline.py               # End-to-end data pipeline orchestrator
├── splits/                  # Group-disjoint 5-fold cross-validation & LOCO split manifests
├── src/                     # Core deep learning & physical modeling library
│   ├── baselines/           # Classical TES implementation (ASTER & HyTES ATBD)
│   ├── data_generation/     # Forward radiative transfer synthesizers & splitters
│   ├── evaluation/          # Paired bootstrap inference & statistical testing
│   ├── models/              # Decoupled PINN architectures & Sobolev/Planck losses
│   └── training/            # Cross-validation, Optuna Bayesian HPO, & LOCO suites
├── tests/                   # Automated unit & integration tests
├── requirements.txt         # Minimal production Python dependencies
└── README.md                # Project documentation & benchmark overview
```

---

## 📜 Citation & Attribution

If you use the OpenTIR-PINN benchmark dataset, neural architectures, or evaluation code in your research, please cite:

### Dataset Citation (Zenodo)
```bibtex
@dataset{grondona2026opentir,
  author       = {Grondona, Atilio},
  title        = {{OpenTIR-PINN: Physics-Informed Neural Network Benchmark for Thermal Infrared Temperature-Emissivity Separation (v1.3.0)}},
  month        = nov,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.3.0},
  doi          = {10.5281/zenodo.24085242},
  url          = {https://doi.org/10.5281/zenodo.24085242}
}
```

### Manuscript Citation (Target: Remote Sensing of Environment)
```bibtex
@article{grondona2026physics,
  title   = {Physics-informed neural networks with {Sobolev} shape regularization for thermal infrared temperature and emissivity separation across multispectral and hyperspectral sensors},
  author  = {Grondona, Atilio and Grondona, Fernanda Zunkowski and Veettil, Bijeesh Kozhikkodan and Morales-Salinas, Luis},
  journal = {Remote Sensing of Environment},
  year    = {2026},
  note    = {Under Review}
}
```

---

## 📄 License

This project is licensed under the **MIT License**; see the [LICENSE](LICENSE) file for details.
