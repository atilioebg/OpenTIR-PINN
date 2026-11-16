import os

expander_code = '''        with st.expander("📖 Comprehensive Scientific Specification: The 2×2 Generalization Matrix, Quadrant Topology & Simulation Parameters", expanded=False):
            st.markdown(r"""
#### 1. The 2×2 Orthogonal Generalization Matrix & Validation Objectives
In terrestrial thermal infrared (TIR) remote sensing and GeoAI modeling, predictive architectures frequently suffer from **spectral autocorrelation** and **atmospheric confounding**—learning spurious statistical correlations between specific laboratory emissivity features and atmospheric water vapor loadings rather than inverting the underlying physical radiative transfer.

To rigorously prevent data leakage and benchmark real-world transferability, the OpenTIR-PINN benchmark organizes all forward coupled radiative observations across an **orthogonal 2×2 matrix** spanning two independent physical axes:
* **Axis 1 (Surface Material Distribution):** In-Distribution (233 development endmembers spanning 11 geological, pedological, and botanical classes) vs. Out-of-Distribution (7 structurally quarantined Golden Targets: quartz, calcite, kaolinite, feldspar, urban concrete, asphalt, and non-photosynthetic vegetation bark).
* **Axis 2 (Atmospheric Column Distribution):** In-Distribution (Nominal MODTRAN 5.2 profiles across Tropical, Mid-Latitude Summer, Sub-Arctic Summer, and 1976 US Standard) vs. Out-of-Distribution (Severe sub-arctic and polar winter regimes with extreme thermal inversions and ultra-dry columns $\mathrm{PWV} < 0.3\,\mathrm{g/cm}^2$).

```
                      [ Atmospheric Distribution Axis ]
                      In-Distribution (Nominal)     OOD (Severe Polar Winter)
                   +-----------------------------+-----------------------------+
   In-Distribution |         QUADRANT I          |        QUADRANT III         |
   (Dev Pool 233)  |  • Master Training: 5,500   |  • LOCO Generalization      |
                   |  • Out-of-Fold Val: 2,024   |  • Tab 5 Dynamic RTE Studio |
                   +-----------------------------+-----------------------------+
   Out-of-Dist.    |         QUADRANT II         |        QUADRANT IV          |
   (Golden Targets)|  • Pure Spectral OOD: 308   |  • Joint Extreme OOD: 770   |
   (7 Immutable)   |  • Zero-Shot Material Eval  |  • Double-Blind Stress Test |
                   +-----------------------------+-----------------------------+
```

---

#### 2. Quadrant-by-Quadrant Architecture & Verification Roles
* **🏋️ Quadrant I — Master Training Pool ($N = 5,500$ observations):**
  * **Role & What It Validates:** In-distribution representation learning. Validates the physics-informed neural network's capacity to minimize multi-task loss (supervised emissivity MSE, kinetic temperature error, Spectral Angle Mapper, and Sobolev gradient/curvature regularizers) without optimization collapse.
  * **Composition:** 500 spectral profiles sampled across the 233 development materials, coupled across 4 nominal MODTRAN atmospheres at 11 discrete surface kinetic temperatures ($T_s \in [280, 330]\,\mathrm{K}$ in $5\,\mathrm{K}$ steps).
* **🧪 Quadrant I — Out-of-Fold Validation ($N = 2,024$ observations):**
  * **Role & What It Validates:** Leak-free in-distribution interpolation. Validates generalization on unseen rock, soil, and vegetation specimens under standard atmospheric regimes.
  * **Composition:** 46 unseen development spectra isolated strictly in Fold 0 via `StratifiedGroupKFold(n_splits=5)` grouped by specimen `sample_id` (guaranteeing 0% intra-sample leakage across different preparation granulometries), coupled across the 4 nominal atmospheres at the 11 kinetic temperatures ($46 \times 4 \times 11 = 2,024$).
* **🏆 Quadrant II — Golden Target Benchmark ($N = 308$ observations):**
  * **Role & What It Validates:** **Zero-Shot Material Transferability.** Validates whether the network can accurately separate surface temperature $T_s$ and recover complex Reststrahlen doublets and Christiansen frequencies from completely novel materials never encountered during optimization, while operating under standard atmospheres.
  * **Composition:** The 7 permanently quarantined Golden Targets coupled across 4 nominal atmospheres at 11 temperature levels ($7 \times 4 \times 11 = 308$).
* **❄️ Quadrant III — Atmospheric Out-of-Distribution Stress Test:**
  * **Role & What It Validates:** Pure atmospheric transferability. Evaluates network resilience against extreme dry atmospheres, polar vortex conditions, and high path transmittance independently of material variations. Evaluated dynamically via Leave-One-Cluster-Out (LOCO) benchmarking and accessible interactively in Tab 5 (Dynamic RTE Studio).
* **❄️ Quadrant IV — Double-Blind Golden Benchmark ($N = 770$ observations):**
  * **Role & What It Validates:** **The Ultimate Double-Quarantine Stress Test.** Evaluates joint domain shift where *both* the ground material AND the atmospheric column are entirely out of distribution. This benchmark provides the $-79.3\%$ error reduction proof demonstrating that Sobolev-PINN fundamentally outperforms NASA's classical TES algorithm under adverse thermodynamics.
  * **Composition:** The 7 Golden Targets coupled exclusively with 10 severe sub-arctic and mid-latitude winter MODTRAN LUTs (quarantined Model 3 & Model 5 across water vapor scales $0.50\times$ to $2.00\times$, $\mathrm{PWV} \in [0.21, 1.70]\,\mathrm{g/cm}^2$) across all 11 temperatures ($7 \times 10 \times 11 = 770$).

---

#### 3. Data Synthesis Methodology & Exact Physical Values
1. **The Boundary-Value Radiative Transfer Formulation:**
   Every synthetic observation is generated via strict integration of the coupled RTE:
   $$L_{\mathrm{sensor}}(\lambda) = \tau(\lambda) \left[ \varepsilon(\lambda) B(\lambda, T_s) + (1 - \varepsilon(\lambda)) L^\downarrow(\lambda) \right] + L^\uparrow(\lambda)$$
   * $\varepsilon(\lambda)$: High-resolution pristine directional emissivity measured by laboratory FTIR spectrophotometry ($7.5\text{--}13.5\,\mu\text{m}$, $\Delta\lambda \le 1.0\,\mathrm{nm}$).
   * $B(\lambda, T_s)$: Planck blackbody function evaluated at kinetic temperature $T_s$:
     $$B(\lambda, T_s) = \frac{2hc^2}{\lambda^5 \left( \exp\left( \frac{hc}{\lambda k_B T_s} \right) - 1 \right)}$$
   * $(1 - \varepsilon(\lambda)) L^\downarrow(\lambda)$: Reflected downwelling sky radiance, adhering strictly to Kirchhoff's law for opaque terrestrial surfaces ($\rho(\lambda) = 1 - \varepsilon(\lambda)$).
   * $\tau(\lambda)$ and $L^\uparrow(\lambda)$: MODTRAN 5.2 total nadir atmospheric transmittance and upwelling thermal path radiance.
2. **Sensor Convolution:**
   * **ASTER (5 Discrete Bands):** Convolved using JPL laboratory Spectral Response Functions (SRFs): Band 10 ($8.291\,\mu\mathrm{m}$, FWHM $354\,\mathrm{nm}$), Band 11 ($8.634\,\mu\mathrm{m}$, FWHM $358\,\mathrm{nm}$), Band 12 ($9.075\,\mu\mathrm{m}$, FWHM $355\,\mathrm{nm}$), Band 13 ($10.657\,\mu\mathrm{m}$, FWHM $702\,\mathrm{nm}$), Band 14 ($11.318\,\mu\mathrm{m}$, FWHM $705\,\mathrm{nm}$).
   * **HyTES (256 Hyperspectral Channels):** Convolved across continuous contiguous channels ($7.49\text{--}12.00\,\mu\mathrm{m}$, $\Delta\lambda \approx 17.5\,\mathrm{nm}$, Gaussian instrument SRF with nominal FWHM $\approx 17.7\,\mathrm{nm}$).
3. **Kinetic Surface Temperature Grid:**
   * Range: $280.0\,\mathrm{K}$ ($6.85\,^\circ\mathrm{C}$) to $330.0\,\mathrm{K}$ ($56.85\,^\circ\mathrm{C}$) in uniform $5.0\,\mathrm{K}$ increments:
     $$\{280, 285, 290, 295, 300, 305, 310, 315, 320, 325, 330\}\,\mathrm{K}$$
   * Represents natural diurnal surface temperature variations across temperate, desert, and urban environments.
4. **Sample Accounting Summary:**
   * Forward samples per sensor: $5,500\text{ (Train)} + 2,024\text{ (Val OOF)} + 308\text{ (Q2)} + 770\text{ (Q4)} = 8,602$ observations.
   * Total benchmark corpus across both sensor architectures: $8,602 \times 2 = 17,204$ coupled radiative observations.
""")'''

target_marker = '            "Extreme Joint OOD (**Quadrant IV:** 770 samples)."\n        )\n'

target_files = [
    "zenodo_release/dataset_explorer_dashboard.py",
    "scripts/dataset_explorer_dashboard.py"
]

for fp in target_files:
    if not os.path.exists(fp):
        continue
    with open(fp, "r", encoding="utf-8") as f:
        content = f.read()

    is_crlf = "\r\n" in content
    norm_content = content.replace("\r\n", "\n")

    if target_marker in norm_content:
        replacement = target_marker + "\n" + expander_code + "\n"
        norm_content = norm_content.replace(target_marker, replacement)
        if is_crlf:
            final_content = norm_content.replace("\n", "\r\n")
        else:
            final_content = norm_content
        with open(fp, "w", encoding="utf-8") as f:
            f.write(final_content)
        print(f"Successfully injected quadrant deep specification into {fp}")
    else:
        print(f"Target marker not found in {fp}")

print("Injection complete.")

