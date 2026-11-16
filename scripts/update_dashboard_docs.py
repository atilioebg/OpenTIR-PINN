import os

def update_guides():
    new_section_4 = r'''## 4. Comprehensive Breakdown of the Seven Analytical Studio Tabs

The dashboard is structured into seven specialized analytical tabs, each addressing a critical pillar of physical remote sensing, radiative transfer physics, or machine learning validation:

### 1. 📈 Spectral Explorer (Tab 1)
* **What It Represents:** The high-resolution continuous laboratory FTIR spectral curve visualizer ($7.50\text{--}13.50\,\mu\text{m}$, 1,201 spectral channels, $\Delta\lambda = 5\,\text{nm}$) for all 241 curated benchmark samples.
* **Operational & Scientific Purpose:**
  * **Mineralogical & Vibrational Lattice Inspection:** Enables granular analysis of fundamental vibrational features, including asymmetric Si-O and Al-O stretching Reststrahlen doublets (prominent in quartz, feldspars, and kaolinite), carbonate vibrational fundamentals ($\text{CO}_3$ $\nu_3$ asymmetric stretching at $11.2\text{--}11.4\,\mu\text{m}$ in calcite), and Christiansen frequencies ($\epsilon \to 1.0$) across geological, pedological, urban, and botanical endmembers.
  * **Kirchhoff's Law Compliance & Dual Synchronized View:** Allows instantaneous toggling between Directional Emissivity ($\epsilon$), Hemispherical Reflectance ($\rho = 1 - \epsilon$), and Dual Synchronized View ($\epsilon \ \& \ \rho$). This validates thermodynamic energy conservation for opaque terrestrial solids in Local Thermodynamic Equilibrium (LTE).
  * **Sensor Response Function (SRF) Overlays:**
    * *ASTER 5-Band Filter Windows:* Overlays the 5 discrete thermal filter bands of the spaceborne ASTER instrument (B10 at $8.3\,\mu\text{m}$, B11 at $8.6\,\mu\text{m}$, B12 at $9.1\,\mu\text{m}$, B13 at $10.6\,\mu\text{m}$, and B14 at $11.3\,\mu\text{m}$).
    * *HyTES 256-Channel Spectrometer Grid:* Renders the complete contiguous array of all 256 narrow spectral channels ($7.49\text{--}11.99\,\mu\text{m}$, $\Delta\lambda \approx 17.5\,\text{nm}$) with alternating shading and 0.5px borders on the sub-layer (`layer="below"`), allowing seamless visual assessment of how hyperspectral sampling captures fine Reststrahlen curvature.
  * **Sample Telemetry & Diagnostics:** Renders interactive metadata cards displaying minimum emissivity $\epsilon_{\min}$, maximum emissivity $\epsilon_{\max}$, mean emissivity $\epsilon_{\text{mean}}$, cross-validation fold assignments (training vs. validation), and the 128-bit MD5 cryptographic hash.
  * **Unclipped Dynamic Range:** Lower y-axis limit is strictly anchored at $0.0$ (`yaxis=dict(range=[0.0, 1.02])`), eliminating clipping of deep reflectance features ($\rho < 0.40$).

---

### 2. 🏆 Golden Targets Suite (Tab 2)
* **What It Represents:** The validation and clean-room isolation suite dedicated to the 7 Golden Target Endmembers (`quartz_1`, `calcite_1`, `kaolinite_2`, `anorthite_1`, concrete `0598UUUCNC`, asphalt `0095UUUASP`, and conifer bark `VH311`), alongside the Quarantined Sibling Control (`calcite_1 solid slab`).
* **Operational & Scientific Purpose:**
  * **Strict Out-of-Distribution (OOD) Clean-Room Audit:** Provides Reviewer #2 with undeniable evidence of a strictly partitioned, zero-shot evaluation vault. These 7 endmembers are permanently withheld from training, cross-validation, and Optuna hyperparameter optimization.
  * **Feldspar Sanitation Verification:** Demonstrates the definitive resolution of the historic feldspar KBr transmission pellet artifact, showing its replacement with pure, coarse Anorthite (`anorthite_1 coarse`, $\epsilon \in [0.957, 0.989]$).
  * **Double-Quarantine Sister Isolation:** Explicitly exposes the quarantine of the solid polished slab sister (`calcite_1 solid`), proving that no specimen sharing physical origin with an OOD evaluation endmember was allowed into the training pool.
  * **Cryptographic Verification Matrix:** Displays the complete MD5 hash ledger, particle size classifications, and raw source filenames, ensuring immutable reproducibility.
  * **Unclipped Visualization:** Anchored at `[0.0, 1.02]`, fully preserving deep quartz Reststrahlen troughs ($\epsilon \approx 0.55$) without truncation.

---

### 3. 🔬 Multi-Spectrum Comparison (Tab 3)
* **What It Represents:** The statistical intra-class dispersion and population envelope analysis module across all material classes.
* **Operational & Scientific Purpose:**
  * **Live Statistical Envelope Computation:** Dynamically computes and plots:
    * The **Mean Class Profile** ($\mu_\lambda$), defining the canonical spectral signature of the selected material class.
    * The **$\pm 1\sigma$ Confidence Band** (spectral standard deviation envelope), quantifying intra-class chemical, textural, and compositional dispersion.
    * The **Total Dynamic Range Envelope** $[\min_\lambda, \max_\lambda]$, demarcating the physical extrema observed across the curated library.
  * **Diagnostic Mineralogical Contrast:** Allows direct comparison between sharp silicate Reststrahlen doublets, carbonate absorption bands, and flat vegetative continuum profiles.
  * **Training Diversity Verification:** Mathematically proves that the training distribution possesses sufficient compositional and granulometric entropy to train robust Sobolev-regularized neural networks capable of domain generalization.

---

### 4. ⚠️ 3rd Validation Gate & Forensic Audit (Tab 4)
* **What It Represents:** The forensic audit chamber dedicated to the 1,119 raw laboratory files excluded and quarantined from the original raw ECOSTRESS library.
* **Operational & Scientific Purpose:**
  * **Scientific Integrity & Anti-Cherry-Picking Defense:** Openly documents the physical criteria used to reject raw spectra (single-specimen rare minerals, non-terrestrial meteorites, and damaged interferograms), refuting any assertion of subjective selection.
  * **Detection of KBr Pellet Transmission Artifacts:** Demonstrates how laboratory measurements conducted in transmission through pressed potassium bromide (KBr) pellets violate Kirchhoff's Law for opaque solids ($\epsilon = 1 - \rho$). Volumetric scattering inside the semi-transparent pellet creates false transparency features and depresses emissivity into unphysical domains ($\epsilon < 0.40$).
  * **Side-by-Side Visual Confrontation:** Renders a direct overlay contrasting a quarantined artifact (e.g. powder in KBr) against its true physical reflectance-derived counterpart (coarse anorthite), visually demarcating the "Forbidden Region" ($\epsilon < 0.40$).

---

### 5. 🌤️ Atmospheric LUTs & Dynamic RTE Studio (Tab 5)
* **What It Represents:** The interactive coupled radiative transfer laboratory and "Component Mixer" driven by 30 high-resolution MODTRAN 5.2 Look-Up Tables (LUTs).
* **Operational & Scientific Purpose:**
  * **Coupled Radiative Transfer Simulation:** Computes the full boundary-value land-atmosphere radiance equation:
    $$L_{\mathrm{sensor}}(\lambda) = \tau(\lambda) \left[ \epsilon(\lambda) B(\lambda, T_s) + [1 - \epsilon(\lambda)] \frac{L^\downarrow(\lambda)}{\pi} \right] + L^\uparrow(\lambda)$$
  * **Dynamic Multi-Parameter Coupling:** Allows pairing any of the 241 surface materials with any of the 30 MODTRAN profiles (6 geographic standard models $\times$ 5 water vapor scaling factors, PWV $0.42\text{--}8.23\,\text{g/cm}^2$) across surface kinetic temperatures $T_s \in [270\,\text{K}, 340\,\text{K}]$.
  * **The 8-Component Mixer:** Provides modular radio toggles and individual checkboxes to isolate and analyze each physical mechanism:
    * At-sensor total radiance ($L_{\mathrm{sensor}}$);
    * Surface-leaving net radiance ($L_{\mathrm{surf}}$);
    * Direct surface thermal emission ($\tau \cdot \epsilon \cdot B$);
    * Reflected downwelling sky radiance ($\tau \cdot (1-\epsilon) \cdot L^\downarrow / \pi$);
    * Atmospheric upwelling path radiance ($L^\uparrow$);
    * Downwelling hemispherical sky irradiance ($L^\downarrow$);
    * Atmospheric transmittance ($\tau$) with $9.6\,\mu\text{m}$ ozone absorption demarcation;
    * Ideal Planck blackbody radiance ($B(\lambda, T_s)$).
  * **Photon Energy Budget Telemetry:** Calculates the exact percentage breakdown of detected photons originating from emission, reflection, and atmospheric path radiance within the critical $8.0\text{--}12.0\,\mu\text{m}$ atmospheric window, with instant CSV download capability.

---

### 6. 📦 Forward Datasets & Sample-Level Inspector (Tab 6)
* **What It Represents:** The deep-dive analytical inspector for individual synthetic forward-coupled radiative observations across ASTER (5 bands) and HyTES (256 channels).
* **Operational & Scientific Purpose:**
  * **Benchmark Partition Exploration:** Enables inspection of the Master Training Set ($N=5,000$), Out-of-Fold Validation Set ($N=2,024$), Quadrant II Golden Targets ($N=308$), and Quadrant IV Polar Winter OOD Targets ($N=770$).
  * **Sample-Level Photon Decomposition:** For any selected sample index in the ledger, decomposes detected at-sensor radiance into its surface and atmospheric components, comparing true ground emissivity ($\epsilon$) with satellite/airborne observations.
  * **Telemetry Metrics:** Dynamically reports kinetic surface temperature $T_s$, Min-Max Difference (MMD) spectral contrast ($\epsilon_{\max} - \epsilon_{\min}$), emitted photon fraction, and coupled PWV.

---

### 7. 📑 Metadata Catalog & FAIR Table (Tab 7)
* **What It Represents:** The master inventory of all 241 physical benchmark spectra, structured in strict compliance with FAIR (Findable, Accessible, Interoperable, Reusable) data principles.
* **Operational & Scientific Purpose:**
  * **Searchable & Filterable Inventory:** Real-time search engine querying across sample IDs, mineralogical names, particle sizes, benchmark partitions, and source filenames.
  * **Provenance & Integrity Verification:** Full ledger displaying emissivity extrema ($\epsilon_{\min}, \epsilon_{\max}, \epsilon_{\text{mean}}$), cross-validation fold assignments, and 128-bit MD5 hashes.
  * **Open Data Export:** One-click download button exporting `opentir_spectral_catalog_filtered.csv` for independent analysis in Python, R, or MATLAB.'''

    for path in ["zenodo_release/DASHBOARD_USER_GUIDE.md", "docs/DASHBOARD_USER_GUIDE.md"]:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        is_crlf = "\r\n" in content
        norm = content.replace("\r\n", "\n")

        start_marker = "## 4. Comprehensive Breakdown of the Seven Analytical Tabs"
        end_marker = "## 5. Scientific & Editorial FAQ (Reviewer #2 Alignment)"

        if start_marker in norm and end_marker in norm:
            prefix = norm.split(start_marker)[0]
            suffix = norm.split(end_marker)[1]
            updated_norm = prefix + new_section_4 + "\n\n---\n\n" + end_marker + suffix
            updated_content = updated_norm.replace("\n", "\r\n") if is_crlf else updated_norm
            with open(path, "w", encoding="utf-8") as f:
                f.write(updated_content)
            print(f"Successfully updated {path}")
        else:
            print(f"Markers not found in {path}")

def update_clarification():
    path = "clarification.md"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    is_crlf = "\r\n" in content
    norm = content.replace("\r\n", "\n")

    toc_marker = "30. [The Five Top-Level KPI Pillars of the OpenTIR-PINN Benchmark: Rigorous Architectural & Physical Breakdown](#30-the-five-top-level-kpi-pillars-of-the-opentir-pinn-benchmark-rigorous-architectural--physical-breakdown)"
    toc_addition = toc_marker + "\n31. [The Seven Analytical Studio Modules of OpenTIR-PINN Dashboard: Functional Architecture & Physical Telemetry](#31-the-seven-analytical-studio-modules-of-opentir-pinn-dashboard-functional-architecture--physical-telemetry)"

    entry_31 = r'''

---

## 31. The Seven Analytical Studio Modules of OpenTIR-PINN Dashboard: Functional Architecture & Physical Telemetry
* **Date Added:** 2026-11-05
* **Category:** *Dashboard Architecture / Radiative Transfer Forensics / Reviewer #2 Defense / FAIR Open Science*
* **Related Code & Artifacts:** [`zenodo_release/dataset_explorer_dashboard.py`](file:///c:/Users/Atilio/Desktop/PROJETOS/AITIR/zenodo_release/dataset_explorer_dashboard.py), [`zenodo_release/DASHBOARD_USER_GUIDE.md`](file:///c:/Users/Atilio/Desktop/PROJETOS/AITIR/zenodo_release/DASHBOARD_USER_GUIDE.md)

### ❓ Inquired Question
> "Explain in detail what each of the seven analytical tabs of the dashboard is, what it represents physically and computationally, and what operational purpose it serves: 1. Spectral Explorer; 2. Golden Targets Suite; 3. Multi-Spectrum Comparison; 4. 3rd Validation Gate & Forensic Audit; 5. Atmospheric LUTs & Dynamic RTE Studio; 6. Forward Datasets & Sample-Level Inspector; 7. Metadata Catalog & FAIR Table."

---

### 💡 Executive Summary
The OpenTIR-PINN Dashboard (`dataset_explorer_dashboard.py`) is structured into seven specialized analytical modules designed to refute the "Synthetic Black Box" objection commonly raised by Reviewer #2 in top-tier journals (*IEEE TGRS, RSE, ISPRS*). Together, the seven tabs cover the entire radiative and computational workflow: from raw laboratory FTIR interferograms ($7.50\text{--}13.50\,\mu\text{m}$, 1,201 channels) and the forensic exclusion of transmission pellet artifacts, through strict zero-shot OOD clean-room benchmarks and statistical class envelopes, to interactive MODTRAN 5.2 forward photon coupling, sample-level forward observation inspection, and FAIR-compliant metadata export.

---

### 🔬 Full Theoretical & Editorial Explanation

#### 1. 📈 Spectral Explorer (Tab 1)
* **Physical Scope:** Continuous laboratory FTIR directional emissivity spectra (1,201 channels, $\Delta\lambda = 5\,\text{nm}$) for all 241 curated samples.
* **Core Capabilities:**
  * *Vibrational Lattice Spectroscopy:* Interactive diagnostic inspection of asymmetric Si-O and Al-O stretching Reststrahlen doublets, carbonate vibrational fundamentals ($\text{CO}_3$ $\nu_3$ at $11.2\text{--}11.4\,\mu\text{m}$), and Christiansen frequencies ($\epsilon \to 1.0$).
  * *Kirchhoff Law Dual View:* Instantaneous toggle between Directional Emissivity ($\epsilon$), Hemispherical Reflectance ($\rho = 1 - \epsilon$), and Dual Synchronized View ($\epsilon \ \& \ \rho$), verifying thermodynamic energy conservation for opaque solids.
  * *Sensor Response Overlays (SRF):* Displays the 5 thermal bands of spaceborne ASTER ($8.125\text{--}11.65\,\mu\text{m}$) and the complete contiguous 256 narrow channel grid of airborne HyTES ($7.49\text{--}11.99\,\mu\text{m}$, $\Delta\lambda \approx 17.5\,\text{nm}$) with alternating shading and 0.5px borders on the sub-layer (`layer="below"`).
  * *Unclipped Y-Axis:* Anchored strictly at $[0.0, 1.02]$ to prevent truncation of deep reflectance curves ($\rho < 0.40$).

#### 2. 🏆 Golden Targets Suite (Tab 2)
* **Physical Scope:** The 7 immutable Out-of-Distribution (OOD) benchmark endmembers (`quartz_1`, `calcite_1`, `kaolinite_2`, `anorthite_1`, concrete `0598UUUCNC`, asphalt `0095UUUASP`, and conifer bark `VH311`) plus the Quarantined Sibling Control (`calcite_1 solid slab`).
* **Core Capabilities:**
  * *Clean-Room OOD Audit:* Proves that evaluation targets are permanently quarantined from the training pool and Optuna hyperparameter optimization.
  * *Feldspar Sanitation Verification:* Visual proof of the replacement of the KBr pellet transmission artifact with coarse, pure Anorthite (`anorthite_1 coarse`, $\epsilon \in [0.957, 0.989]$).
  * *Double-Quarantine Sister Isolation:* Documents the complete exclusion of the solid polished slab sister (`calcite_1 solid`), eliminating intra-sample crystallographic leakage.

#### 3. 🔬 Multi-Spectrum Comparison (Tab 3)
* **Physical Scope:** Population-level spectral envelopes across all 10 material classes.
* **Core Capabilities:**
  * *Live Statistical Metrics:* Computes mean spectral signature $\mu_\lambda$, standard deviation confidence band $\pm 1\sigma_\lambda$, and total dynamic range $[\min_\lambda, \max_\lambda]$.
  * *Variance Verification:* Demonstrates that the training distribution possesses sufficient compositional, grain-size, and roughness variance for robust physical generalization in PINN.

#### 4. ⚠️ 3rd Validation Gate & Forensic Audit (Tab 4)
* **Physical Scope:** The 1,119 raw laboratory spectra excluded from the benchmark.
* **Core Capabilities:**
  * *Anti-Cherry-Picking Defense:* Documents physical rejection criteria (rare singletons, non-terrestrial meteorites, KBr pellet transmission matrices).
  * *Side-by-Side Confrontation:* Direct visual overlay comparing unphysical KBr pellet transmission artifacts ($\epsilon < 0.40$, volumetric scattering) against true physical reflectance benchmarks (coarse anorthite).

#### 5. 🌤️ Atmospheric LUTs & Dynamic RTE Studio (Tab 5)
* **Physical Scope:** Interactive Forward Radiative Transfer laboratory driven by 30 MODTRAN 5.2 profiles.
* **Core Capabilities:**
  * *RTE Numerical Engine:* Solves $L_{\mathrm{sensor}}(\lambda) = \tau(\lambda) [\epsilon(\lambda) B(\lambda, T_s) + (1-\epsilon(\lambda)) L^\downarrow(\lambda)/\pi] + L^\uparrow(\lambda)$.
  * *The 8-Component Mixer:* Modular toggling of individual components ($L_{\mathrm{sensor}}, L_{\mathrm{surf}}, L_{\mathrm{emit}}, L_{\mathrm{refl}}, L^\uparrow, L^\downarrow, \tau, B$).
  * *Energy Budget Telemetry:* Percentage breakdown of detected photons in the $8.0\text{--}12.0\,\mu\text{m}$ atmospheric window with one-click CSV export.

#### 6. 📦 Forward Datasets & Sample-Level Inspector (Tab 6)
* **Physical Scope:** Deep-dive inspection of individual synthetic observations across ASTER and HyTES datasets.
* **Core Capabilities:**
  * *Partition Ledger:* Explores Master Training ($N=5,000$), Out-of-Fold Validation ($N=2,024$), Quadrant II ($N=308$), and Quadrant IV ($N=770$).
  * *Photon Decomposition:* Decomposes sensor radiance and contrasts it against true ground emissivity ($\epsilon$), computing MMD spectral contrast and emitted photon fraction.

#### 7. 📑 Metadata Catalog & FAIR Table (Tab 7)
* **Physical Scope:** Complete 241-spectrum master inventory in FAIR data standard.
* **Core Capabilities:**
  * *Full-Text Filtering:* Queries across sample IDs, mineral names, particle sizes, and cross-validation partitions.
  * *Cryptographic Provenance:* Full table displaying 128-bit MD5 hashes, emissivity extrema, and one-click CSV export (`opentir_spectral_catalog_filtered.csv`).

---

### 📌 Impact on Manuscript & Implementation
* **Manuscript Placement:** Section III.D (*Forensic Audit & Interactive Exploration Dashboard*).
* **Reviewer #2 Shield:** Directly counters claims that synthetic data generation is an unverified black box by providing an open, interactive analytical studio where reviewers can independently audit every calculation.
'''

    if toc_marker in norm:
        norm = norm.replace(toc_marker, toc_addition)
    
    norm = norm + entry_31
    updated_content = norm.replace("\n", "\r\n") if is_crlf else norm
    with open(path, "w", encoding="utf-8") as f:
        f.write(updated_content)
    print(f"Successfully updated {path}")

if __name__ == "__main__":
    update_guides()
    update_clarification()
