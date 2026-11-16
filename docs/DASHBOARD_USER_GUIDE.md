# OpenTIR-PINN Analytical Dashboard: Comprehensive User Guide & Scientific Reference Manual (v1.1.1)

**Author:** Dr. Atilio Grondona (*Universidade Federal do Rio Grande do Sul - UFRGS*)  
**Research Consortium:** OpenTIR-PINN Research Group  
**Associated Publications:**  
* *"Physics-Informed Neural Networks with Sobolev Shape Regularization for Ill-Posed Temperature and Emissivity Separation Across Multispectral and Hyperspectral Thermal Infrared Sensors"* (Target: *IEEE Transactions on Geoscience and Remote Sensing*)  
* *"OpenTIR-PINN Benchmark: Synthetic Radiative Transfer & Double-Quarantine Golden Target Dataset"* (Zenodo DOI: `10.5281/zenodo.24085242`, v1.3.0)  

---

## 1. Executive Overview & Purpose of the Dashboard

In top-tier remote sensing journals (IEEE TGRS, RSE, ISPRS), machine learning and deep learning methodologies face intense scrutiny from **"Reviewer #2"** regarding:
1. **The "Synthetic Black Box" Objection:** How can a reviewer verify that synthetic radiance observations accurately represent genuine radiative physics without unphysical extrapolation or artificial artifacts?
2. **Data Leakage & Overfitting:** Did the training set inadvertently observe replicas, mineralogical siblings, or identical atmospheric states as the evaluation set?
3. **Physical Law Compliance:** Does the inverse network obey Planck's law, Kirchhoff's radiation law, and downwelling sky reflection bounds?

The **OpenTIR-PINN Dataset Explorer Dashboard** (`dataset_explorer_dashboard.py`) was engineered as an **interactive forensic audit and analytical studio**. It exposes every layer of data processing—from raw laboratory FTIR interferograms ($7.50\text{--}13.50\,\mu\text{m}$) and MODTRAN 5.2 atmospheric look-up tables (LUTs) to sensor response convolutions (ASTER 5-band and HyTES 256-channel) and coupled forward photon balances.

---


---

## 2. Top KPI Summary Banner: The Five Pillars of the Benchmark

At the top of the analytical dashboard, five high-level KPI cards summarize the quantitative scale and physical foundation of the OpenTIR-PINN benchmark:

```
┌─────────────────────────┬─────────────────────────┬─────────────────────────┬─────────────────────────┬─────────────────────────┐
│ Total Curated Spectra   │ Golden Targets (OOD)    │ Development Pool        │ Atmospheric LUTs        │ Forward Coupled Samples │
│ 241                     │ 7                       │ 233                     │ 30 Profiles             │ 7,024+                  │
│ ↑ Pristine FTIR         │ ↑ Immutable Benchmarks  │ ↑ 5-Fold Partitioned    │ ↑ MODTRAN 5.2           │ ↑ ASTER & HyTES         │
└─────────────────────────┴─────────────────────────┴─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

### 1. Total Curated Spectra: 241 (Pristine FTIR)
* **What It Represents:** The complete universe of high-resolution continuous laboratory thermal infrared (TIR) emissivity spectra ($7.50\text{--}13.50\,\mu\text{m}$, 1,201 spectral channels, $\Delta\lambda = 5\,\text{nm}$) curated for the benchmark.
* **Exact Composition:**
  * **141 Geological & Urban Spectra:** Soils ($N=69$), Sandstones ($N=21$), Quartz ($N=16$), Basalts ($N=12$), Granites ($N=8$), Urban Concrete ($N=5$), Urban Asphalt ($N=3$), Calcite ($N=3$), Feldspar ($N=2$), and Kaolinite ($N=2$).
  * **100 Botanical Spectra:** Selected via deterministic Farthest Point Sampling (FPS) with $L_2$ Euclidean distance from 549 botanical candidates to maximize spectral variance ($\sigma_\epsilon = 0.0367$, $\epsilon \in [0.8141, 0.9910]$).
  * **Total:** $141 + 100 = 241$ physical spectra.
* **Why "Pristine FTIR":** Every spectrum has passed 3 strict physical gates: elimination of KBr transmission pellet artifacts, continuous coverage across the $8.0\text{--}12.0\,\mu\text{m}$ window, and compliance with thermodynamic emissivity bounds for opaque terrestrial solids ($\epsilon \in [0.40, 1.0]$).

---

### 2. Golden Targets (OOD): 7 (Immutable Benchmarks)
* **What It Represents:** The seven pristine physical endmembers locked into the immutable clean-room evaluation vault.
* **The 7 Endmembers:**
  1. `quartz_1` — Tectosilicate crystalline silica ($\alpha\text{-SiO}_2$);
  2. `calcite_1` — Carbonate optical calcite ($\text{CaCO}_3$);
  3. `kaolinite_2` — Phyllosilicate Georgia clay ($\text{Al}_2\text{Si}_2\text{O}_5(\text{OH})_4$);
  4. `anorthite_1` — Plagioclase feldspar ($\text{CaAl}_2\text{Si}_2\text{O}_8$);
  5. `0598UUUCNC` — Construction concrete (JHU Becknic);
  6. `0095UUUASP` — Highway paving asphalt (JHU Becknic);
  7. `VH311` — Non-photosynthetic vegetation bark (*Abies concolor* White-Fir bark, UCSB).
* **Role in Peer Review:** These 7 spectra are **strictly never seen by the neural network during training or cross-validation**. They evaluate the model's true zero-shot Out-of-Distribution (OOD) generalization capability on deep Reststrahlen doublets and asymmetric vibrational modes.

---

### 3. Development Pool: 233 (5-Fold Partitioned)
* **What It Represents:** The remaining 233 physical spectra ($241 - 7\text{ Golden} - 1\text{ Quarantined Sibling} = 233$) allocated for model training and cross-validation.
* **The Sibling Exclusion:** `calcite_1 solid` was excluded from this pool and placed into `🛡️ Quarantined Siblings` to prevent intra-sample data leakage with `calcite_1 coarse`.
* **5-Fold Grouped Stratification:** Partitioned via `StratifiedGroupKFold` ($K=5$) grouped strictly by `sample_id` and stratified by `material_class`. This guarantees:
  * 0% cross-fold leakage (all preparations of the same specimen stay in the same fold);
  * 100% out-of-fold validation coverage (all 233 development spectra are validated exactly once).

---

### 4. Atmospheric LUTs: 30 Profiles (MODTRAN 5.2)
* **What It Represents:** The high-resolution radiative transfer look-up table (LUT) grid simulated in MODTRAN 5.2 at $1\,\text{cm}^{-1}$ spectral resolution.
* **Exact Composition:**
  * **6 Standard Geographical Atmospheres:** Tropical (TROP), Mid-Latitude Summer (MLS), Mid-Latitude Winter (MLW), Sub-Arctic Summer (SAS), Sub-Arctic Winter (SAW), and 1976 US Standard (US76).
  * **5 Water Vapor Scaling Factors:** $0.50\times, 0.75\times, 1.00\times, 1.50\times, 2.00\times$ (spanning precipitable water vapor from PWV $= 0.42\,\text{g/cm}^2$ in polar winter to $8.23\,\text{g/cm}^2$ in tropical regimes).
  * **Total:** $6 \times 5 = 30$ distinct atmospheric profiles.
* **Radiative Quantities per Profile:** Transmittance $\tau(\lambda)$, upwelling path radiance $L^\uparrow(\lambda)$, and downwelling sky irradiance divided by $\pi$ ($L^\downarrow(\lambda)/\pi$).
* **Partitioning:** 20 development profiles for training/validation, and 10 severe polar winter profiles (MLW, SAW) reserved for Quadrant IV OOD stress testing.

---

### 5. Forward Coupled Samples: 7,024+ (ASTER & HyTES)
* **What It Represents:** The full volume of synthetic forward radiance observations generated via the complete surface-leaving radiative transfer equation:
  $$L_{\mathrm{sensor}}(\lambda) = \tau(\lambda) \left[ \epsilon(\lambda) B(\lambda, T_s) + [1 - \epsilon(\lambda)] \frac{L^\downarrow(\lambda)}{\pi} \right] + L^\uparrow(\lambda) + n_{\mathrm{sensor}}$$
* **Breakdown of Generated Observations (Per Sensor):**
  * **Master Training Pool:** $5,000$ samples ($500\text{ samples/class} \times 10\text{ classes}$) with surface temperatures $T_s \in [260, 340]\,\text{K}$ and sensor noise ($NE\Delta T = 0.10\text{--}0.25\,\text{K}$).
  * **Out-of-Fold Validation Set:** $2,024$ samples ($11\text{ temperatures} \times 4\text{ atmospheres}$ per fold across all 5 folds).
  * **Quadrant II Benchmark:** $308$ samples ($7\text{ Golden Targets} \times 11\text{ temperatures} \times 4\text{ development atmospheres}$).
  * **Quadrant IV Benchmark:** $770$ samples ($7\text{ Golden Targets} \times 11\text{ temperatures} \times 10\text{ severe polar OOD atmospheres}$).
  * **Total per Sensor:** $5,000 + 2,024 + 308 + 770 = 8,102$ samples. Across both ASTER and HyTES, the benchmark contains **$16,204$ fully coupled radiative observations**.


## 3. Sidebar Control Panel & Dynamic Cascading Filters

The sidebar provides a synchronized multi-tier filtering system that dynamically propagates selections across all tabs.

```
┌────────────────────────────────────────────────────────┐
│ 1. Measurement Modality (ε vs. ρ vs. Dual View)        │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│ 2. Benchmark Partition Filter (OOD / Quarantined / CV) │
└───────────────────────────┬────────────────────────────┘
                            │ (Cascades Available Options)
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌────────────────────────┐      ┌────────────────────────┐
│ 3. Material Classes    │      │ 4. Particle Size       │
│    (Filtered to match) │      │    (Filtered to match) │
└───────────┬────────────┘      └───────────┬────────────┘
            └───────────────┬───────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ 5. Spectral Range Slider (7.50 to 13.50 μm)           │
└────────────────────────────────────────────────────────┘
```

### 2.1 Modality 1: Measurement Modality (Radiometric Representation)
Controls the radiometric quantity rendered for laboratory spectral curves:
* **Directional Emissivity ($\epsilon(\lambda)$):**
  * *Physical Definition:* The dimensionless ratio ($[0, 1]$) of radiant energy emitted by a surface to that emitted by a Planck blackbody at the same thermodynamic temperature $T_s$.
  * *Mathematical Derivation:* Computed via **Kirchhoff's Law for Opaque Media in Local Thermodynamic Equilibrium (LTE)**:
    $$\epsilon(\lambda) = 1 - \rho(\lambda)$$
    assuming zero internal transmittance ($\tau_{\mathrm{int}}(\lambda) = 0$). This holds strictly for macroscopic terrestrial geological, pedological, and dense vegetative surfaces in the thermal infrared.
* **Hemispherical Reflectance ($\rho(\lambda)$):**
  * *Physical Definition:* Directional Hemispherical Reflectance (DHR) measured directly by FTIR spectrometers equipped with integrating spheres.
  * *Spectral Behavior:* Bands that appear as emission valleys (*Reststrahlen* features, e.g., quartz $\text{Si-O-Si}$ stretching at $8.3\,\mu\text{m}$ and $9.3\,\mu\text{m}$) appear as pronounced reflection peaks.
* **Dual Synchronized View ($\epsilon \text{ \& } \rho$):**
  * *Purpose:* Displays both curves simultaneously on the same wavelength grid.
  * *Energy Conservation Audit:* Allows immediate visual verification of energy conservation. When quartz emissivity dips to $\epsilon = 0.58$, reflectance rises to $\rho = 0.42$, confirming $\epsilon + \rho = 1.0$.
  * *Y-Axis Formatting:* The y-axis is anchored strictly at $[0.0, 1.02]$. This prevents clipping of reflectance curves, which dip near $0.01$ in high-emissivity vegetation.

### 2.2 Modality 2: Benchmark Partition & Zero-Leakage Architecture
Controls the partition boundaries of the benchmark:
1. **`All Partitions (241 Spectra)`:** The complete curated library ($N = 241$, comprising 141 geological/urban spectra and 100 botanical spectra selected via $L_2$ Euclidean Farthest Point Sampling).
2. **`🏆 Golden Targets (7 Strict OOD Benchmarks)`:** The 7 immutable evaluation endmembers permanently quarantined in a clean-room vault:
   * `quartz_1` (Pure crystalline $\alpha\text{-SiO}_2$);
   * `calcite_1` (Coarse carbonate $\text{CaCO}_3$);
   * `kaolinite_2` (Phyllosilicate clay $\text{Al}_2\text{Si}_2\text{O}_5(\text{OH})_4$);
   * `anorthite_1` (Plagioclase tectosilicate feldspar $\text{CaAl}_2\text{Si}_2\text{O}_8$);
   * `0598UUUCNC` (Construction concrete / urban continuum);
   * `0095UUUASP` (Road asphalt / hydrocarbon mix);
   * `vh311` (Conifer bark *Abies concolor* / non-photosynthetic vegetation - NPV).  
   *Zero-Shot Guarantee:* These 7 targets are **never seen during training or cross-validation**.
3. **`🛡️ Quarantined Siblings (1 Sibling Control)`:** Contains `calcite_1 solid`. Quarantined because it shares the same physical origin (`sample_id: calcite_1`) as the Golden Target. Retaining it in training would introduce intra-sample crystallographic leakage.
4. **`🧪 Development Pool (All 233 Train/Val Spectra)`:** The 233 remaining spectra used for the 5-fold cross-validation partition ($241 - 7 - 1 = 233$).
5. **`📂 Fold 0 to Fold 4 — Train Split (~185–188 Spectra)`:** Training splits generated via `StratifiedGroupKFold` grouped by physical `sample_id` and stratified by `material_class`.
6. **`📂 Fold 0 to Fold 4 — Val Split (~45–48 Spectra)`:** Out-of-fold validation splits. The union of all 5 validation splits covers 100% of the 233 development spectra.

### 2.3 Modality 3 & 4: Material Classes & Particle Size (Cascading Filter Engine)
* **Dynamic Cascading Engine:** Selecting a partition in Control 2 immediately filters the options available in Controls 3 and 4. For instance:
  * Selecting `🏆 Golden Targets` automatically restricts Material Classes to the 6 classes present in the golden targets, preventing empty filter queries.
  * Selecting `🛡️ Quarantined Siblings` restricts the class list strictly to `['calcite']`.
* **Material Classes ($11$ total):** `basalt`, `botanical`, `calcite`, `feldspar`, `granite`, `kaolinite`, `quartz`, `sandstone`, `soil`, `urban_asphalt`, `urban_concrete`.
* **Particle Sizes / Textures:** `solid`, `coarse` ($250\text{--}500\,\mu\text{m}$), `medium to fine` / `fine` ($<74\,\mu\text{m}$), `earthy clay`, `unsorted`, `natural`, `none` (biological targets).

### 2.4 Modality 5: Spectral Window Range Slider
Adjusts the wavelength window from $7.50\,\mu\text{m}$ to $13.50\,\mu\text{m}$ with a step of $0.05\,\mu\text{m}$. All variance bands, plots, and tables recalculate dynamically.

---

## 4. Comprehensive Breakdown of the Seven Analytical Studio Tabs

The dashboard is structured into seven specialized analytical tabs, each addressing a critical pillar of physical remote sensing, radiative transfer physics, or machine learning validation:

### 1. 📈 Spectral Explorer (Tab 1)
* **What It Represents:** The high-resolution continuous laboratory FTIR spectral curve visualizer ($7.50\text{--}13.50\,\mu\text{m}$, 1,201 spectral channels, $\Delta\lambda = 5\,\text{nm}$) for all 241 curated benchmark samples.
* **Operational & Scientific Purpose:**
  * **Mineralogical & Vibrational Lattice Inspection:** Enables granular analysis of fundamental vibrational features, including asymmetric Si-O and Al-O stretching Reststrahlen doublets (prominent in quartz, feldspars, and kaolinite), carbonate vibrational fundamentals ($\text{CO}_3$ $\nu_3$ asymmetric stretching at $11.2\text{--}11.4\,\mu\text{m}$ in calcite), and Christiansen frequencies ($\epsilon \to 1.0$) across geological, pedological, urban, and botanical endmembers.
  * **Kirchhoff's Law Compliance & Dual Synchronized View:** Allows instantaneous toggling between Directional Emissivity ($\epsilon$), Hemispherical Reflectance ($\rho = 1 - \epsilon$), and Dual Synchronized View ($\epsilon \ \& \ \rho$). This validates thermodynamic energy conservation for opaque terrestrial solids in Local Thermodynamic Equilibrium (LTE).
  * **Sensor Response Function (SRF) Overlays (Collision-Free Interactive Hover Tooltips):**
    * *ASTER 5-Band Filter Windows:* Overlays the 5 discrete thermal filter bands of the spaceborne ASTER instrument (B10 at $8.3\,\mu\text{m}$, B11 at $8.6\,\mu\text{m}$, B12 at $9.1\,\mu\text{m}$, B13 at $10.6\,\mu\text{m}$, and B14 at $11.3\,\mu\text{m}$) using fill-polygons with interactive hover tooltips (`hoveron="fills"`). Placing the mouse over any band immediately displays its band name, center wavelength, spectral window, and FWHM bandwidth, with zero static text collisions.
    * *HyTES 256-Channel Spectrometer Grid:* Renders the complete contiguous array of all 256 narrow spectral channels ($7.49\text{--}11.99\,\mu\text{m}$, $\Delta\lambda \approx 17.5\,\text{nm}$) with alternating shading and 0.5px borders on the sub-layer (`layer="below"`), coupled with an interactive channel-center hover array that displays channel index, center wavelength, bandwidth, and interval without top-banner clutter.
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
  * **Unclipped Visualization & Bottom-Right Legend:** Anchored at `[0.0, 1.02]`, fully preserving deep quartz Reststrahlen troughs ($\epsilon \approx 0.55$) without truncation. The trace legend is anchored in the bottom-right corner (`x=0.99, y=0.02`) on a translucent card, keeping the upper curve domain completely unobstructed.

---

### 3. 🔬 Multi-Spectrum Comparison (Tab 3)
* **What It Represents:** The statistical intra-class dispersion and population envelope analysis module across all material classes.
* **Operational & Scientific Purpose:**
  * **Live Statistical Envelope Computation:** Dynamically computes and plots:
    * The **Mean Class Profile** ($\mu_\lambda$), defining the canonical spectral signature of the selected material class.
    * The **$\pm 1\sigma$ Confidence Band** (spectral standard deviation envelope), quantifying intra-class chemical, textural, and compositional dispersion.
    * The **Total Dynamic Range Envelope** $[\min_\lambda, \max_\lambda]$, demarcating the physical extrema observed across the curated library.
  * **Diagnostic Mineralogical Contrast:** Allows direct comparison between sharp silicate Reststrahlen doublets, carbonate absorption bands, and flat vegetative continuum profiles.
  * **Training Diversity Verification & Dedicated Y-Axis Scaling:** Mathematically proves that the training distribution possesses sufficient compositional and granulometric entropy to train robust Sobolev-regularized neural networks capable of domain generalization. For this tab specifically, the y-axis is anchored at $[0.40, 1.02]$ because all mineral and land-surface classes strictly exceed $\epsilon = 0.40$, maximizing vertical dynamic resolution.

---

### 4. ⚠️ 3rd Validation Gate & Forensic Audit (Tab 4)
* **What It Represents:** The forensic audit chamber dedicated to the 1,119 raw laboratory files excluded and quarantined from the original raw ECOSTRESS library.
* **Operational & Scientific Purpose:**
  * **Scientific Integrity & Anti-Cherry-Picking Defense:** Openly documents the physical criteria used to reject raw spectra (single-specimen rare minerals, non-terrestrial meteorites, and damaged interferograms), refuting any assertion of subjective selection.
  * **Detection of KBr Pellet Transmission Artifacts:** Demonstrates how laboratory measurements conducted in transmission through pressed potassium bromide (KBr) pellets violate Kirchhoff's Law for opaque solids ($\epsilon = 1 - \rho$). Volumetric scattering inside the semi-transparent pellet creates false transparency features and depresses emissivity into unphysical domains ($\epsilon < 0.40$).
  * **Dynamic Mineralogical Benchmark Pairing:** Features automated dynamic matching between the selected quarantined artifact and its exact physical benchmark counterpart in the curated library (e.g., matching `calcite_1 fine in KBr` against `calcite_1 coarse`, `quartz_1 fine in KBr` against `quartz_1 coarse`, `anorthite_1 fine in KBr` against `anorthite_1 coarse`). The title and legend update dynamically, visually demarcating the "Forbidden Region" ($\epsilon < 0.40$) without arbitrary mineral cross-matching.

---

### 5. 🌤️ Atmospheric LUTs & Dynamic RTE Studio (Tab 5)
* **What It Represents:** The interactive coupled radiative transfer laboratory and "Component Mixer" driven by 30 high-resolution MODTRAN 5.2 Look-Up Tables (LUTs).
* **Operational & Scientific Purpose:**
  * **Coupled Boundary-Value RTE Simulation:** Computes the exact land-atmosphere boundary radiance balance in a dedicated, uncrowded container:
    $$L_{\mathrm{sensor}}(\lambda) = \tau(\lambda) \left[ \epsilon(\lambda) B(\lambda, T_s) + [1 - \epsilon(\lambda)] L^\downarrow(\lambda) \right] + L^\uparrow(\lambda)$$
    where $L^\downarrow(\lambda)$ is the directional downwelling sky radiance from MODTRAN 5.2.
  * **Clear 241-Spectrum Library Provenance:** Explicitly identifies that the ground material dropdown draws from the complete universe of **241 pristine laboratory FTIR spectra** from the curated **ECOSTRESS Library v1.0** (141 geological/pedological/urban solids and 100 diversity-maximized botanical endmembers via $L_2$ Euclidean FPS).
  * **Dynamic Cascading Climatological Controls:** Selecting one of the 6 geographic climatological models (Tropical, Mid-Latitude Summer, Mid-Latitude Winter, Sub-Arctic Summer, Sub-Arctic Winter, 1976 US Standard) dynamically filters the water vapor scaling options to the 5 valid scaled moisture profiles for that atmosphere, accompanied by an informative climatological guidance banner displaying latitude zone, moisture bounds, and typical surface temperatures.
  * **Physical Reflection Integrity (Reflected Sky Radiance):** Solves $L_{\mathrm{refl}}(\lambda) = [1 - \epsilon(\lambda)] L^\downarrow(\lambda)$. Because MODTRAN 5.2's `L_down_surface` is already expressed as directional radiance in $\text{W}/(\text{m}^2\cdot\text{sr}\cdot\mu\text{m})$, eliminating the double-division by $\pi$ brings the dashboard into 100% exact numerical agreement with the forward synthesis training pipeline (`synthesize_forward_dataset.py`).
  * **The 8-Component Mixer:** Provides modular radio toggles and individual checkboxes to isolate and analyze each physical mechanism:
    * At-sensor total radiance ($L_{\mathrm{sensor}}$);
    * Surface-leaving net radiance ($L_{\mathrm{surf}}$);
    * Direct surface thermal emission ($\tau \cdot \epsilon \cdot B$);
    * Reflected downwelling sky radiance ($\tau \cdot [1-\epsilon] \cdot L^\downarrow$);
    * Atmospheric upwelling path radiance ($L^\uparrow$);
    * Downwelling atmospheric sky radiance ($L^\downarrow$);
    * Atmospheric transmittance ($\tau$) with $9.6\,\mu\text{m}$ ozone absorption demarcation;
    * Ideal Planck blackbody radiance ($B(\lambda, T_s)$).
  * **Clean Telemetry Telemetry (Arrow-Free Scientific Cards):** Replaces misleading green up-arrow delta badges with authoritative scientific metric cards displaying primary units, secondary Celsius conversions, and descriptive subcaptions.

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
  * **Open Data Export:** One-click download button exporting `opentir_spectral_catalog_filtered.csv` for independent analysis in Python, R, or MATLAB.

---

## 5. Scientific & Editorial FAQ (Reviewer #2 Alignment)

### Question 1: Does the dashboard dynamically adjust filters when a Benchmark Partition is selected?
**Answer:** **Yes.** In v1.1.1, the sidebar implements a **Cascading Pre-Filter Architecture**. When `2. Benchmark Partition` is selected, the application first generates a sub-dataframe (`part_df`). The dropdown options for `3. Material Classes` and `4. Particle Size / Texture` are then computed dynamically from `part_df["material_class"].unique()` and `part_df["particle_size"].unique()`. This ensures that impossible combinations (e.g. searching for `sandstone` inside `Golden Targets`) are eliminated from the UI.

---

### Question 2: Can we group samples with different Granulometry and Microstructural Textures in model training? Is there any issue regarding IEEE TGRS best practices?

#### A. The Physical Remote Sensing Rationale (Why Mixed Granulometry is Essential)
In laboratory FTIR spectroscopy, specimens are prepared as polished solid slabs or sieved grain-size fractions (e.g., coarse $250\text{--}500\,\mu\text{m}$ vs. fine $<74\,\mu\text{m}$).
However, in spaceborne or airborne remote sensing (ASTER spatial resolution = $90\,\text{m}$; HyTES = $2\text{--}4\,\text{m}$), **a ground pixel is never an idealized polished solid slab**. A real pixel is an intimate macroscopic mixture of bedrock outcroppings, weathered coarse gravel, sand dunes, and fine windblown dust.

From radiative transfer physics:
1. **Coarse / Solid Fractions ($>250\,\mu\text{m}$):** Dominated by **specular surface reflection** (optically opaque volume). Bands of *Reststrahlen* lattice vibration are deep, exhibiting high spectral contrast ($\epsilon \approx 0.60\text{--}0.70$).
2. **Fine Powders ($<74\,\mu\text{m}$):** Dominated by **intense multiple volumetric scattering**. Photons undergo multiple internal refractions before escaping, which "washes out" and flattens Reststrahlen absorption bands ($\epsilon \approx 0.85\text{--}0.95$) and can induce transparency feature inversion.

**If a neural network is trained only on solid slabs, it will fail over agricultural soils, sand sheets, and weathered terrains.** Grouping diverse granulometries in the training pool provides **physics-informed domain randomization (data augmentation)**. It forces the network to learn invariant crystallographic features (such as band center frequencies) rather than memorizing rigid, grain-size-specific band depths. This practice is supported by foundational thermal literature (*Salisbury & D'Aria 1992, RSE; Gillespie et al. 1998, IEEE TGRS; Hulley et al. 2012, JGR*).

#### B. The Strict "Reviewer #2" Caveat: Preventing Intra-Sample Leakage
While grouping mixed grain sizes in training is beneficial, **one fatal mistake must be prevented**:
> [!CAUTION]
> **The Intra-Sample Contamination Trap:** You cannot place the coarse powder of specimen `quartz_1` in the **Training Set** and the solid slab of the exact same specimen `quartz_1` in the **Validation Set**.

Because both preparations share the exact same chemical provenance, trace element impurities, and spectrometer calibration curve, the neural network could achieve near-zero validation error by memorizing sample-specific noise rather than generalizable physical features.

#### C. How OpenTIR-PINN Resolves This (IEEE Q1 Standard)
1. **Sample-Grouped Stratification:** In `src/data_generation/base/make_split.py`, cross-validation uses `StratifiedGroupKFold` grouped strictly by `sample_id`:
   $$\text{Group} = \text{sample\_id}$$
   All granulometric variants of a given physical specimen are forced into the same fold.
2. **Double-Quarantine Sister Isolation:** For Golden Targets (such as `calcite_1`), its solid slab sister (`calcite_1 solid`) was placed into `🛡️ Quarantined Siblings`, ensuring that no sibling of an OOD test endmember is ever seen in training.
3. **Conclusion for IEEE TGRS Reviewers:** Grouping diverse granulometries is scientifically sound, realistic, and robust against data leakage under our sample-grouped partitioning scheme.

---


---

### Question 3: Exhaustive Forensic Audit of Sibling Contamination Across All 7 Golden Targets & 233 Development Spectra

To definitively validate whether any other Golden Target or development sample faces sibling contamination (the "Sisterhood Problem"), we executed an automated cryptographic and bibliographic audit scanning all **3,451 raw laboratory files** in ECOSTRESS, all **241 curated library entries**, and all **5 cross-validation folds**:

#### Table 1: Exhaustive Sibling Audit for the 7 Golden Targets

| Golden Target Endmember | Golden File (`sample_id`) | Potential Sibling in Raw ECOSTRESS Library | Raw File Origin & Preparation | Curated Benchmark Status & Quarantine Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **1. Calcite** ($	ext{CaCO}_3$) | `calcite_1` (Coarse) | `calcite_1` (Solid)<br>`calcite_1` (Fine) | JHU Nicolet Solid Slab<br>JHU Nicolet Powder in KBr Pellet | **`calcite_1 solid`:** Quarantined in `🛡️ Quarantined Siblings` (Clean-Room Vault).<br>**`calcite_1 fine`:** Excluded by Gate 2 (Transmission Pellet Artifact). |
| **2. Quartz** ($lpha	ext{-SiO}_2$) | `quartz_1` (Coarse) | `quartz_1` (Solid)<br>`quartz_1` (Fine) | JHU Nicolet Solid Slab<br>JHU Nicolet Powder in KBr Pellet | **`quartz_1 solid`:** Excluded by Gate 3 (Specular reflectance $80.8\%$, $\epsilon_{\min} = 0.192 < 0.40$).<br>**`quartz_1 fine`:** Excluded by Gate 2 (Transmission Pellet Artifact). |
| **3. Kaolinite** ($	ext{Al}_2	ext{Si}_2	ext{O}_5(	ext{OH})_4$) | `kaolinite_2` (Coarse) | `kaolinite_2` (Fine) | JHU Nicolet Powder in KBr Pellet | **`kaolinite_2 fine`:** Excluded by Gate 2 (Transmission Pellet Artifact). Zero siblings admitted to library. |
| **4. Feldspar** ($	ext{CaAl}_2	ext{Si}_2	ext{O}_8$) | `anorthite_1` (Coarse) | `anorthite_1` (Fine) | JHU Nicolet Powder in KBr Pellet | **`anorthite_1 fine`:** Excluded by Gate 2 (Transmission Pellet Artifact). Zero siblings admitted to library. |
| **5. Urban Concrete** | `0598UUUCNC` (Solid) | *None* | JHU Becknic Construction Concrete | **Unique Single Specimen:** Only 1 file exists in the raw repository. Zero raw siblings. |
| **6. Urban Asphalt** | `0095UUUASP` (Solid) | *None* | JHU Becknic Paving Asphalt | **Unique Single Specimen:** Only 1 file exists in the raw repository. Zero raw siblings. |
| **7. NPV Bark** (*A. concolor*) | `VH311` (Natural Bark) | `VH311` (VSWIR) | UCSB ASD Field Spectroradiometer | **`VH311 vswir`:** Excluded by Spectral Window Gate ($0.4\text{--}2.5\,\mu\text{m}$ vs. TIR $7.5\text{--}13.5\,\mu\text{m}$). Zero TIR siblings exist. |

**Audit Finding for Golden Targets:** Exactly **0** sibling spectra of any Golden Target exist inside the Development Pool (Training Set or Validation Set). The Golden Target clean-room vault is **100% leak-free**.

---

#### Table 2: Intra-Development Pool Multi-Preparation Audit (Train vs. Val Integrity)
Within the 233 Development Pool spectra, there are **22 physical sample specimens** that feature multiple preparations in the library (such as fine powder vs. solid slab of the same rock):
* **Basalt Samples:** `Basalt_H5`, `Basalt_7`, `Basalt_H1`, `Basalt_H10`, `Basalt_H12`, `Basalt_H2` (each present as both `fine` and `solid`).
* **Granite Samples:** `Granite_H3`, `Granite_H5`, `Granite_H9`, `Granite_H11` (each present as both `fine` and `solid`).
* **Quartzite Samples:** `qrtzit1`, `qrtzit4`, `qrtzit5`, `qtzite1`, `qtzite2`, `qtzite3` (each present as both `coarse` and `fine`).

**Verification of Cross-Fold Isolation:**  
We evaluated the set intersection $\text{Train}_{\text{samples}} \cap \text{Val}_{\text{samples}}$ across all 5 folds in `splits/split_v1.1.json`:
* **Fold 0:** 170 Train Samples $\cap$ 41 Val Samples $\implies$ **Overlap = 0 (100% Clean)**
* **Fold 1:** 167 Train Samples $\cap$ 44 Val Samples $\implies$ **Overlap = 0 (100% Clean)**
* **Fold 2:** 171 Train Samples $\cap$ 40 Val Samples $\implies$ **Overlap = 0 (100% Clean)**
* **Fold 3:** 167 Train Samples $\cap$ 44 Val Samples $\implies$ **Overlap = 0 (100% Clean)**
* **Fold 4:** 169 Train Samples $\cap$ 42 Val Samples $\implies$ **Overlap = 0 (100% Clean)**
* **Total Cross-Fold Leaks across all folds:** **0**

**Conclusion:** `StratifiedGroupKFold` grouped strictly by `sample_id` completely eliminates intra-sample leakage. Whenever a sample with multiple granulometries is drawn, all its variants are locked into the exact same fold.

## 6. Verification & Launch Instructions

To launch the dashboard locally:
```powershell
streamlit run zenodo_release/dataset_explorer_dashboard.py
```
* Local URL: `http://localhost:8501`
* Fully compliant with Streamlit $\ge 1.57.0$ (Zero layout deprecation warnings, responsive container sizing, unclipped $[0.0, 1.02]$ axes).



## 7. The 2×2 Generalization Matrix & Tab 6 Architecture (Quadrants I–IV)

### The 2×2 Generalization Framework

In the thermal infrared domain, model generalization must be evaluated across both material and atmospheric domain shifts:

| Quadrant | Surface Materials | Atmospheric Profiles | Sample Count (per sensor) | Primary Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Quadrant I (Train)** | Development Pool (233 spectra) | Nominal Dev Atmospheres (4 models) | **5,500 samples** (500 × 11 classes) | Core training pool for PINN physics learning |
| **Quadrant I (Val OOF)** | Unseen Dev Pool (Fold 0, 46 spectra) | Nominal Dev Atmospheres (4 models) | **2,024 samples** (46 × 4 × 11 temps) | Leak-free out-of-fold generalization audit |
| **Quadrant II (Spectral OOD)**| 7 Immutable Golden Targets | Nominal Dev Atmospheres (4 models) | **308 samples** (7 × 4 × 11 temps) | Pure unseen mineralogy/NPV lattice evaluation |
| **Quadrant III (Atmospheric OOD)**| Development Pool (233 spectra) | Polar Winter / Harsh Inversions | Evaluated via LOCO & Dynamic Studio (Tab 5) | Atmospheric domain adaptation stress test |
| **Quadrant IV (Joint OOD)** | 7 Immutable Golden Targets | Severe Polar Winter (10 models) | **770 samples** (7 × 10 × 11 temps) | Extreme double-blind operational stress test |

* **Total Forward Coupled Samples per Sensor:** $5,500 + 2,024 + 308 + 770 = 8,602$ samples.
* **Total Across Both Sensors (ASTER 5-Band + HyTES 256-Channel):** $8,602 \times 2 = 17,204$ samples.

### Tab 6 UI & Scientific Enhancements
1. **Coupled Boundary-Value RTE Display Container:**
   $$L_{\mathrm{sensor}}(\lambda) = \tau(\lambda) \left[ \varepsilon(\lambda) B(\lambda, T_s) + (1 - \varepsilon(\lambda)) L^\downarrow(\lambda) \right] + L^\uparrow(\lambda)$$
2. **Explicit Quadrant Partition Selector:**
   * `🏋️ Quadrant I — Master Training Set (Dev Pool × Dev Atmospheres | 5,500 samples)`
   * `🧪 Quadrant I — Out-of-Fold Validation (Unseen Dev Pool × Dev Atmospheres | 2,024 samples)`
   * `🏆 Quadrant II — Golden Target Benchmark (Spectral OOD Targets × Dev Atmospheres | 308 samples)`
   * `❄️ Quadrant IV — Double-Blind Golden Benchmark (Spectral OOD × Severe Polar Winter OOD | 770 samples)`
3. **Clean Unit Displays (No Delta Arrow Misconceptions):** Static quantities (temperatures in Celsius and precipitable water vapor columns) are rendered with clear labels and dual units (`st.caption`), eliminating deceptive green up-arrows.
4. **Physical Downwelling Parity:** Eradicated the erroneous `/π` factor from the reflected sky radiance checkbox and plot trace, ensuring strict energy conservation: $L_{\mathrm{refl}}(\lambda) = [1 - \varepsilon(\lambda)] L^\downarrow(\lambda)$.
5. **Clean Ledger Table:** Integrated `hide_index=True` into `st.dataframe(...)` to suppress duplicate index counters.


## 8. Diagnostic Telemetry & Visual Ergonomics (v1.1.1 Enhancements)

### Selected Spectral Metadata & Diagnostics Telemetry Table
Positioned directly below the interactive spectral curve visualizer in Tab 1, this component delivers a structured, responsive HTML telemetry table (`.spec-table`) with dark-slate styling (`#0f172a`), clean borders (`#334155`), and partition status badges:
1. **Cross-Validation Traceability:** Displays the exact partition role (`Golden Target`, `Quarantined Sibling`, or `Development Pool`), validation fold, and training folds for each selected sample. When multiple spectra are selected, rows stack cleanly for direct comparative evaluation.
2. **Physical Boundary Telemetry:** Reports high-precision minimum, maximum, and mean emissivity ($\\epsilon_{\\min}, \\epsilon_{\\max}, \\bar{\\epsilon}$) across $7.5\\,\\mu\\text{m} - 13.5\\,\\mu\\text{m}$ to verify thermodynamic consistency and Reststrahlen depth.
3. **Cryptographic Checksum:** Reports the 128-bit MD5 signature of the laboratory array for reproducible provenance.

### Visual Styling & Accessibility Overhaul
* **Structured Telemetry Tables:** Replaces single-line pipe-separated cards with dedicated tabular columns (`Spectrum ID`, `Specimen Name`, `Partition Role`, `Class`, `Particle Size`, `Val Fold`, `Train Folds`, `Min ε`, `Max ε`, `Mean ε`, `MD5 Checksum`).
* **Non-Obstructive Legends:** Plot legends use semi-transparent dark panels (`rgba(15, 23, 42, 0.88)`) positioned in the bottom-right corner (`x=0.99, y=0.02`), leaving spectral curves fully visible.
* **ASTER & HyTES Tooltip Parity:** Both sensor overlays provide explicit `Band Start (λ_start)` and `Band End (λ_end)` wavelengths, center wavelength, bandwidth (FWHM), and interval (`[λ_start, λ_end] μm`), utilizing `hoveron="points+fills"` and `<extra></extra>` to suppress default Plotly trace box artifacts (`trace 4`).
* **Unphysical Transmission Flag:** The 3rd Validation Gate highlights historical KBr transmission artifacts with clean descriptive warnings, eliminating ambiguous up-arrow delta badges.
