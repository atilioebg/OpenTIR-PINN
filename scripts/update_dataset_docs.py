import re

def update_data_dictionary():
    path = "zenodo_release/DATA_DICTIONARY.md"
    print(f"Updating {path}...")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    target = "* **Total Sample Count Across Both Sensors (ASTER 5-Band + HyTES 256-Channel):** $8,602 \\times 2 = 17,204$ forward coupled radiance observations.\n"
    
    addition = """
#### Intuitive Combinatorial Summary Formula ("The Quick Rule-of-Thumb"):
For fast physical calculation of sample accounting across any partition:

$$\\mathbf{N_{\\text{observations}} = (N_{\\text{targets}} \\times N_{\\text{atmospheres}} \\times N_{\\text{temperatures}}) + \\text{Sensor Radiometric Noise } (\\mathrm{NE}\\Delta T)}$$

* **🏋️ Quadrant I (Master Train):** $500\\text{ samples/class} \\times 11\\text{ classes} = \\mathbf{5,500\\text{ observations}}$ *(continuous $T_s \\in [275, 335]\\,\\mathrm{K}$, stochastic shape jitter $\\varepsilon_{\\text{aug}}$, dynamic sensor noise $0.5\\times\\text{--}1.5\\times \\mathrm{NE}\\Delta T$)*.
* **🧪 Quadrant I (Val OOF):** $46\\text{ unseen spectra} \\times 4\\text{ nominal atmospheres} \\times 11\\text{ temperatures} = \\mathbf{2,024\\text{ observations}} + \\text{nominal sensor noise } (\\mathrm{NE}\\Delta T = 0.1\\,\\mathrm{K})$.
* **🏆 Quadrant II (Golden OOD):** $7\\text{ golden targets} \\times 4\\text{ nominal atmospheres} \\times 11\\text{ temperatures} = \\mathbf{308\\text{ observations}} + \\text{nominal sensor noise } (\\mathrm{NE}\\Delta T = 0.1\\,\\mathrm{K})$.
* **❄️ Quadrant IV (Double OOD):** $7\\text{ golden targets} \\times 10\\text{ polar winter atmospheres} \\times 11\\text{ temperatures} = \\mathbf{770\\text{ observations}} + \\text{nominal sensor noise } (\\mathrm{NE}\\Delta T = 0.1\\,\\mathrm{K})$.
* **⚡ Benchmark Total per Sensor:** $5,500 + 2,024 + 308 + 770 = \\mathbf{8,602\\text{ observations}}$ ($\\times 2\\text{ sensors (ASTER + HyTES)} = \\mathbf{17,204\\text{ total forward samples}}$).

#### Two-Tier Forward Perturbation Architecture (Realistic Noise Modeling):
The dataset is engineered to reflect real-world detector noise and eliminate synthetic overfitting:
1. **Tier 1 (Sensor-Specific Instrumental Radiometric Noise):** Present across **ALL 4 partitions** (Train, Val OOF, Q2, Q4). Noise standard deviation is analytically derived via Planck's temperature derivative:
   $$\\sigma_L(\\lambda) = \\left. \\frac{\\partial B(\\lambda, T)}{\\partial T} \\right|_{T = 300\\,\\mathrm{K}} \\times \\mathrm{NE}\\Delta T \\times \\text{noise\\_multiplier}$$
   with $\\mathrm{NE}\\Delta T = 0.1\\,\\mathrm{K}$ at $T_{\\mathrm{ref}} = 300\\,\\mathrm{K}$. Independent Gaussian noise is injected into surface radiance $L_{\\mathrm{surf}}$, at-sensor radiance $L_{\\mathrm{sensor}}$, and downwelling radiance $L^\\downarrow$ (with $0.5\\times$ scaling for hemispherical integration).
2. **Tier 2 (Stochastic Shape Augmentation & Continuous Temperature):** Applied **exclusively to Quadrant I Training** ($\\varepsilon_{\\mathrm{aug}} = \\operatorname{clip}(\\varepsilon \\times \\mathcal{U}(0.995, 1.005) + \\mathcal{N}(0, 0.002^2), 0.0, 1.0)$ and $T_s \\sim \\mathcal{U}(275, 335)\\,\\mathrm{K}$ with $\\text{noise\\_multiplier} \\sim \\mathcal{U}(0.5, 1.5)$). All test and validation vaults (Val OOF, Q2, Q4) preserve 100% pristine laboratory FTIR crystal lattice signatures ($\\varepsilon_{\\mathrm{pure}}$, `is_augmented = False`) under deterministic temperature grids ($T_s \\in [280, 330]\\,\\mathrm{K}$ in $5\\,\\mathrm{K}$ steps) and nominal sensor noise ($\\text{noise\\_multiplier} = 1.0$).
"""
    if target in content and "Intuitive Combinatorial Summary Formula" not in content:
        content = content.replace(target, target + addition)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("  Updated DATA_DICTIONARY.md successfully.")
    else:
        print("  DATA_DICTIONARY.md already updated or target not found.")


def update_readme():
    path = "zenodo_release/README.md"
    print(f"Updating {path}...")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    target = "* **Total Forward Coupled Samples Across Both Sensors (ASTER + HyTES):** $8,602 \\times 2 = 17,204$ forward coupled radiance observations.\n"
    
    addition = """
### 5.3 Intuitive Combinatorial Summary Formula ("The Quick Rule-of-Thumb")
For fast physical calculation of sample accounting across the benchmark:

$$\\mathbf{N_{\\text{observations}} = (N_{\\text{targets}} \\times N_{\\text{atmospheres}} \\times N_{\\text{temperatures}}) + \\text{Sensor Radiometric Noise } (\\mathrm{NE}\\Delta T)}$$

* **🏋️ Quadrant I (Master Train):** $500\\text{ samples/class} \\times 11\\text{ classes} = \\mathbf{5,500\\text{ observations}}$ *(continuous $T_s \\in [275, 335]\\,\\mathrm{K}$, stochastic shape jitter $\\varepsilon_{\\text{aug}}$, dynamic sensor noise $0.5\\times\\text{--}1.5\\times \\mathrm{NE}\\Delta T$)*.
* **🧪 Quadrant I (Val OOF):** $46\\text{ unseen spectra} \\times 4\\text{ nominal atmospheres} \\times 11\\text{ temperatures} = \\mathbf{2,024\\text{ observations}} + \\text{nominal sensor noise } (\\mathrm{NE}\\Delta T = 0.1\\,\\mathrm{K})$.
* **🏆 Quadrant II (Golden OOD):** $7\\text{ golden targets} \\times 4\\text{ nominal atmospheres} \\times 11\\text{ temperatures} = \\mathbf{308\\text{ observations}} + \\text{nominal sensor noise } (\\mathrm{NE}\\Delta T = 0.1\\,\\mathrm{K})$.
* **❄️ Quadrant IV (Double OOD):** $7\\text{ golden targets} \\times 10\\text{ polar winter atmospheres} \\times 11\\text{ temperatures} = \\mathbf{770\\text{ observations}} + \\text{nominal sensor noise } (\\mathrm{NE}\\Delta T = 0.1\\,\\mathrm{K})$.
* **⚡ Benchmark Total per Sensor:** $5,500 + 2,024 + 308 + 770 = \\mathbf{8,602\\text{ observations}}$ ($\\times 2\\text{ sensors (ASTER + HyTES)} = \\mathbf{17,204\\text{ total forward samples}}$).

### 5.4 The Two-Tier Forward Perturbation Architecture (Realistic Radiometric Noise Modeling)
To prevent synthetic-to-real performance degradation, the benchmark decouples instrumental detection noise from training augmentation:
1. **Tier 1 — Sensor-Specific Radiometric Noise Injection ($\\mathrm{NE}\\Delta T = 0.1\\,\\mathrm{K}$):**
   * **Active in ALL 4 Quadrants** (Master Train, Val OOF, Q2 Pure OOD, Q4 Double OOD).
   * Noise standard deviation $\\sigma_L(\\lambda)$ is computed from the physical thermal sensitivity derivative:
     $$\\sigma_L(\\lambda) = \\left. \\frac{\\partial B(\\lambda, T)}{\\partial T} \\right|_{T = 300\\,\\mathrm{K}} \\times \\mathrm{NE}\\Delta T \\times \\text{noise\\_multiplier}$$
     where $\\partial B/\\partial T = \\frac{c_1 c_2}{\\lambda^6 T^2} \\frac{\\exp(c_2 / (\\lambda T))}{[\\exp(c_2 / (\\lambda T)) - 1]^2}$. Independent Gaussian noise is injected into $L_{\\mathrm{surf}}$, $L_{\\mathrm{sensor}}$, and diffuse $L^\\downarrow$ ($0.5\\times$ scaling).
2. **Tier 2 — Stochastic Shape Augmentation ($\\varepsilon_{\\mathrm{aug}}$):**
   * **Applied EXCLUSIVELY to Quadrant I Training:** $\\varepsilon_{\\mathrm{aug}} = \\operatorname{clip}(\\varepsilon \\times \\mathcal{U}(0.995, 1.005) + \\mathcal{N}(0, 0.002^2), 0.0, 1.0)$.
   * **Quarantined Test Vaults (Val OOF, Q2, Q4) Preserve PURE FTIR Spectra:** Emissivities remain 100% pristine laboratory curves without shape jitter (`is_augmented = False`), maintaining genuine mineral Reststrahlen lattice doublets for double-blind benchmarking.
"""
    if target in content and "Intuitive Combinatorial Summary Formula" not in content:
        content = content.replace(target, target + addition)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("  Updated README.md successfully.")
    else:
        print("  README.md already updated or target not found.")


def update_dashboard_guide():
    path = "zenodo_release/DASHBOARD_USER_GUIDE.md"
    print(f"Updating {path}...")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    target = "* **Total Across Both Sensors (ASTER 5-Band + HyTES 256-Channel):** $8,602 \\times 2 = 17,204$ samples.\n"
    
    addition = """
### Quick-Reference Combinatorial Formula ("The Quick Rule-of-Thumb")
$$\\mathbf{N_{\\text{observations}} = (N_{\\text{targets}} \\times N_{\\text{atmospheres}} \\times N_{\\text{temperatures}}) + \\text{Sensor Radiometric Noise } (\\mathrm{NE}\\Delta T)}$$

* **Quadrant I (Train):** $500\\text{ samples/class} \\times 11\\text{ classes} = \\mathbf{5,500\\text{ samples}}$ *(continuous $T_s$, shape jitter $\\varepsilon_{\\text{aug}}$, dynamic $\\text{NE}\\Delta T$)*.
* **Quadrant I (Val OOF):** $46\\text{ unseen specs} \\times 4\\text{ atms} \\times 11\\text{ temps} = \\mathbf{2,024\\text{ samples}} + \\text{nominal sensor noise } (\\mathrm{NE}\\Delta T = 0.1\\,\\mathrm{K})$.
* **Quadrant II (Pure OOD):** $7\\text{ golden targets} \\times 4\\text{ atms} \\times 11\\text{ temps} = \\mathbf{308\\text{ samples}} + \\text{nominal sensor noise } (\\mathrm{NE}\\Delta T = 0.1\\,\\mathrm{K})$.
* **Quadrant IV (Double OOD):** $7\\text{ golden targets} \\times 10\\text{ polar atms} \\times 11\\text{ temps} = \\mathbf{770\\text{ samples}} + \\text{nominal sensor noise } (\\mathrm{NE}\\Delta T = 0.1\\,\\mathrm{K})$.
* **Total per Sensor:** $5,500 + 2,024 + 308 + 770 = \\mathbf{8,602\\text{ samples}}$ ($\\times 2\\text{ sensors} = \\mathbf{17,204\\text{ observations}}$).

### Two-Tier Forward Noise Architecture (Sensor Noise vs. Augmentation)
* **Tier 1 (Instrumental Noise):** Present in ALL 4 quadrants. Noise standard deviation $\\sigma_L(\\lambda) = \\left.\\frac{\\partial B}{\\partial T}\\right|_{300\\,\\mathrm{K}} \\times \\mathrm{NE}\\Delta T \\times \\text{noise\\_multiplier}$ with $\\mathrm{NE}\\Delta T = 0.1\\,\\mathrm{K}$.
* **Tier 2 (Stochastic Augmentation):** Applied ONLY to Quadrant I Training. Quarantined test vaults (Val OOF, Q2, Q4) preserve 100% pristine laboratory FTIR crystal lattice signatures.
"""
    if target in content and "Quick-Reference Combinatorial Formula" not in content:
        content = content.replace(target, target + addition)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("  Updated DASHBOARD_USER_GUIDE.md successfully.")
    else:
        print("  DASHBOARD_USER_GUIDE.md already updated or target not found.")

if __name__ == "__main__":
    update_data_dictionary()
    update_readme()
    update_dashboard_guide()

