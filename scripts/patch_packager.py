with open("scripts/package_zenodo_v1.1.0.py", "r", encoding="utf-8") as f:
    code = f.read()

target_dict = "* **Total Sample Count Across Both Sensors (ASTER 5-Band + HyTES 256-Channel):** $8,602 \\times 2 = 17,204$ forward coupled radiance observations.\n"

addition_dict = """
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

if target_dict in code and "Intuitive Combinatorial Summary Formula" not in code:
    code = code.replace(target_dict, target_dict + addition_dict, 1)
    with open("scripts/package_zenodo_v1.1.0.py", "w", encoding="utf-8") as f:
        f.write(code)
    print("Patched scripts/package_zenodo_v1.1.0.py successfully!")
else:
    print("Target already present or not found in scripts/package_zenodo_v1.1.0.py.")

