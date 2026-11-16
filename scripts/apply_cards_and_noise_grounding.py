import sys
import re

expander_appendix = """Total benchmark corpus across both sensor architectures: $8,602 \\times 2 = 17,204$ coupled radiative observations.

---

#### 4. 💡 Intuitive Combinatorial Summary Formula ("The Quick Rule-of-Thumb")
For instantaneous, high-level verification of every observation count across the benchmark, remember this simple governing relation:

$$\\mathbf{N_{\\text{observations}} = (N_{\\text{targets}} \\times N_{\\text{atmospheres}} \\times N_{\\text{temperatures}}) + \\text{Sensor Radiometric Noise } (\\mathrm{NE}\\Delta T)}$$

* **🏋️ Quadrant I (Master Train):** $500\\text{ samples/class} \\times 11\\text{ classes} = \\mathbf{5,500\\text{ observations}}$ *(continuous $T_s \\in [275, 335]\\,\\mathrm{K}$, stochastic shape jitter $\\varepsilon_{\\text{aug}}$, dynamic sensor noise $0.5\\times\\text{--}1.5\\times \\mathrm{NE}\\Delta T$)*.
* **🧪 Quadrant I (Val OOF):** $46\\text{ unseen spectra} \\times 4\\text{ nominal atmospheres} \\times 11\\text{ temperatures} = \\mathbf{2,024\\text{ observations}} + \\text{nominal sensor noise } (\\mathrm{NE}\\Delta T = 0.1\\,\\mathrm{K})$.
* **🏆 Quadrant II (Golden OOD):** $7\\text{ golden targets} \\times 4\\text{ nominal atmospheres} \\times 11\\text{ temperatures} = \\mathbf{308\\text{ observations}} + \\text{nominal sensor noise } (\\mathrm{NE}\\Delta T = 0.1\\,\\mathrm{K})$.
* **❄️ Quadrant IV (Double OOD):** $7\\text{ golden targets} \\times 10\\text{ polar winter atmospheres} \\times 11\\text{ temperatures} = \\mathbf{770\\text{ observations}} + \\text{nominal sensor noise } (\\mathrm{NE}\\Delta T = 0.1\\,\\mathrm{K})$.
* **⚡ Benchmark Total per Sensor:** $5,500 + 2,024 + 308 + 770 = \\mathbf{8,602\\text{ observations}}$ ($\\times 2\\text{ sensors (ASTER + HyTES)} = \\mathbf{17,204\\text{ total forward samples}}$).

---

#### 5. 🔬 The Two-Tier Forward Perturbation Architecture (Noise vs. Augmentation)
Remote sensing models trained exclusively on noise-free synthetic radiances fail on physical instruments due to detector sensitivity thresholds. Our simulation strictly enforces two distinct physical perturbation layers:
1. **Tier 1 — Instrumental Radiometric Noise Injection ($\\mathrm{NE}\\Delta T$-Grounded):**
   * **Active in ALL 4 Quadrants** (Master Train, Val OOF, Q2 Pure OOD, Q4 Double OOD).
   * Scaled analytically using the Planck temperature derivative at $300\\,\\mathrm{K}$:
     $$\\sigma_L(\\lambda) = \\left. \\frac{\\partial B(\\lambda, T)}{\\partial T} \\right|_{T = 300\\,\\mathrm{K}} \\times \\mathrm{NE}\\Delta T \\times \\text{noise\\_multiplier}$$
     with $\\mathrm{NE}\\Delta T = 0.1\\,\\mathrm{K}$. Injected into $L_{\\mathrm{surf}}$, $L_{\\mathrm{sensor}}$, and diffuse $L^\\downarrow$ (with $0.5\\times$ scaling for hemispherical sky integration).
2. **Tier 2 — Stochastic Shape Augmentation ($\\varepsilon_{\\mathrm{aug}}$):**
   * **Applied EXCLUSIVELY to Quadrant I Training:** $\\varepsilon_{\\mathrm{aug}} = \\operatorname{clip}(\\varepsilon \\times \\mathcal{U}(0.995, 1.005) + \\mathcal{N}(0, 0.002^2), 0.0, 1.0)$.
   * **Quarantined Test Vaults (Val OOF, Q2, Q4) Use PURE FTIR Spectra:** Emissivities remain 100% pristine laboratory curves without shape jitter (`is_augmented = False`), preserving genuine mineral Reststrahlen lattice doublets for double-blind benchmarking.
\\\"\\\"\\\")"""

def update_dashboard_file(filepath):
    print(f"Updating {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update Top KPI metric from 16,204 to 17,204
    old_kpi = 'st.metric("Forward Coupled Samples", "16,204")'
    new_kpi = 'st.metric("Forward Coupled Samples", "17,204")'
    if old_kpi in content:
        content = content.replace(old_kpi, new_kpi)
        print("  Updated KPI metric to 17,204.")
    else:
        print("  KPI metric 16,204 not found (already updated?).")

    # 2. Update the 4 cards in Tab 6
    cards_pattern = re.compile(
        r'part_c1, part_c2, part_c3, part_c4 = st\.columns\(4\)\s+with part_c1:.*?with part_c4:.*?'
        r'unsafe_allow_html=True\s+\)',
        re.DOTALL
    )

    new_cards_code = '''part_c1, part_c2, part_c3, part_c4 = st.columns(4)
        with part_c1:
            st.markdown(
                '<div style="background-color: #0f172a; border: 1px solid #334155; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 12px 14px; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">'
                '<div>'
                '<div style="font-size: 0.72rem; font-weight: 700; color: #93c5fd; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">🏋️ Quadrant I: Master Train</div>'
                '<div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">5,500 <span style="font-size: 0.78rem; font-weight: 400; color: #94a3b8;">samples</span></div>'
                '<div style="font-family: monospace; font-size: 0.73rem; color: #60a5fa; background-color: #1e293b; border-radius: 4px; padding: 3px 6px; margin-bottom: 6px; border: 1px solid #3b82f633;">500/class × 11 classes = 5,500</div>'
                '<div style="font-size: 0.75rem; color: #cbd5e1; line-height: 1.35; margin-bottom: 6px;">Master Training: Continuous Ts ∈ [275, 335] K across 4 nominal atmospheres.</div>'
                '</div>'
                '<div style="font-size: 0.71rem; font-weight: 600; color: #93c5fd; border-top: 1px dashed #334155; padding-top: 6px; margin-top: 4px;">⚡ + Stochastic ε<sub>aug</sub> + Sensor Noise (0.5×–1.5× NEΔT)</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with part_c2:
            st.markdown(
                '<div style="background-color: #0f172a; border: 1px solid #334155; border-left: 4px solid #10b981; border-radius: 8px; padding: 12px 14px; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">'
                '<div>'
                '<div style="font-size: 0.72rem; font-weight: 700; color: #6ee7b7; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">🧪 Quadrant I: Val OOF</div>'
                '<div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">2,024 <span style="font-size: 0.78rem; font-weight: 400; color: #94a3b8;">samples</span></div>'
                '<div style="font-family: monospace; font-size: 0.73rem; color: #34d399; background-color: #1e293b; border-radius: 4px; padding: 3px 6px; margin-bottom: 6px; border: 1px solid #10b98133;">46 specs × 4 atms × 11 Ts = 2,024</div>'
                '<div style="font-size: 0.75rem; color: #cbd5e1; line-height: 1.35; margin-bottom: 6px;">Out-of-Fold Val: 0% intra-sample leakage across 11 discrete Ts [280–330 K].</div>'
                '</div>'
                '<div style="font-size: 0.71rem; font-weight: 600; color: #6ee7b7; border-top: 1px dashed #334155; padding-top: 6px; margin-top: 4px;">🛡️ + Pure FTIR + Nominal Sensor Noise (1.0× NEΔT = 0.1 K)</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with part_c3:
            st.markdown(
                '<div style="background-color: #0f172a; border: 1px solid #334155; border-left: 4px solid #f59e0b; border-radius: 8px; padding: 12px 14px; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">'
                '<div>'
                '<div style="font-size: 0.72rem; font-weight: 700; color: #fcd34d; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">🏆 Quadrant II: Pure OOD</div>'
                '<div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">308 <span style="font-size: 0.78rem; font-weight: 400; color: #94a3b8;">samples</span></div>'
                '<div style="font-family: monospace; font-size: 0.73rem; color: #fbbf24; background-color: #1e293b; border-radius: 4px; padding: 3px 6px; margin-bottom: 6px; border: 1px solid #f59e0b33;">7 targets × 4 atms × 11 Ts = 308</div>'
                '<div style="font-size: 0.75rem; color: #cbd5e1; line-height: 1.35; margin-bottom: 6px;">Zero-Shot Material: Quarantined Golden Targets on seen atmospheres.</div>'
                '</div>'
                '<div style="font-size: 0.71rem; font-weight: 600; color: #fcd34d; border-top: 1px dashed #334155; padding-top: 6px; margin-top: 4px;">🛡️ + Pure FTIR + Nominal Sensor Noise (1.0× NEΔT = 0.1 K)</div>'
                '</div>',
                unsafe_allow_html=True
            )
        with part_c4:
            st.markdown(
                '<div style="background-color: #0f172a; border: 1px solid #334155; border-left: 4px solid #a855f7; border-radius: 8px; padding: 12px 14px; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">'
                '<div>'
                '<div style="font-size: 0.72rem; font-weight: 700; color: #d8b4fe; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">❄️ Quadrant IV: Double OOD</div>'
                '<div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">770 <span style="font-size: 0.78rem; font-weight: 400; color: #94a3b8;">samples</span></div>'
                '<div style="font-family: monospace; font-size: 0.73rem; color: #c084fc; background-color: #1e293b; border-radius: 4px; padding: 3px 6px; margin-bottom: 6px; border: 1px solid #a855f733;">7 targets × 10 atms × 11 Ts = 770</div>'
                '<div style="font-size: 0.75rem; color: #cbd5e1; line-height: 1.35; margin-bottom: 6px;">Joint OOD Stress Test: Novel materials on Arctic winter profiles.</div>'
                '</div>'
                '<div style="font-size: 0.71rem; font-weight: 600; color: #d8b4fe; border-top: 1px dashed #334155; padding-top: 6px; margin-top: 4px;">🛡️ + Pure FTIR + Nominal Sensor Noise (1.0× NEΔT = 0.1 K)</div>'
                '</div>',
                unsafe_allow_html=True
            )'''

    match = cards_pattern.search(content)
    if match:
        content = content[:match.start()] + new_cards_code + content[match.end():]
        print("  Replaced cards successfully.")
    else:
        print("  WARNING: cards_pattern did not match!")

    # 3. Add Section 4 (Intuitive Summary Formula) and Section 5 (Two-Tier Noise Architecture) inside the expander
    target_str = 'Total benchmark corpus across both sensor architectures: 8,602 \\times 2 = 17,204$ coupled radiative observations.\n""")'
    if target_str in content:
        content = content.replace(target_str, expander_appendix)
        print("  Added Sections 4 and 5 to expander.")
    else:
        # Check alternative string
        alt_target = 'Total benchmark corpus across both sensor architectures: $8,602 \\times 2 = 17,204$ coupled radiative observations.\n""")'
        if alt_target in content:
            content = content.replace(alt_target, expander_appendix)
            print("  Added Sections 4 and 5 to expander (matched alt_target).")
        else:
            print("  WARNING: target_str in expander not found!")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Finished updating {filepath}.\n")

if __name__ == "__main__":
    update_dashboard_file("zenodo_release/dataset_explorer_dashboard.py")
    update_dashboard_file("scripts/dataset_explorer_dashboard.py")
