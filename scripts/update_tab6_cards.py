import os

old_caption_block = '''        st.caption(
            "Evaluate full photon decomposition across the **2×2 Generalization Matrix**: "
            "In-Distribution Training (**Quadrant I:** 5,500 samples), "
            "Out-of-Fold Validation (**Quadrant I:** 2,024 samples), "
            "Pure Spectral OOD (**Quadrant II:** 308 samples), and "
            "Extreme Joint OOD (**Quadrant IV:** 770 samples)."
        )'''

new_cards_block = '''        part_c1, part_c2, part_c3, part_c4 = st.columns(4)
        with part_c1:
            st.markdown(
                \'<div style="background-color: #0f172a; border: 1px solid #334155; border-left: 4px solid #3b82f6; border-radius: 6px; padding: 10px 14px; height: 100%;">\'
                \'<div style="font-size: 0.74rem; font-weight: 700; color: #93c5fd; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">🏋️ Quadrant I: Train</div>\'
                \'<div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 2px;">5,500 <span style="font-size: 0.78rem; font-weight: 400; color: #94a3b8;">samples</span></div>\'
                \'<div style="font-size: 0.80rem; color: #cbd5e1; line-height: 1.35;">Master Training Pool (Dev Materials × Nominal Atmospheres).</div>\'
                \'</div>\',
                unsafe_allow_html=True
            )
        with part_c2:
            st.markdown(
                \'<div style="background-color: #0f172a; border: 1px solid #334155; border-left: 4px solid #10b981; border-radius: 6px; padding: 10px 14px; height: 100%;">\'
                \'<div style="font-size: 0.74rem; font-weight: 700; color: #6ee7b7; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">🧪 Quadrant I: Val OOF</div>\'
                \'<div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 2px;">2,024 <span style="font-size: 0.78rem; font-weight: 400; color: #94a3b8;">samples</span></div>\'
                \'<div style="font-size: 0.80rem; color: #cbd5e1; line-height: 1.35;">Out-of-Fold Validation (Unseen Fold 0 Dev Specimens).</div>\'
                \'</div>\',
                unsafe_allow_html=True
            )
        with part_c3:
            st.markdown(
                \'<div style="background-color: #0f172a; border: 1px solid #334155; border-left: 4px solid #f59e0b; border-radius: 6px; padding: 10px 14px; height: 100%;">\'
                \'<div style="font-size: 0.74rem; font-weight: 700; color: #fcd34d; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">🏆 Quadrant II: Pure OOD</div>\'
                \'<div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 2px;">308 <span style="font-size: 0.78rem; font-weight: 400; color: #94a3b8;">samples</span></div>\'
                \'<div style="font-size: 0.80rem; color: #cbd5e1; line-height: 1.35;">Zero-Shot Material Generalization (7 Quarantined Golden Targets).</div>\'
                \'</div>\',
                unsafe_allow_html=True
            )
        with part_c4:
            st.markdown(
                \'<div style="background-color: #0f172a; border: 1px solid #334155; border-left: 4px solid #a855f7; border-radius: 6px; padding: 10px 14px; height: 100%;">\'
                \'<div style="font-size: 0.74rem; font-weight: 700; color: #d8b4fe; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">❄️ Quadrant IV: Double OOD</div>\'
                \'<div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 2px;">770 <span style="font-size: 0.78rem; font-weight: 400; color: #94a3b8;">samples</span></div>\'
                \'<div style="font-size: 0.80rem; color: #cbd5e1; line-height: 1.35;">Double-Blind Stress Test (Golden Targets × Severe Polar Winter).</div>\'
                \'</div>\',
                unsafe_allow_html=True
            )'''

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

    if old_caption_block in norm_content:
        norm_content = norm_content.replace(old_caption_block, new_cards_block, 1)
        if is_crlf:
            final_content = norm_content.replace("\n", "\r\n")
        else:
            final_content = norm_content
        with open(fp, "w", encoding="utf-8") as f:
            f.write(final_content)
        print(f"Updated Tab 6 partition cards in {fp} successfully!")
    else:
        print(f"Old caption block not found in {fp}")

print("Update complete.")

