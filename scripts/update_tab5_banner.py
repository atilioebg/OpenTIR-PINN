import os

old_block = '''        st.caption(
            "**Library & Physical Origin:** This interactive studio couples any of the **241 pristine laboratory FTIR spectra** "
            "from the curated **ECOSTRESS Library v1.0** (141 geological/pedological/urban solids + 100 diversity-maximized botanical "
            "endmembers via $L_2$ Euclidean FPS) with **30 MODTRAN 5.2 look-up tables** across arbitrary kinetic temperatures $T_s$."
        )'''

new_block = '''        info_c1, info_c2, info_c3 = st.columns(3)
        with info_c1:
            st.markdown(
                \'<div style="background-color: #0f172a; border: 1px solid #334155; border-left: 4px solid #3b82f6; border-radius: 6px; padding: 10px 14px; height: 100%;">\'
                \'<div style="font-size: 0.76rem; font-weight: 700; color: #93c5fd; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">🔬 Curated Emissivity (241 Spectra)</div>\'
                \'<div style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.45;">\'
                \'Curated from <b>ECOSTRESS v1.0</b>: 141 geological & urban solids + 100 diversity-maximized botanical endmembers (via <i>L</i><sub>2</sub> Euclidean FPS).\'
                \'</div></div>\',
                unsafe_allow_html=True
            )
        with info_c2:
            st.markdown(
                \'<div style="background-color: #0f172a; border: 1px solid #334155; border-left: 4px solid #10b981; border-radius: 6px; padding: 10px 14px; height: 100%;">\'
                \'<div style="font-size: 0.76rem; font-weight: 700; color: #6ee7b7; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">🌤️ Atmospheric Profiles (30 LUTs)</div>\'
                \'<div style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.45;">\'
                \'High-fidelity <b>MODTRAN 5.2</b> look-up tables spanning 6 standard geographic models across 5 water vapor scales (PWV: 0.21–8.23 g/cm²).\'
                \'</div></div>\',
                unsafe_allow_html=True
            )
        with info_c3:
            st.markdown(
                \'<div style="background-color: #0f172a; border: 1px solid #334155; border-left: 4px solid #f59e0b; border-radius: 6px; padding: 10px 14px; height: 100%;">\'
                \'<div style="font-size: 0.76rem; font-weight: 700; color: #fcd34d; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">🌡️ Thermodynamic State (Ts)</div>\'
                \'<div style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.45;">\'
                \'Arbitrary kinetic surface temperature <b><i>T</i><sub>s</sub></b> dynamically coupled with Planck blackbody radiation <i>B</i>(λ, <i>T</i><sub>s</sub>).\'
                \'</div></div>\',
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

    if old_block in norm_content:
        norm_content = norm_content.replace(old_block, new_block, 1)
        if is_crlf:
            final_content = norm_content.replace("\n", "\r\n")
        else:
            final_content = norm_content
        with open(fp, "w", encoding="utf-8") as f:
            f.write(final_content)
        print(f"Updated Tab 5 banner in {fp} successfully!")
    else:
        print(f"Old block not found in {fp}")

print("Update complete.")

