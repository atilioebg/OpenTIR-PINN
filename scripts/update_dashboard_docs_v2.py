import os

def update_guides_and_clarification():
    # 1. Update DASHBOARD_USER_GUIDE.md
    for fpath in ["zenodo_release/DASHBOARD_USER_GUIDE.md", "docs/DASHBOARD_USER_GUIDE.md"]:
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        
        is_crlf = "\r\n" in content
        norm = content.replace("\r\n", "\n")

        # Update Tab 1 description
        old_tab1 = "* **Sensor Response Function (SRF) Overlays:**\n    * *ASTER 5-Band Filter Windows:* Overlays the 5 discrete thermal filter bands of the spaceborne ASTER instrument (B10 at $8.3\\,\\mu\\text{m}$, B11 at $8.6\\,\\mu\\text{m}$, B12 at $9.1\\,\\mu\\text{m}$, B13 at $10.6\\,\\mu\\text{m}$, and B14 at $11.3\\,\\mu\\text{m}$).\n    * *HyTES 256-Channel Spectrometer Grid:* Renders the complete contiguous array of all 256 narrow spectral channels ($7.49\\text{--}11.99\\,\\mu\\text{m}$, $\\Delta\\lambda \\approx 17.5\\,\\text{nm}$) with alternating shading and 0.5px borders on the sub-layer (`layer=\"below\"`), allowing seamless visual assessment of how hyperspectral sampling captures fine Reststrahlen curvature."
        new_tab1 = "* **Sensor Response Function (SRF) Overlays (Collision-Free Interactive Hover Tooltips):**\n    * *ASTER 5-Band Filter Windows:* Overlays the 5 discrete thermal filter bands of the spaceborne ASTER instrument (B10 at $8.3\\,\\mu\\text{m}$, B11 at $8.6\\,\\mu\\text{m}$, B12 at $9.1\\,\\mu\\text{m}$, B13 at $10.6\\,\\mu\\text{m}$, and B14 at $11.3\\,\\mu\\text{m}$) using fill-polygons with interactive hover tooltips (`hoveron=\"fills\"`). Placing the mouse over any band immediately displays its band name, center wavelength, spectral window, and FWHM bandwidth, with zero static text collisions.\n    * *HyTES 256-Channel Spectrometer Grid:* Renders the complete contiguous array of all 256 narrow spectral channels ($7.49\\text{--}11.99\\,\\mu\\text{m}$, $\\Delta\\lambda \\approx 17.5\\,\\text{nm}$) with alternating shading and 0.5px borders on the sub-layer (`layer=\"below\"`), coupled with an interactive channel-center hover array that displays channel index, center wavelength, bandwidth, and interval without top-banner clutter."

        if old_tab1 in norm:
            norm = norm.replace(old_tab1, new_tab1)

        # Update Tab 2 description
        old_tab2 = "* **Unclipped Visualization:** Anchored at `[0.0, 1.02]`, fully preserving deep quartz Reststrahlen troughs ($\\epsilon \\approx 0.55$) without truncation."
        new_tab2 = "* **Unclipped Visualization & Bottom-Right Legend:** Anchored at `[0.0, 1.02]`, fully preserving deep quartz Reststrahlen troughs ($\\epsilon \\approx 0.55$) without truncation. The trace legend is anchored in the bottom-right corner (`x=0.99, y=0.02`) on a translucent card, keeping the upper curve domain completely unobstructed."

        if old_tab2 in norm:
            norm = norm.replace(old_tab2, new_tab2)

        # Update Tab 3 description
        old_tab3 = "* **Training Diversity Verification:** Mathematically proves that the training distribution possesses sufficient compositional and granulometric entropy to train robust Sobolev-regularized neural networks capable of domain generalization."
        new_tab3 = "* **Training Diversity Verification & Dedicated Y-Axis Scaling:** Mathematically proves that the training distribution possesses sufficient compositional and granulometric entropy to train robust Sobolev-regularized neural networks capable of domain generalization. For this tab specifically, the y-axis is anchored at $[0.40, 1.02]$ because all mineral and land-surface classes strictly exceed $\\epsilon = 0.40$, maximizing vertical dynamic resolution."

        if old_tab3 in norm:
            norm = norm.replace(old_tab3, new_tab3)

        # Update Tab 4 description
        old_tab4 = "* **Side-by-Side Visual Confrontation:** Renders a direct overlay contrasting a quarantined artifact (e.g. powder in KBr) against its true physical reflectance-derived counterpart (coarse anorthite), visually demarcating the \"Forbidden Region\" ($\\epsilon < 0.40$)."
        new_tab4 = "* **Dynamic Mineralogical Benchmark Pairing:** Features automated dynamic matching between the selected quarantined artifact and its exact physical benchmark counterpart in the curated library (e.g., matching `calcite_1 fine in KBr` against `calcite_1 coarse`, `quartz_1 fine in KBr` against `quartz_1 coarse`, `anorthite_1 fine in KBr` against `anorthite_1 coarse`). The title and legend update dynamically, visually demarcating the \"Forbidden Region\" ($\\epsilon < 0.40$) without arbitrary mineral cross-matching."

        if old_tab4 in norm:
            norm = norm.replace(old_tab4, new_tab4)

        updated = norm.replace("\n", "\r\n") if is_crlf else norm
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(updated)
        print(f"Updated {fpath}")

    # 2. Update clarification.md (Entry 31)
    with open("clarification.md", "r", encoding="utf-8") as f:
        c_content = f.read()
    
    is_crlf = "\r\n" in c_content
    c_norm = c_content.replace("\r\n", "\n")

    old_c_tab1 = "* *Sensor Response Overlays (SRF):* Displays the 5 thermal bands of spaceborne ASTER ($8.125\\text{--}11.65\\,\\mu\\text{m}$) and the complete contiguous 256 narrow channel grid of airborne HyTES ($7.49\\text{--}11.99\\,\\mu\\text{m}$, $\\Delta\\lambda \\approx 17.5\\,\\text{nm}$) with alternating shading and 0.5px borders on the sub-layer (`layer=\"below\"`)."
    new_c_tab1 = "* *Sensor Response Overlays (SRF):* Displays the 5 thermal bands of spaceborne ASTER ($8.125\\text{--}11.65\\,\\mu\\text{m}$) and the complete contiguous 256 narrow channel grid of airborne HyTES ($7.49\\text{--}11.99\\,\\mu\\text{m}$, $\\Delta\\lambda \\approx 17.5\\,\\text{nm}$) with interactive hover tooltips (`hoveron=\"fills\"`), eliminating top-banner text collisions with curve titles and legends."

    old_c_tab4 = "* *Side-by-Side Confrontation:* Direct visual overlay comparing unphysical KBr pellet transmission artifacts ($\\epsilon < 0.40$, volumetric scattering) against true physical benchmarks (coarse anorthite)."
    new_c_tab4 = "* *Dynamic Mineralogical Matching:* Automatically pairs any quarantined artifact with its genuine physical benchmark in the library (e.g. Calcite artifact against Calcite coarse benchmark, Quartz against Quartz, Kaolinite against Kaolinite), ensuring scientifically valid side-by-side confrontation."

    if old_c_tab1 in c_norm:
        c_norm = c_norm.replace(old_c_tab1, new_c_tab1)
    if old_c_tab4 in c_norm:
        c_norm = c_norm.replace(old_c_tab4, new_c_tab4)

    updated_c = c_norm.replace("\n", "\r\n") if is_crlf else c_norm
    with open("clarification.md", "w", encoding="utf-8") as f:
        f.write(updated_c)
    print("Updated clarification.md")

if __name__ == "__main__":
    update_guides_and_clarification()

