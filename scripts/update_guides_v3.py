import os

def update_guides_and_clarification():
    # Update DASHBOARD_USER_GUIDE.md
    for fpath in ["zenodo_release/DASHBOARD_USER_GUIDE.md", "docs/DASHBOARD_USER_GUIDE.md"]:
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        is_crlf = "\r\n" in content
        norm = content.replace("\r\n", "\n")

        old_tab5_sec = """### 5. 🌤️ Atmospheric LUTs & Dynamic RTE Studio (Tab 5)
* **What It Represents:** The interactive coupled radiative transfer laboratory and "Component Mixer" driven by 30 high-resolution MODTRAN 5.2 Look-Up Tables (LUTs).
* **Operational & Scientific Purpose:**
  * **Coupled Radiative Transfer Simulation:** Computes the full boundary-value land-atmosphere radiance equation:
    $$L_{\\mathrm{sensor}}(\\lambda) = \\tau(\\lambda) \\left[ \\epsilon(\\lambda) B(\\lambda, T_s) + [1 - \\epsilon(\\lambda)] \\frac{L^\\downarrow(\\lambda)}{\\pi} \\right] + L^\\uparrow(\\lambda)$$
  * **Dynamic Multi-Parameter Coupling:** Allows pairing any of the 241 surface materials with any of the 30 MODTRAN profiles (6 geographic standard models $\\times$ 5 water vapor scaling factors, PWV $0.42\\text{--}8.23\\,\\text{g/cm}^2$) across surface kinetic temperatures $T_s \\in [270\\,\\text{K}, 340\\,\\text{K}]$.
  * **The 8-Component Mixer:** Provides modular radio toggles and individual checkboxes to isolate and analyze each physical mechanism:
    * At-sensor total radiance ($L_{\\mathrm{sensor}}$);
    * Surface-leaving net radiance ($L_{\\mathrm{surf}}$);
    * Direct surface thermal emission ($\\tau \\cdot \\epsilon \\cdot B$);
    * Reflected downwelling sky radiance ($\\tau \\cdot (1-\\epsilon) \\cdot L^\\downarrow / \\pi$);
    * Atmospheric upwelling path radiance ($L^\\uparrow$);
    * Downwelling hemispherical sky irradiance ($L^\\downarrow$);
    * Atmospheric transmittance ($\\tau$) with $9.6\\,\\mu\\text{m}$ ozone absorption demarcation;
    * Ideal Planck blackbody radiance ($B(\\lambda, T_s)$).
  * **Photon Energy Budget Telemetry:** Calculates the exact percentage breakdown of detected photons originating from emission, reflection, and atmospheric path radiance within the critical $8.0\\text{--}12.0\\,\\mu\\text{m}$ atmospheric window, with instant CSV download capability."""

        new_tab5_sec = """### 5. 🌤️ Atmospheric LUTs & Dynamic RTE Studio (Tab 5)
* **What It Represents:** The interactive coupled radiative transfer laboratory and "Component Mixer" driven by 30 high-resolution MODTRAN 5.2 Look-Up Tables (LUTs).
* **Operational & Scientific Purpose:**
  * **Coupled Boundary-Value RTE Simulation:** Computes the exact land-atmosphere boundary radiance balance in a dedicated, uncrowded container:
    $$L_{\\mathrm{sensor}}(\\lambda) = \\tau(\\lambda) \\left[ \\epsilon(\\lambda) B(\\lambda, T_s) + [1 - \\epsilon(\\lambda)] L^\\downarrow(\\lambda) \\right] + L^\\uparrow(\\lambda)$$
    where $L^\\downarrow(\\lambda)$ is the directional downwelling sky radiance from MODTRAN 5.2.
  * **Clear 241-Spectrum Library Provenance:** Explicitly identifies that the ground material dropdown draws from the complete universe of **241 pristine laboratory FTIR spectra** from the curated **ECOSTRESS Library v1.0** (141 geological/pedological/urban solids and 100 diversity-maximized botanical endmembers via $L_2$ Euclidean FPS).
  * **Dynamic Cascading Climatological Controls:** Selecting one of the 6 geographic climatological models (Tropical, Mid-Latitude Summer, Mid-Latitude Winter, Sub-Arctic Summer, Sub-Arctic Winter, 1976 US Standard) dynamically filters the water vapor scaling options to the 5 valid scaled moisture profiles for that atmosphere, accompanied by an informative climatological guidance banner displaying latitude zone, moisture bounds, and typical surface temperatures.
  * **Physical Reflection Integrity (Reflected Sky Radiance):** Solves $L_{\\mathrm{refl}}(\\lambda) = [1 - \\epsilon(\\lambda)] L^\\downarrow(\\lambda)$. Because MODTRAN 5.2's `L_down_surface` is already expressed as directional radiance in $\\text{W}/(\\text{m}^2\\cdot\\text{sr}\\cdot\\mu\\text{m})$, eliminating the double-division by $\\pi$ brings the dashboard into 100% exact numerical agreement with the forward synthesis training pipeline (`synthesize_forward_dataset.py`).
  * **The 8-Component Mixer:** Provides modular radio toggles and individual checkboxes to isolate and analyze each physical mechanism:
    * At-sensor total radiance ($L_{\\mathrm{sensor}}$);
    * Surface-leaving net radiance ($L_{\\mathrm{surf}}$);
    * Direct surface thermal emission ($\\tau \\cdot \\epsilon \\cdot B$);
    * Reflected downwelling sky radiance ($\\tau \\cdot [1-\\epsilon] \\cdot L^\\downarrow$);
    * Atmospheric upwelling path radiance ($L^\\uparrow$);
    * Downwelling atmospheric sky radiance ($L^\\downarrow$);
    * Atmospheric transmittance ($\\tau$) with $9.6\\,\\mu\\text{m}$ ozone absorption demarcation;
    * Ideal Planck blackbody radiance ($B(\\lambda, T_s)$).
  * **Clean Telemetry Telemetry (Arrow-Free Scientific Cards):** Replaces misleading green up-arrow delta badges with authoritative scientific metric cards displaying primary units, secondary Celsius conversions, and descriptive subcaptions."""

        if old_tab5_sec in norm:
            norm = norm.replace(old_tab5_sec, new_tab5_sec)
            updated = norm.replace("\n", "\r\n") if is_crlf else norm
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(updated)
            print(f"Updated {fpath}")
        else:
            print(f"Old Tab 5 sec not found in {fpath}")

    # Update clarification.md (Entry 31)
    with open("clarification.md", "r", encoding="utf-8") as f:
        c_content = f.read()

    is_crlf = "\r\n" in c_content
    c_norm = c_content.replace("\r\n", "\n")

    old_c_tab5 = "* *RTE Numerical Engine:* Solves $L_{\\mathrm{sensor}}(\\lambda) = \\tau(\\lambda) [\\epsilon(\\lambda) B(\\lambda, T_s) + (1-\\epsilon(\\lambda)) L^\\downarrow(\\lambda)/\\pi] + L^\\uparrow(\\lambda)$."
    new_c_tab5 = "* *RTE Numerical Engine:* Solves $L_{\\mathrm{sensor}}(\\lambda) = \\tau(\\lambda) [\\epsilon(\\lambda) B(\\lambda, T_s) + (1-\\epsilon(\\lambda)) L^\\downarrow(\\lambda)] + L^\\uparrow(\\lambda)$, eliminating erroneous double-division by $\\pi$ and establishing exact parity with `synthesize_forward_dataset.py`."

    if old_c_tab5 in c_norm:
        c_norm = c_norm.replace(old_c_tab5, new_c_tab5)
        updated_c = c_norm.replace("\n", "\r\n") if is_crlf else c_norm
        with open("clarification.md", "w", encoding="utf-8") as f:
            f.write(updated_c)
        print("Updated clarification.md")
    else:
        print("Old c_tab5 not found in clarification.md")

if __name__ == "__main__":
    update_guides_and_clarification()

