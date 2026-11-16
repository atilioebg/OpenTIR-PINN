import os

new_sec8 = """## 8. Diagnostic Telemetry & Visual Ergonomics (v1.1.1 Enhancements)

### Selected Spectral Metadata & Diagnostics Telemetry Table
Positioned directly below the interactive spectral curve visualizer in Tab 1, this component delivers a structured, responsive HTML telemetry table (`.spec-table`) with dark-slate styling (`#0f172a`), clean borders (`#334155`), and partition status badges:
1. **Cross-Validation Traceability:** Displays the exact partition role (`Golden Target`, `Quarantined Sibling`, or `Development Pool`), validation fold, and training folds for each selected sample. When multiple spectra are selected, rows stack cleanly for direct comparative evaluation.
2. **Physical Boundary Telemetry:** Reports high-precision minimum, maximum, and mean emissivity ($\\\\epsilon_{\\\\min}, \\\\epsilon_{\\\\max}, \\\\bar{\\\\epsilon}$) across $7.5\\\\,\\\\mu\\\\text{m} - 13.5\\\\,\\\\mu\\\\text{m}$ to verify thermodynamic consistency and Reststrahlen depth.
3. **Cryptographic Checksum:** Reports the 128-bit MD5 signature of the laboratory array for reproducible provenance.

### Visual Styling & Accessibility Overhaul
* **Structured Telemetry Tables:** Replaces single-line pipe-separated cards with dedicated tabular columns (`Spectrum ID`, `Specimen Name`, `Partition Role`, `Class`, `Particle Size`, `Val Fold`, `Train Folds`, `Min ε`, `Max ε`, `Mean ε`, `MD5 Checksum`).
* **Non-Obstructive Legends:** Plot legends use semi-transparent dark panels (`rgba(15, 23, 42, 0.88)`) positioned in the bottom-right corner (`x=0.99, y=0.02`), leaving spectral curves fully visible.
* **ASTER & HyTES Tooltip Parity:** Both sensor overlays provide explicit `Band Start (λ_start)` and `Band End (λ_end)` wavelengths, center wavelength, bandwidth (FWHM), and interval (`[λ_start, λ_end] μm`), utilizing `hoveron="points+fills"` and `<extra></extra>` to suppress default Plotly trace box artifacts (`trace 4`).
* **Unphysical Transmission Flag:** The 3rd Validation Gate highlights historical KBr transmission artifacts with clean descriptive warnings, eliminating ambiguous up-arrow delta badges.
"""

files = [
    "zenodo_release/DASHBOARD_USER_GUIDE.md",
    "docs/DASHBOARD_USER_GUIDE.md"
]

for fp in files:
    if not os.path.exists(fp):
        continue
    with open(fp, "r", encoding="utf-8") as f:
        text = f.read()
    
    # replace section 8
    idx = text.find("## 8. Diagnostic Telemetry & Visual Ergonomics")
    if idx != -1:
        text = text[:idx] + new_sec8
        with open(fp, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Updated {fp}")

print("Documentation update complete.")

