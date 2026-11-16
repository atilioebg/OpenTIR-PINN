import os

with open("clarification.md", "r", encoding="utf-8") as f:
    text = f.read()

target = """3. **ASTER Band Overlay Tooltip Parity:**
   * Configured `hovertemplate` with `<extra></extra>` to suppress Plotly's default `trace 4` secondary box.
   * Tooltip structure now mirrors HyTES:
     ```
     ASTER Band #10
     Center Wavelength: 8.2910 μm
     Bandwidth: 354.0 nm
     Interval: [8.1250, 8.4750] μm
     ```"""

replacement = """3. **ASTER & HyTES Tooltip Parity with Explicit Wavelength Boundaries:**
   * Configured `hoveron="points+fills"`, `hoverinfo="text"`, and `hovertemplate="%{hovertext}<extra></extra>"` on both ASTER and HyTES to permanently eliminate fallback `trace X` labels.
   * Explicitly added **Band/Channel Start** ($\\\\lambda_{\\\\mathrm{start}}$) and **Band/Channel End** ($\\\\lambda_{\\\\mathrm{end}}$) fields:
     ```
     ASTER Band #10
     Center Wavelength: 8.2910 μm
     Bandwidth (FWHM): 354.0 nm
     Band Start (λ_start): 8.1250 μm
     Band End (λ_end): 8.4750 μm
     Interval: [8.1250, 8.4750] μm
     ```
     ```
     HyTES Channel #171
     Center Wavelength: 10.4888 μm
     Bandwidth: 17.7 nm
     Channel Start (λ_start): 10.4800 μm
     Channel End (λ_end): 10.4976 μm
     Interval: [10.4800, 10.4976] μm
     ```

4. **Structured HTML Telemetry Table for Selected Spectra:**
   * Transformed the single-line, pipe-delimited card into a publication-grade, responsive HTML telemetry table (`.spec-table`) styled in deep slate (`#0f172a`), with slate borders (`#334155`), header (`#1e293b`), and hover row highlighting.
   * **Columns:** `Spectrum ID` | `Specimen Name` | `Partition Role` | `Class` | `Particle Size` | `Val Fold` | `Train Folds` | `Min ε` | `Max ε` | `Mean ε` | `MD5 Checksum`.
   * Enables immediate multi-sample side-by-side comparison when multiple spectra are selected."""

if target in text:
    text = text.replace(target, replacement)
    with open("clarification.md", "w", encoding="utf-8") as f:
        f.write(text)
    print("Updated clarification.md successfully!")
else:
    print("Target not found in clarification.md")

