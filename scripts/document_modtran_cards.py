"""
Generate the card-by-card documentation of the MODTRAN 5.2 decks used to build the atmospheric LUTs.
Reads the archived decks (tape5/*.tp5) and deck_index.csv, parses every card with the FORTRAN formats of the
MODTRAN 5.2.1 user's manual, and writes README_MODTRAN_CARDS.md next to them.

    python scripts/document_modtran_cards.py <cards_dir>
"""
import os
import sys
import glob
import pandas as pd

# (name, first_col, last_col, meaning) - columns are 1-based, as in the manual
CARD_FIELDS = {
    "CARD 1": ("FORMAT (4A1, I1, A1, I4, 10I5, 1X, I4, F8.0, A7) - manual §3, p.11-15", [
        ("MODTRN", 1, 1, "'T' = MODTRAN band-model radiative transport"),
        ("SPEED", 2, 2, "blank = 'slow' (only relevant to the correlated-k option)"),
        ("BINARY", 3, 3, "'F' = ASCII output files"),
        ("LYMOLC", 4, 4, "blank = the 16 auxiliary trace species are not added"),
        ("MODEL", 5, 5, "model atmosphere: 1 Tropical, 2 Mid-Lat. Summer, 3 Mid-Lat. Winter, 4 Sub-Arctic Summer, 5 Sub-Arctic Winter, 6 US Standard 1976 (VARIES)"),
        ("T_BEST", 6, 6, "blank"),
        ("ITYPE", 7, 10, "2 = vertical or slant path between two altitudes"),
        ("IEMSCT", 11, 15, "1 = spectral thermal radiance, no sun/moon"),
        ("IMULT", 16, 20, "1 = multiple scattering on (flux evaluated at the observer)"),
        ("M1-M6", 21, 50, "0 = default temperature/pressure and gas profiles of MODEL"),
        ("MDEF", 51, 55, "0 = default profiles for the heavy molecules"),
        ("I_RD2C", 56, 60, "0 = no user-defined atmosphere"),
        ("NOPRNT", 62, 65, "0 = standard tape6 output"),
        ("TPTEMP", 66, 73, "0 = lower-boundary temperature taken from the first atmospheric level"),
        ("SURREF", 74, 80, "0.06 = spectrally constant surface albedo (= 1 - emissivity) of the lower boundary"),
    ]),
    "CARD 1A": ("FORMAT (3A1, I3, F4.0, F10.0, 2A10, 2A1, 4(1X, A1), F10.0, A1, F9.0, 3F10.0, I10) - manual p.16-19", [
        ("DIS", 1, 1, "'t' = DISORT discrete-ordinate multiple scattering"),
        ("DISAZM", 2, 2, "'F' = no azimuth dependence (irrelevant without sun)"),
        ("DISALB", 3, 3, "'F' = no spherical-albedo calculation"),
        ("NSTR", 4, 6, "8 = number of DISORT streams"),
        ("SFWHM", 7, 10, "0 = default solar irradiance smoothing (not used, IEMSCT = 1)"),
        ("CO2MX", 11, 20, "300 = CO2 mixing ratio in ppmv (MODTRAN default 330; see limitations)"),
        ("H2OSTR", 21, 30, "positive number = scaling factor of the model water-vapour column: 0.5, 0.75, 1.0, 1.5, 2.0 (VARIES)"),
        ("O3STR", 31, 40, "1.0 = scaling factor of the model ozone column (default column)"),
        ("C_PROF", 41, 41, "'0' = no molecular profile scaling"),
        ("LSUNFL", 42, 42, "'F' = default solar irradiance file"),
        ("LBMNAM", 44, 44, "'T' = read the band-model root name from CARD 1A2"),
        ("LFLTNM", 46, 46, "'F' = no instrument filter file"),
        ("H2OAER", 48, 48, "'F' = aerosol optical properties not modified by the water-vapour scaling"),
        ("CDTDIR", 50, 50, "'F' = data files in the default DATA/ directory"),
        ("SOLCON ... NSSALB", 51, 110, "0 / blank = defaults (solar constant, aerosol Angstrom perturbations not used)"),
    ]),
    "CARD 1A2": ("FORMAT (A256) - manual p.21", [
        ("BMNAME", 1, 7, "'01_2009' = 1 cm-1 band-model parameter files (2009 database)"),
    ]),
    "CARD 2": ("FORMAT (A2, I3, A1, I4, A3, I2, 3I5, 5F10.5) - manual §6, p.22-26", [
        ("APLUS", 1, 2, "blank = no user-placed aerosol layers"),
        ("IHAZE", 3, 5, "2 = RURAL boundary-layer aerosol extinction, default meteorological range VIS = 5 km"),
        ("CNOVAM", 6, 6, "blank = NOVAM not used"),
        ("ISEASN", 7, 10, "0 = season set by MODEL (spring-summer for 1, 2, 4, 6; fall-winter for 3, 5)"),
        ("ARUSS", 11, 13, "blank = built-in aerosol optical properties"),
        ("IVULCN", 14, 15, "0 = background stratospheric aerosol profile and extinction"),
        ("ICSTL", 16, 20, "0 (used only by the Navy maritime model)"),
        ("ICLD", 21, 25, "0 = no clouds or rain"),
        ("IVSA", 26, 30, "0 = vertical structure algorithm not used"),
        ("VIS", 31, 40, "0 = default meteorological range of IHAZE (5 km)"),
        ("WSS, WHH, RAINRT", 41, 70, "0 = defaults / no rain"),
        ("GNDALT", 71, 80, "0 = ground altitude 0 km (sea level)"),
    ]),
    "CARD 3": ("FORMAT (6F10.0, I5, 5X, 2F10.0) - manual §13, p.40", [
        ("H1", 1, 10, "observer altitude in km (radiance mode: H1 is always the observer) (VARIES)"),
        ("H2", 11, 20, "final altitude in km (VARIES)"),
        ("ANGLE", 21, 30, "initial zenith angle at H1 in degrees: 180 = nadir view, 0 = zenith, others = quadrature nodes (VARIES)"),
        ("RANGE", 31, 40, "0 = not used"),
        ("BETA", 41, 50, "0 = not used"),
        ("RO", 51, 60, "0 = default Earth radius"),
        ("LENN", 61, 65, "0 = set automatically"),
        ("PHI", 71, 80, "0 = not used"),
    ]),
    "CARD 4": ("FORMAT (4F10.0, 2A1, A8, A7, I3, F10.0) - manual §16, p.45-46", [
        ("V1", 1, 10, "660 cm-1 = initial wavenumber (15.15 um)"),
        ("V2", 11, 20, "1435 cm-1 = final wavenumber (6.97 um)"),
        ("DV", 21, 30, "1 cm-1 = output spectral increment"),
        ("FWHM", 31, 40, "2 cm-1 = slit-function full width at half maximum"),
        ("YFLAG", 41, 41, "'R' = radiances in plot files (plot files are not used)"),
        ("XFLAG", 42, 42, "'N' = nanometres in plot files (plot files are not used)"),
        ("DLIMIT", 43, 50, "blank"),
        ("FLAGS(1:1)", 51, 51, "blank = V1, V2, DV, FWHM and tape7.scn in wavenumbers"),
        ("FLAGS(2:2)", 52, 52, "'1' = triangular slit function"),
        ("FLAGS(3:3)", 53, 53, "'A' = FWHM is absolute"),
        ("FLAGS(4:4)", 54, 54, "'A' = all radiance and transmittance components degraded by the slit"),
        ("FLAGS(5:7)", 55, 57, "blank = no saving/re-use of results, no spectral flux file"),
        ("MLFLX", 58, 60, "0 = not used"),
        ("VRFRAC", 61, 70, "0 = not used"),
    ]),
    "CARD 5": ("FORMAT (I5) - manual §18, p.52-53", [
        ("IRPT", 1, 5, "0 = stop after this run"),
    ]),
}
CARD_ORDER = ["CARD 1", "CARD 1A", "CARD 1A2", "CARD 2", "CARD 3", "CARD 4", "CARD 5"]


def ruler(n=80):
    tens = "".join(f"{(i // 10) % 10}" if i % 10 == 0 else " " for i in range(1, n + 1))
    ones = "".join(str(i % 10) for i in range(1, n + 1))
    return tens, ones


def main(cards_dir):
    idx = pd.read_csv(os.path.join(cards_dir, "deck_index.csv"))
    example = idx[idx.role == "aster_spaceborne_nadir"].iloc[0]
    lines = open(os.path.join(cards_dir, example.deck_file), encoding="ascii").read().splitlines()
    L = ["# MODTRAN 5.2 Input Decks (tape5) Used to Build the 30 Atmospheric LUTs", "",
         "This folder contains **every MODTRAN deck that was executed** to produce the atmospheric look-up tables of the "
         "dataset, plus the corresponding `tape7.scn` outputs (`tape7_reference/`, in the pipeline inputs). A user holding "
         "a licensed MODTRAN 5.2 can re-run them and verify that the shipped tables are reproduced.", "",
         "## 1. Inventory", "",
         f"* **{len(idx)} decks** = {idx.profile_id.nunique()} atmospheric profiles x "
         f"{len(idx) // idx.profile_id.nunique()} runs per profile.",
         "* `deck_index.csv`: one row per deck, with tag, file, MD5, profile, role, geometry, quadrature node, weight and "
         "the LUT variable it feeds.", ""]
    roles = idx.groupby("role").agg(n=("tag", "count"), H1=("H1_km", "first"), H2=("H2_km", "first")).reset_index()
    L += ["| Role | Decks | H1 (km) | H2 (km) | Zenith angle(s) | Feeds LUT variable |", "|---|---|---|---|---|---|"]
    for r in roles.itertuples():
        angs = sorted(idx[idx.role == r.role].angle_deg.round(3).unique())
        lv = idx[idx.role == r.role].lut_variable.iloc[0]
        L.append(f"| {r.role} | {r.n} | {r.H1:g} | {r.H2:g} | {', '.join(f'{a:g}' for a in angs)} | {lv} |")
    q = idx[idx.role == "downwelling_quadrature_node"].drop_duplicates("node").sort_values("node")
    L += ["", "**Hemispheric downwelling quadrature (8-node Gauss-Legendre in mu = cos(zenith) on [0, 1]):**", "",
          "| Node | mu | Zenith angle (deg) | Gauss weight w | Radiance weight 2 w mu |", "|---|---|---|---|---|"]
    for r in q.itertuples():
        L.append(f"| {int(r.node)} | {r.mu:.6f} | {r.angle_deg:.4f} | {r.gauss_weight:.6f} | {r.radiance_weight:.6f} |")
    L += ["", f"Sum of radiance weights = {q.radiance_weight.sum():.6f} (= 1 for an isotropic sky).", "",
          "## 2. Fields that vary between decks", "",
          "| Card | Field | Values |", "|---|---|---|",
          "| CARD 1 | MODEL (col 5) | 1-6 |",
          "| CARD 1A | H2OSTR (cols 21-30) | 0.5, 0.75, 1.0, 1.5, 2.0 (water-vapour column scaling) |",
          "| CARD 3 | H1, H2, ANGLE (cols 1-30) | see the role table above |", "",
          "All other fields are identical in every deck. The decks are generated by `build_tape5_deck()` in "
          "`src/data_generation/base/automate_modtran.py` from the template `modtran/05vs15nadirRAD.ltn`.", "",
          f"## 3. Card-by-card description (example deck `{example.deck_file}`)", ""]
    t, o = ruler()
    L += ["```", t, o] + lines + ["```", ""]
    # map deck lines to cards (IEMSCT = 1: no CARD 3A1/3A2)
    card_lines = dict(zip(CARD_ORDER, lines))
    for card in CARD_ORDER:
        fmt, fields = CARD_FIELDS[card]
        raw = card_lines.get(card, "")
        L += [f"### {card}", "", f"`{raw}`", "", f"*{fmt}*", "", "| Field | Columns | Value in deck | Meaning |", "|---|---|---|---|"]
        for name, a, b, meaning in fields:
            val = raw[a - 1:b].strip() if len(raw) >= a else ""
            L.append(f"| {name} | {a}-{b} | `{val if val else '(blank)'}` | {meaning} |")
        L.append("")
    L += ["## 4. From tape7.scn to the LUT columns", "",
          "With ν in cm⁻¹ (`FREQ(CM-1)`), MODTRAN radiances in W cm⁻² sr⁻¹ (cm⁻¹)⁻¹ are converted to W m⁻² sr⁻¹ µm⁻¹ "
          "by L_λ = L_ν · ν². The wavelength is λ = 10⁴/ν µm, and the tables are sorted by ascending wavelength.", "",
          "| LUT column | Source deck(s) | tape7.scn column | Operation |", "|---|---|---|---|",
          "| `tau_aster`, `tau_hytes` | `*__aster_nadir`, `*__hytes_nadir` | `TRAN` | none |",
          "| `L_up_aster`, `L_up_hytes` | same | `PTH_THRML` | × ν² |",
          "| `L_down_zenith` | `*__down_zenith` | `TOTAL_RAD` | × ν² |",
          "| `L_down_surface` (= F↓/π) | `*__down_q1..q8` | `TOTAL_RAD` | Σ_q (2 w_q μ_q) · TOTAL_RAD_q · ν² |",
          "| `pwv_g_cm2` (catalog) | `*__down_q1` | header line `H2O & O3 COLUMNS` | 4th number |", "",
          "## 5. How to re-run and verify (licensed MODTRAN 5.2 required)", "",
          "**Automatic (recommended):** from the pipeline folder,", "```bash",
          "set OPENTIR_MODTRAN_DIR=C:\\PcModWin5\\bin        # folder containing Mod5.2.0.0.exe and DATA/",
          "python run_pipeline.py --stages setup atmosphere --run-modtran", "```",
          "This regenerates all 30 LUTs in `data/validation/lut_replication/`, archives the executed decks, and writes "
          "`lut_replication_report.csv`. The comparison with the shipped LUTs uses |a − b| ≤ 10⁻⁶ + 10⁻⁴·|a| per spectral point. "
          "The shipped LUTs are never overwritten.", "",
          "**Manual (any single deck):** copy a deck to the MODTRAN working folder as `tape5`, run `Mod5.2.0.0.exe`, and "
          "compare the resulting `tape7.scn` with `tape7_reference/<tag>.7sc`.", "",
          "## 6. Modelling choices and known limitations", "",
          "* Thermal radiance only (IEMSCT = 1). The measured solar contribution to the 8-12 µm sky radiance is "
          "0.002-0.17 % (`scripts/modtran_downwelling_check2.py`).",
          "* DISORT 8-stream multiple scattering. Running without multiple scattering overestimates sky radiance by 2-28 % "
          "(`scripts/modtran_ms_check.py`).",
          "* Lower-boundary albedo 0.06 (= 1 − mean 8-12 µm emissivity of the development library) and boundary "
          "temperature = first-level air temperature. The scattered-surface-emission part of `PTH_THRML` couples weakly "
          "to the pixel emissivity: about −0.33 K for ε ≈ 0.5 in the driest atmosphere.",
          "* **Aerosol: IHAZE = 2, rural extinction with the default 5 km meteorological range** (a hazy boundary layer).",
          "* **CO₂ = 300 ppmv**, written into CARD 1A by `build_tape5_deck()` (the template carries the MODTRAN default 330 ppmv; "
          "MODTRAN 5.2.1 manual p. 17). The present-day value is 424.61 ppm (Mauna Loa 2024 annual mean, NOAA/GML and Scripps). "
          "Measured with `scripts/co2_sensitivity_check.py` (300 vs 424.61 ppmv, same RT configuration), the LST bias from correcting a "
          "present-day observation with these LUTs is: median 0.02-0.30 K over 8-12 µm; worst single bands 0.15-0.21 K "
          "(ASTER B13, ε = 0.95) and up to ≈ 1.1-1.3 K for ε = 0.7 near 8.3 µm (ASTER B10, HyTES). The dataset is internally "
          "consistent (all partitions share the same physics). For real-data use, set CO2MX to the scene value and rerun "
          "`python run_pipeline.py --run-modtran` (details: `docs/perturbation_model_error_sources.md` §3).",
          "* **H₂O saturation cap (v1.3.0 finding).** Scaling the water column (H2OSTR) cannot exceed 100 % relative humidity: "
          "MODTRAN redistributes water and, when no layer can take more, caps the column and reports in `tape6` "
          "\"INPUT WATER COLUMN ... IS ABOVE MAXIMUM ALLOWED. THE WATER COLUMN IS BEING SET TO THE MAXIMUM\" (manual p. 17). "
          "The `tape7` header keeps the requested column. Capped profiles (H₂O × 2.0): atm1 8.23 → 6.77, atm2 5.84 → 5.35, "
          "atm3 1.70 → 1.37, atm4 4.16 → 3.13, atm5 0.83 → 0.63 g/cm² (requested → actual). `automate_modtran.py` now reads "
          "the actual column from `tape6`; `lut_catalog.csv` stores `pwv_g_cm2` (actual), `pwv_requested_g_cm2` and "
          "`h2o_capped_at_saturation`. Radiative arrays are unchanged (decks identical). atm4 and atm5 at × 2.0 are "
          "radiatively almost identical to × 1.5 (actual columns differ by < 0.01 g/cm²).",
          "* 1 cm⁻¹ band model (`01_2009`), triangular slit with FWHM = 2 cm⁻¹ and 1 cm⁻¹ sampling, 660-1435 cm⁻¹.", ""]
    out = os.path.join(cards_dir, "README_MODTRAN_CARDS.md")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])

