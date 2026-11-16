"""
Automated MODTRAN 5.2 Atmospheric Profile Generator for OpenTIR-PINN v1.1.2
========================================================================

Executes PCModWin 5.2.0.0 via exact character-aligned Fortran card decks
over a multi-condition grid:
- 6 Standard Atmospheric Models (Tropical, MLS, MLW, SAS, SAW, US Standard)
- 5 Water Vapor Scaling Factors (0.5, 0.75, 1.0, 1.5, 2.0)
- 3 Radiative Geometries:
  * ASTER Spaceborne TOA (100.0 km -> 0.0 km nadir)
  * HyTES Airborne (20.0 km -> 0.0 km nadir)
  * Ground Downwelling Sky Radiance (0.02 km -> 100.0 km), integrated over the sky hemisphere
    (8-point Gauss-Legendre quadrature in mu = cos(zenith)); the zenith radiance is kept for provenance

Computes (thermal-only IEMSCT = 1; DISORT 8-stream multiple scattering; surface albedo 0.06):
- Spectral transmittance tau(lambda)
- At-sensor upwelling path radiance L_up(lambda) in W/(m^2 sr um)
- At-surface hemispheric downwelling sky radiance L_down = F_down/pi in W/(m^2 sr um)

Ensures zero MD5 collision, extracts true Precipitable Water Vapor (PWV),
and partitions profiles into Development vs. Golden Quarantined regimes.
"""

import os
import re
import subprocess
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# MODTRAN 5.2 is licensed software and is NOT redistributed. Point OPENTIR_MODTRAN_DIR to a local
# PcModWin5 'bin' folder to regenerate the LUTs; otherwise the pipeline uses the shipped LUTs.
MODTRAN_DIR = os.environ.get("OPENTIR_MODTRAN_DIR", r"C:\PcModWin5\bin")
MODTRAN_EXE = os.environ.get("OPENTIR_MODTRAN_EXE", "Mod5.2.0.0.exe")
_REPO_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "modtran", "05vs15nadirRAD.ltn")
TEMPLATE_PATH = os.environ.get("OPENTIR_MODTRAN_TEMPLATE",
                               os.path.abspath(_REPO_TEMPLATE) if os.path.exists(_REPO_TEMPLATE)
                               else r"C:\PcModWin5\usr\05vs15nadirRAD.ltn")

ATM_MODELS = {
    1: "Tropical",
    2: "Mid-Latitude Summer",
    3: "Mid-Latitude Winter",   # Quarantined
    4: "Sub-Arctic Summer",
    5: "Sub-Arctic Winter",   # Quarantined
    6: "1976 US Standard"
}

WATER_VAPOR_SCALES = [0.5, 0.75, 1.0, 1.5, 2.0]

QUARANTINED_MODELS = {3, 5}

# v1.1.2 radiative-transfer configuration (see clarification.md Entry 38 and
# scripts/modtran_downwelling_experiment.py / modtran_downwelling_check2.py):
#  * IEMSCT = 1 (thermal radiance, no sun): the forward model has no solar term; the solar share of
#    the 8-12 um sky radiance was measured at 0.002-0.17 %.
#  * IMULT = 1 with DISORT, 8 streams (scripts/modtran_ms_check.py): with IMULT = 0 MODTRAN lets the
#    whole aerosol extinction emit, overestimating 8-12 um sky radiance by 2-28 % (dry > humid); the
#    Isaacs two-stream and DISORT-8 solutions agree within ~2 %, so MS is kept and the more accurate
#    DISORT solver is used.
#  * SURREF = 0.06: lower-boundary albedo of the MS calculation = 1 - mean development-pool emissivity
#    in 8-12 um (0.935). The v1.1.1 template value 0.40 (emissivity 0.60) made the scattered-surface-
#    emission part of PTH_THRML unrepresentative of natural surfaces.
#  * L_down = hemispheric sky radiance F_down/pi = 2 * int_0^1 L(mu) mu dmu (Lambertian surface),
#    evaluated by Gauss-Legendre quadrature in mu = cos(zenith) over N_DOWN_QUAD slant sky paths
#    (8 vs 16 nodes agree within 0.05 %); the v1.1.1 zenith-only radiance underestimated it by 17-41 %.
RT_IEMSCT = 1
RT_IMULT = 1
RT_DIS = "t"
RT_NSTR = 8
RT_SURREF = 0.06
N_DOWN_QUAD = 8

# When set (generate_atmospheric_grid(..., archive_dir=...)), every executed tape5 deck and its tape7.scn
# output are archived with a descriptive tag, and an index row is recorded (deck_index.csv).
DECK_ARCHIVE = None

# Water column actually used by MODTRAN in the last run. When the scaled H2O column exceeds 100 % relative humidity
# in every layer that can absorb more, MODTRAN caps it and reports in tape6: "INPUT WATER COLUMN, x GM / CM2, IS ABOVE
# MAXIMUM ALLOWED. THE WATER COLUMN IS BEING SET TO THE MAXIMUM, y GM / CM2" (MODTRAN 5.2.1 manual p. 17, H2OSTR).
# The tape7 header keeps the *requested* column x, so the actual column y must be read from tape6 (v1.3.0 fix).
LAST_RUN_INFO = {"pwv_requested": None, "pwv_actual": None, "h2o_capped": False}
_CAP_RE = re.compile(r"WATER COLUMN IS BEING SET TO THE MAXIMUM,\s*([0-9.]+)\s*GM", re.I)


def hemispheric_downwelling(template_lines: List[str], model_idx: int, h2o_scale: float,
                            n_quad: int = N_DOWN_QUAD, profile_id: Optional[str] = None
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Hemispheric downwelling sky radiance at the surface for a Lambertian reflector,
    L_down = F_down / pi = 2 * int_0^1 L_sky(mu) * mu dmu, by n_quad-point Gauss-Legendre quadrature.

    Returns (wavenumber_cm1, L_down_hemispheric, L_down_zenith) in W/(m^2 sr um) and the PWV (g/cm^2).
    """
    x, w = np.polynomial.legendre.leggauss(n_quad)
    mu = 0.5 * (x + 1.0)
    wq = 0.5 * w
    acc, nu_ref, pwv = None, None, 0.0
    for q, (m_i, w_i) in enumerate(zip(mu, wq), start=1):
        ang = float(np.degrees(np.arccos(m_i)))
        deck = build_tape5_deck(template_lines, model_idx, h2o_scale, h1=0.02, h2=100.0,
                                angle=ang, iemsct=RT_IEMSCT, imult=RT_IMULT, dis=RT_DIS, nstr=RT_NSTR, surref=RT_SURREF)
        df, pwv = run_modtran_deck(deck, tag=f"{profile_id}__down_q{q}" if profile_id else None, meta=dict(
            profile_id=profile_id, role="downwelling_quadrature_node", node=q, H1_km=0.02, H2_km=100.0, angle_deg=ang,
            mu=float(m_i), gauss_weight=float(w_i), radiance_weight=float(2.0 * w_i * m_i),
            column_used="TOTAL_RAD", lut_variable="L_down_surface (sum of radiance_weight * TOTAL_RAD * nu^2)"))
        nu = df['FREQ(CM-1)'].values
        rad = df['TOTAL_RAD'].values * (nu ** 2)
        if nu_ref is None:
            nu_ref, acc = nu, np.zeros_like(rad)
        elif not np.array_equal(nu, nu_ref):
            raise RuntimeError("Spectral grid changed between quadrature nodes")
        acc += 2.0 * w_i * m_i * rad
    deck_z = build_tape5_deck(template_lines, model_idx, h2o_scale, h1=0.02, h2=100.0, angle=0.0,
                              iemsct=RT_IEMSCT, imult=RT_IMULT, dis=RT_DIS, nstr=RT_NSTR, surref=RT_SURREF)
    df_z, _ = run_modtran_deck(deck_z, tag=f"{profile_id}__down_zenith" if profile_id else None, meta=dict(
        profile_id=profile_id, role="downwelling_zenith", H1_km=0.02, H2_km=100.0, angle_deg=0.0,
        column_used="TOTAL_RAD", lut_variable="L_down_zenith (TOTAL_RAD * nu^2)"))
    zen = df_z['TOTAL_RAD'].values * (df_z['FREQ(CM-1)'].values ** 2)
    return nu_ref, acc, zen, pwv


def build_tape5_deck(
    template_lines: List[str],
    model_idx: int,
    h2o_scale: float,
    h1: float,
    h2: float,
    angle: float,
    v1: float = 660.0,
    v2: float = 1435.0,
    dv: float = 1.0,
    fwhm: float = 2.0,
    iemsct: Optional[int] = None,
    imult: Optional[int] = None,
    dis: Optional[str] = None,
    nstr: Optional[int] = None,
    surref: Optional[float] = None
) -> List[str]:
    """
    Constructs a valid character-exact MODTRAN 5.2 tape5 deck with 1 cm^-1 resolution.

    iemsct / imult (CARD 1, cols 11-15 / 16-20) override the template when given. With
    IEMSCT = 1 (thermal radiance, no sun) the solar-geometry CARDs 3A1/3A2 are removed,
    because MODTRAN reads them only when IEMSCT = 2 or 4 (MODTRAN 5.2.1 manual, CARD list).
    """
    lines = list(template_lines)

    # Card 1 (L0): Col 5 is MODEL (1 to 6)
    l0 = list(lines[0])
    l0[4] = str(model_idx)        # Atmospheric model index (1-6)
    lines[0] = "".join(l0)
    if iemsct is not None or imult is not None:
        s = lines[0]
        if iemsct is not None:
            s = s[:10] + f"{iemsct:5d}" + s[15:]
        if imult is not None:
            s = s[:15] + f"{imult:5d}" + s[20:]
        lines[0] = s
    if surref is not None:
        # CARD 1 FORMAT (..., F8.0, A7): TPTEMP cols 66-73, SURREF cols 74-80
        lines[0] = lines[0].rstrip("\n").ljust(73)[:73] + f"{surref:7.2f}" + "\n"

    # Card 1A (L1): Water vapor scaling factor at index 23..32. CO2MX (cols 11-20) = 300 ppmv is deliberate and
    # frozen for reproducibility of the published LUTs (template/MODTRAN default: 330; present day ~425 ppm).
    # Measured impact and remedy: docs/perturbation_model_error_sources.md section 3, scripts/co2_sensitivity_check.py.
    lines[1] = f"fFF  2   0   300.000{h2o_scale:10.6f}  1.0000000F T F F F               0.000     0.000     0.000     0.000         0\n"
    if dis is not None or nstr is not None:
        # CARD 1A FORMAT (3A1, I3, ...): DIS col 1, NSTR cols 4-6
        d = dis if dis is not None else lines[1][0]
        n = nstr if nstr is not None else int(lines[1][3:6])
        lines[1] = d + lines[1][1:3] + f"{n:3d}" + lines[1][6:]

    # Card 1A1 (L2): Specify 01_2009 1 cm^-1 molecular database
    lines[2] = "01_2009\n"

    # Card 3 (L4): Observer and target geometry (H1, H2, ANGLE)
    lines[4] = f"{h1:10.3f}{h2:10.3f}{angle:10.3f}   0.00000     0.000     0.000    0          0.000     0.000\n"

    # Card 4 (L7): Spectral range (V1, V2, DV, FWHM)
    v1_str = f"{v1:10.3f}"
    v2_str = f"{v2:10.3f}"
    dv_str = f"{dv:10.3f}"
    fwhm_str = f"{fwhm:10.3f}"
    rest = lines[7][40:]
    lines[7] = v1_str + v2_str + dv_str + fwhm_str + rest

    if iemsct == 1:
        del lines[5:7]   # drop CARD 3A1 / 3A2 (solar geometry)

    return lines


def run_modtran_deck(lines: List[str], tag: Optional[str] = None, meta: Optional[Dict] = None) -> Tuple[pd.DataFrame, float]:
    """
    Writes tape5 to MODTRAN directory, executes MODTRAN 5.2, and parses tape7.scn.
    If DECK_ARCHIVE is active and a tag is given, the exact deck and its tape7.scn are archived.
    Returns: (DataFrame with spectral results, PWV in g/cm^2).
    """
    target_tape5 = os.path.join(MODTRAN_DIR, "tape5")
    target_tape7_scn = os.path.join(MODTRAN_DIR, "tape7.scn")
    target_tape7_std = os.path.join(MODTRAN_DIR, "tape7")
    target_tape6 = os.path.join(MODTRAN_DIR, "tape6")

    with open(target_tape5, "w", encoding="ascii") as f:
        f.writelines(lines)

    for p in [target_tape7_scn, target_tape7_std, target_tape6]:
        if os.path.exists(p):
            os.remove(p)

    exe_full_path = os.path.join(MODTRAN_DIR, MODTRAN_EXE)
    res = subprocess.run([exe_full_path], cwd=MODTRAN_DIR, capture_output=True, text=True)

    if os.path.exists(target_tape7_scn):
        actual_output = target_tape7_scn
    elif os.path.exists(target_tape7_std):
        actual_output = target_tape7_std
    else:
        raise RuntimeError(f"MODTRAN failed with code {res.returncode}. Stderr: {res.stderr.strip()[:200]}")

    with open(actual_output, "r", encoding="ascii", errors="ignore") as f:
        t7_raw = f.readlines()

    if DECK_ARCHIVE is not None and tag:
        import shutil
        deck_file = os.path.join(DECK_ARCHIVE["dir"], "tape5", f"{tag}.tp5")
        out_file = os.path.join(DECK_ARCHIVE["dir"], "tape7_reference", f"{tag}.7sc")
        with open(deck_file, "w", encoding="ascii", newline="\n") as f:
            f.writelines(lines)
        shutil.copy(actual_output, out_file)
        DECK_ARCHIVE["index"].append(dict(tag=tag, deck_file=f"tape5/{tag}.tp5", reference_output=f"tape7_reference/{tag}.7sc",
                                          deck_md5=hashlib.md5(open(deck_file, "rb").read()).hexdigest().upper(),
                                          **(meta or {})))

    # Parse PWV from header (line index 3: Card 1A H2O column)
    pwv_val = 0.0
    for l in t7_raw[:10]:
        if "H2O & O3 COLUMNS" in l:
            parts = l.split("!")[0].split()
            if len(parts) >= 4:
                try:
                    pwv_val = float(parts[3])
                except ValueError:
                    pass
            break

    # Actual water column: tape6 reports the saturation cap (the tape7 header keeps the requested value)
    LAST_RUN_INFO.update(pwv_requested=pwv_val, pwv_actual=pwv_val, h2o_capped=False)
    if os.path.exists(target_tape6):
        m = _CAP_RE.search(open(target_tape6, "r", encoding="ascii", errors="ignore").read())
        if m:
            LAST_RUN_INFO.update(pwv_actual=float(m.group(1)), h2o_capped=True)
            pwv_val = float(m.group(1))

    # Find table header
    header_idx = -1
    for i, l in enumerate(t7_raw):
        if "FREQ(CM-1)" in l:
            header_idx = i
            break

    if header_idx == -1:
        raise ValueError("Could not find table header in tape7.scn")

    cols = t7_raw[header_idx].strip().split()
    # Fixed-width parsing: columns are right-aligned under their header labels. In thermal-only mode
    # (IEMSCT = 1) the solar columns are left blank, so whitespace splitting would misalign them.
    header = t7_raw[header_idx].rstrip("\n")
    ends, starts, pos = [], [], 0
    for name in cols:
        s = header.index(name, pos)
        e = s + len(name)
        starts.append(s); ends.append(e); pos = e
    bounds = [(0 if i == 0 else ends[i - 1], ends[i]) for i in range(len(cols))]
    data = []
    for l in t7_raw[header_idx + 1:]:
        if not l.strip() or l.strip().startswith("-9999"):
            break
        row = []
        for (a, b) in bounds:
            tok = l[a:b].strip()
            try:
                row.append(float(tok) if tok else np.nan)
            except ValueError:
                row = None
                break
        if row is not None and not np.isnan(row[0]):
            data.append(row)

    df = pd.DataFrame(data, columns=cols)
    return df, pwv_val


def _profile_id(model_idx: int, h2o_scale: float) -> str:
    return f"atm{model_idx}_h2o{h2o_scale:.2f}" if abs(round(h2o_scale, 2) - h2o_scale) < 1e-9 else f"atm{model_idx}_h2o{h2o_scale:.3f}"


def generate_atmospheric_grid(output_dir: str, archive_dir: Optional[str] = None, models: Optional[List[int]] = None,
                              scales: Optional[List[float]] = None):
    """
    Generates all 30 atmospheric profiles across ASTER, HyTES, and Downwelling geometries.
    If archive_dir is given, every executed tape5 deck, its tape7.scn and a deck index are archived there.
    """
    global DECK_ARCHIVE
    os.makedirs(output_dir, exist_ok=True)
    if archive_dir:
        os.makedirs(os.path.join(archive_dir, "tape5"), exist_ok=True)
        os.makedirs(os.path.join(archive_dir, "tape7_reference"), exist_ok=True)
        DECK_ARCHIVE = {"dir": archive_dir, "index": []}

    with open(TEMPLATE_PATH, "r", encoding="ascii") as f:
        template_lines = f.readlines()

    catalog_records = []
    model_items = [(m, n) for m, n in ATM_MODELS.items() if models is None or m in models]
    scale_list = list(scales) if scales is not None else WATER_VAPOR_SCALES
    total_profiles = len(model_items) * len(scale_list)
    profile_counter = 0

    print(f"================================================================")
    print(f"Generating {total_profiles} Atmospheric Profiles with MODTRAN 5.2")
    print(f"Target Directory: {output_dir}")
    print(f"================================================================")

    for model_idx, model_name in model_items:
        is_quarantined = model_idx in QUARANTINED_MODELS
        quarantine_tag = "golden_quarantined" if is_quarantined else "development"

        for h2o_scale in scale_list:
            profile_counter += 1
            profile_id = _profile_id(model_idx, h2o_scale)
            print(f"[{profile_counter:02d}/{total_profiles}] Processing {profile_id} ({model_name}, H2O={h2o_scale}x, {quarantine_tag})...")

            # 1. Downwelling: hemispheric sky radiance F_down/pi by Gauss-Legendre quadrature (v1.1.2)
            nu_down, l_down_surface, l_down_zenith, pwv_measured = hemispheric_downwelling(
                template_lines, model_idx, h2o_scale, profile_id=profile_id)
            pwv_requested, h2o_capped = LAST_RUN_INFO["pwv_requested"], LAST_RUN_INFO["h2o_capped"]
            if h2o_capped:
                print(f"    H2O capped at saturation: requested {pwv_requested:.4f} -> actual {pwv_measured:.4f} g/cm2")

            # 2. Run ASTER TOA (space looking nadir: H1=100.0km, H2=0.0km, angle=180.0)
            deck_aster = build_tape5_deck(template_lines, model_idx, h2o_scale, h1=100.0, h2=0.0, angle=180.0,
                                          iemsct=RT_IEMSCT, imult=RT_IMULT, dis=RT_DIS, nstr=RT_NSTR, surref=RT_SURREF)
            df_aster, _ = run_modtran_deck(deck_aster, tag=f"{profile_id}__aster_nadir", meta=dict(
                profile_id=profile_id, role="aster_spaceborne_nadir", H1_km=100.0, H2_km=0.0, angle_deg=180.0,
                column_used="TRAN, PTH_THRML", lut_variable="tau_aster (TRAN), L_up_aster (PTH_THRML * nu^2)"))

            # 3. Run HyTES Airborne (aircraft looking nadir: H1=20.0km, H2=0.0km, angle=180.0)
            deck_hytes = build_tape5_deck(template_lines, model_idx, h2o_scale, h1=20.0, h2=0.0, angle=180.0,
                                          iemsct=RT_IEMSCT, imult=RT_IMULT, dis=RT_DIS, nstr=RT_NSTR, surref=RT_SURREF)
            df_hytes, _ = run_modtran_deck(deck_hytes, tag=f"{profile_id}__hytes_nadir", meta=dict(
                profile_id=profile_id, role="hytes_airborne_nadir", H1_km=20.0, H2_km=0.0, angle_deg=180.0,
                column_used="TRAN, PTH_THRML", lut_variable="tau_hytes (TRAN), L_up_hytes (PTH_THRML * nu^2)"))

            # Conversion from MODTRAN W/(cm^2 sr cm^-1) to W/(m^2 sr um): L_lambda = L_nu * nu^2
            nu = df_aster['FREQ(CM-1)'].values
            if not (np.array_equal(nu, nu_down) and np.array_equal(nu, df_hytes['FREQ(CM-1)'].values)):
                raise RuntimeError("Spectral grids differ between geometries")
            wavelength_um = 10000.0 / nu

            # ASTER transmission and upwelling path radiance
            tau_aster = df_aster['TRAN'].values
            l_up_aster = df_aster['PTH_THRML'].values * (nu ** 2)

            # HyTES transmission and upwelling path radiance
            tau_hytes = df_hytes['TRAN'].values
            l_up_hytes = df_hytes['PTH_THRML'].values * (nu ** 2)

            # Build consolidated profile DataFrame (sorted ascending in wavelength)
            profile_df = pd.DataFrame({
                'wavelength_um': wavelength_um,
                'wavenumber_cm1': nu,
                'tau_aster': tau_aster,
                'L_up_aster': l_up_aster,
                'tau_hytes': tau_hytes,
                'L_up_hytes': l_up_hytes,
                'L_down_surface': l_down_surface,
                'L_down_zenith': l_down_zenith
            }).sort_values('wavelength_um').reset_index(drop=True)

            csv_filename = f"profile_{profile_id}.csv"
            npz_filename = f"profile_{profile_id}.npz"
            csv_path = os.path.join(output_dir, csv_filename)
            npz_path = os.path.join(output_dir, npz_filename)

            profile_df.to_csv(csv_path, index=False)
            np.savez_compressed(
                npz_path,
                wavelength_um=profile_df['wavelength_um'].values,
                wavenumber_cm1=profile_df['wavenumber_cm1'].values,
                tau_aster=profile_df['tau_aster'].values,
                L_up_aster=profile_df['L_up_aster'].values,
                tau_hytes=profile_df['tau_hytes'].values,
                L_up_hytes=profile_df['L_up_hytes'].values,
                L_down_surface=profile_df['L_down_surface'].values,
                L_down_zenith=profile_df['L_down_zenith'].values,
                downwelling_method=f"hemispheric F_down/pi, {N_DOWN_QUAD}-pt Gauss-Legendre in mu",
                rt_config=f"IEMSCT={RT_IEMSCT}, IMULT={RT_IMULT}, DIS={RT_DIS}, NSTR={RT_NSTR}, SURREF={RT_SURREF}",
                model_idx=model_idx,
                h2o_scale=h2o_scale,
                pwv_g_cm2=pwv_measured,
                pwv_requested_g_cm2=pwv_requested,
                h2o_capped_at_saturation=h2o_capped,
                quarantine_status=quarantine_tag
            )

            # Compute MD5 of CSV for cryptographic verification
            with open(csv_path, "rb") as f:
                md5_val = hashlib.md5(f.read()).hexdigest().upper()

            catalog_records.append({
                'profile_id': profile_id,
                'atm_model_idx': model_idx,
                'atm_model_name': model_name,
                'h2o_scale': h2o_scale,
                'pwv_g_cm2': round(pwv_measured, 4),
                'pwv_requested_g_cm2': round(pwv_requested, 4),
                'h2o_capped_at_saturation': bool(h2o_capped),
                'quarantine_status': quarantine_tag,
                'num_spectral_channels': len(profile_df),
                'min_wavelength_um': round(float(profile_df['wavelength_um'].min()), 3),
                'max_wavelength_um': round(float(profile_df['wavelength_um'].max()), 3),
                'md5_hash': md5_val,
                'csv_file': csv_filename,
                'npz_file': npz_filename
            })

    catalog_df = pd.DataFrame(catalog_records)
    catalog_path = os.path.join(output_dir, "lut_catalog.csv")
    catalog_df.to_csv(catalog_path, index=False)

    # Verification: Assert all 30 MD5 hashes are strictly unique
    unique_hashes = catalog_df['md5_hash'].nunique()
    print(f"\n================================================================")
    print(f"Atmospheric Grid Generation Completed!")
    print(f"Total Profiles Created: {len(catalog_df)}")
    print(f"Unique MD5 Checksums:   {unique_hashes} / {len(catalog_df)}")
    if unique_hashes != len(catalog_df):
        raise AssertionError(f"CRITICAL DEFECT: Duplicate MD5 hashes detected in atmospheric LUTs!")
    print(f"Verification: 100% Unique Physical Profiles Confirmed!")
    print(f"Catalog saved to: {catalog_path}")
    if DECK_ARCHIVE is not None:
        idx = pd.DataFrame(DECK_ARCHIVE["index"])
        idx = idx.merge(catalog_df[["profile_id", "atm_model_idx", "atm_model_name", "h2o_scale", "quarantine_status"]],
                        on="profile_id", how="left")
        idx.to_csv(os.path.join(DECK_ARCHIVE["dir"], "deck_index.csv"), index=False)
        print(f"Archived {len(idx)} tape5 decks + tape7.scn outputs to: {DECK_ARCHIVE['dir']}")
        DECK_ARCHIVE = None


LUT_VARIABLES = ["tau_aster", "L_up_aster", "tau_hytes", "L_up_hytes", "L_down_surface", "L_down_zenith"]


def compare_lut_dirs(reference_dir: str, candidate_dir: str, rel_tol: float = 1e-4, abs_tol: float = 1e-6) -> pd.DataFrame:
    """
    Compare two LUT directories profile by profile. A value matches if |a-b| <= abs_tol + rel_tol*|a|.
    Returns one row per (profile, variable) with max absolute / relative differences and a pass flag.
    """
    rows = []
    cat = pd.read_csv(os.path.join(reference_dir, "lut_catalog.csv"))
    for pid in cat["profile_id"]:
        a = np.load(os.path.join(reference_dir, f"profile_{pid}.npz"), allow_pickle=True)
        b = np.load(os.path.join(candidate_dir, f"profile_{pid}.npz"), allow_pickle=True)
        for v in LUT_VARIABLES:
            if v not in a.files or v not in b.files:
                rows.append(dict(profile_id=pid, variable=v, max_abs_diff=np.nan, max_rel_diff=np.nan, passed=False))
                continue
            x, y = a[v].astype(float), b[v].astype(float)
            d = np.abs(x - y)
            rel = d / np.maximum(np.abs(x), 1e-12)
            rows.append(dict(profile_id=pid, variable=v, max_abs_diff=float(d.max()), max_rel_diff=float(rel.max()),
                             passed=bool(np.all(d <= abs_tol + rel_tol * np.abs(x)))))
        rows.append(dict(profile_id=pid, variable="pwv_g_cm2", max_abs_diff=float(abs(a["pwv_g_cm2"] - b["pwv_g_cm2"])),
                         max_rel_diff=np.nan, passed=bool(abs(a["pwv_g_cm2"] - b["pwv_g_cm2"]) <= 1e-4)))
    return pd.DataFrame(rows)
    print(f"================================================================")


if __name__ == "__main__":
    import argparse
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    ap = argparse.ArgumentParser(description="Generate (and optionally archive / verify) the 30 MODTRAN 5.2 LUTs")
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "processed", "atmospheric_luts"))
    ap.add_argument("--archive-decks", default=None, help="folder receiving every tape5 deck, tape7.scn and deck_index.csv")
    ap.add_argument("--compare-to", default=None, help="reference LUT folder; writes lut_replication_report.csv in --out")
    ap.add_argument("--rel-tol", type=float, default=1e-4)
    ap.add_argument("--models", type=int, nargs="+", default=None, help="subset of MODTRAN models (validation grids)")
    ap.add_argument("--scales", type=float, nargs="+", default=None, help="H2O scales (default: the 5 dataset scales)")
    args = ap.parse_args()
    generate_atmospheric_grid(args.out, archive_dir=args.archive_decks, models=args.models, scales=args.scales)
    if args.compare_to:
        rep = compare_lut_dirs(args.compare_to, args.out, rel_tol=args.rel_tol)
        rep.to_csv(os.path.join(args.out, "lut_replication_report.csv"), index=False)
        print(f"LUT replication: {int(rep.passed.sum())}/{len(rep)} checks within tolerance; "
              f"max rel diff = {np.nanmax(rep.max_rel_diff.values):.3e}")
        if not rep.passed.all():
            raise SystemExit(1)
