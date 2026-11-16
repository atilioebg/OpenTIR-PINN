"""
Rebalanced Spectral Library Curator for OpenTIR-PINN v1.1.2
===========================================================

Mines, parses, converts, and balances intrinsic terrestrial spectral emissivity
from NASA/ECOSTRESS (ecospeclib-all):
- 145 Geological, Pedological, and Anthropogenic endmembers:
  * Natural Soils (68)
  * Sandstones (21)
  * Basalts (12)
  * Granites (8)
  * Quartz (19)
  * Calcite (5)
  * Kaolinite (4)
  * Urban Concrete (5)
  * Urban Asphalt (3)
- Subsamples 1,162 botanical spectra to ~100 diverse botanical species:
  * Deciduous trees, conifers, shrubs, grasses, and non-photosynthetic bark
- Uniform high-resolution spectral grid: 7.50 to 13.50 um (step 0.005 um, 1201 channels)
- Implements Kirchhoff's Law conversion: epsilon(lambda) = 1 - rho(lambda)
- Identifies and isolates the 7 Golden Targets + same-sample-ID siblings
- Outputs clean CSV, NPZ, and metadata catalog for StratifiedGroupKFold partitioning.
"""

import os
import glob
import re
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Grid parameters
WAVELENGTH_MIN = 7.50
WAVELENGTH_MAX = 13.50
WAVELENGTH_STEP = 0.005
GRID_WAVELENGTHS = np.round(np.arange(WAVELENGTH_MIN, WAVELENGTH_MAX + WAVELENGTH_STEP / 2.0, WAVELENGTH_STEP), 4)

# Exact 7 Primary Golden Target Benchmark Files
GOLDEN_TARGET_PRIMARY_FILES = {
    "quartz": "mineral.silicate.tectosilicate.coarse.tir.quartz_1.jhu.nicolet.spectrum.txt",
    "calcite": "mineral.carbonate.none.coarse.tir.calcite_1.jhu.nicolet.spectrum.txt",
    "kaolinite": "mineral.silicate.phyllosilicate.coarse.tir.kaolinite_2.jhu.nicolet.spectrum.txt",
    "feldspar": "mineral.silicate.tectosilicate.coarse.tir.anorthite_1.jhu.nicolet.spectrum.txt",
    "urban_concrete": "manmade.concrete.constructionconcrete.solid.all.0598uuucnc.jhu.becknic.spectrum.txt",
    "urban_asphalt": "manmade.road.pavingasphalt.solid.all.0095uuuasp.jhu.becknic.spectrum.txt",
    "npv_bark": "nonphotosyntheticvegetation.bark.abies.concolor.tir.vh311.ucsb.nicolet.spectrum.txt"
}

GOLDEN_SAMPLE_IDS = {"quartz_1", "calcite_1", "kaolinite_2", "anorthite_1", "0598uuucnc", "0095uuuasp", "vh311"}

# v1.1.2 leakage controls (PI decision 2026-11-05, clarification.md Entry 38)
# Spectral indistinguishability radius: an emissivity RMSE of 0.005 over 7.5-13.5 um corresponds to
# ~0.3 K of brightness temperature at 300 K (~3x the 0.1 K NEdT), i.e. two spectra closer than this are
# not radiometrically separable at ~3 sigma. Used for (a) spectral-sibling quarantine of golden targets,
# (b) similarity grouping in the CV split (make_split.py), (c) the validator thresholds.
SPECTRAL_TWIN_RMSE = 0.005
# Explicit specimen-series siblings approved by the PI (consecutive JHU paving-asphalt specimen,
# RMSE 0.0055 to golden 0095UUUASP, inside the training augmentation envelope).
EXPLICIT_SIBLING_SAMPLE_IDS = {"0096uuuasp"}


# Geological and urban targets with respective keywords
TARGET_DEFS = [
    {"class": "soil", "keywords": ["soil"], "superclass": "pedological"},
    {"class": "sandstone", "keywords": ["sandstone", "greywacke"], "superclass": "geological"},
    {"class": "basalt", "keywords": ["basalt"], "superclass": "geological"},
    {"class": "granite", "keywords": ["granite"], "superclass": "geological"},
    {"class": "quartz", "keywords": ["quartz"], "superclass": "geological"},
    {"class": "calcite", "keywords": ["calcite"], "superclass": "geological"},
    {"class": "kaolinite", "keywords": ["kaolinite"], "superclass": "geological"},
    {"class": "feldspar", "keywords": ["feldspar", "albite", "anorthite", "orthoclase", "microcline", "plagioclase"], "superclass": "geological"},
    {"class": "urban_concrete", "keywords": ["concrete"], "superclass": "urban"},
    {"class": "urban_asphalt", "keywords": ["asphalt"], "superclass": "urban"}
]

BOTANICAL_KEYWORDS = ["vegetation", "tree", "leaf", "grass", "shrub", "bark", "conifer", "abies", "pinus", "quercus"]


def parse_ecostress_spectrum(filepath: str) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
    """
    Parses an ECOSTRESS / ASTER .spectrum.txt file with strict 3-tier validation:
      Gate 1: Spectral coverage in TIR (8.0 - 12.0 um) and minimum channel count.
      Gate 2: Explicit rejection of transmission/transmittance (KBr pellets) to protect Kirchhoff's Law.
      Gate 3: Physical terrestrial emissivity bounds validation (eps in [0.40, 1.00]).

    Returns: (raw_wavelengths_um, raw_emissivity, metadata_dict) or None if unusable.
    """
    meta = {
        'sample_id': 'unknown',
        'name': 'unknown',
        'type': 'unknown',
        'class': 'unknown',
        'particle_size': 'unknown',
        'measurement': 'unknown',
        'units': 'percent',
        'source_file': os.path.basename(filepath)
    }

    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()
    except Exception:
        return None

    data_start = -1
    for i, line in enumerate(lines):
        ll = line.lower().strip()
        if ll.startswith("sample no.") or ll.startswith("sample no :") or ll.startswith("sample:"):
            parts = line.split(":", 1)
            if len(parts) > 1:
                meta['sample_id'] = parts[1].strip()
        elif ll.startswith("name:"):
            meta['name'] = line.split(":", 1)[1].strip()
        elif ll.startswith("type:"):
            meta['type'] = line.split(":", 1)[1].strip()
        elif ll.startswith("class:"):
            meta['class'] = line.split(":", 1)[1].strip()
        elif ll.startswith("particle size"):
            # Only the header field (e.g. "Particle Size: Coarse"); description lines that merely
            # mention "particle size was 74-250 micrometers" must not overwrite it (v1.1.2 fix).
            meta['particle_size'] = line.split(":", 1)[1].strip()
        elif "y units" in ll or "unit" in ll:
            if "percent" in ll or "%" in ll:
                meta['units'] = 'percent'
            elif "0-1" in ll or "unity" in ll or "fraction" in ll:
                meta['units'] = 'fraction'
            if "transmittance" in ll:
                meta['measurement'] = 'transmission'
        elif "measurement" in ll:
            if "transmission" in ll or "transmittance" in ll:
                meta['measurement'] = 'transmission'
            elif "emissivity" in ll:
                meta['measurement'] = 'emissivity'
            elif "reflectance" in ll or "biconical" in ll or "directional" in ll:
                meta['measurement'] = 'reflectance'

        # Look for the start of the numeric data table (two floats on a line)
        parts = ll.split()
        if len(parts) >= 2:
            try:
                float(parts[0])
                float(parts[1])
                data_start = i
                break
            except ValueError:
                pass

    if data_start == -1:
        return None

    # GATE 2: Strict Modality Gate - Reject Transmission Pellets (KBr)
    if meta['measurement'] == 'transmission':
        return None

    # Parse numeric data
    wavs = []
    vals = []
    for line in lines[data_start:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                w = float(parts[0])
                v = float(parts[1])
                wavs.append(w)
                vals.append(v)
            except ValueError:
                continue

    # GATE 1: Minimum Channels and Spectral Coverage
    if len(wavs) < 10:
        return None

    wavs = np.array(wavs, dtype=np.float64)
    vals = np.array(vals, dtype=np.float64)

    # Validate that spectrum spans the critical TIR window (8.0 to 12.0 um)
    if wavs.min() > 8.5 or wavs.max() < 12.0:
        return None

    # Handle units: if percent (values > 1.5), divide by 100
    if meta['units'] == 'percent' or vals.max() > 1.5:
        vals = vals / 100.0

    # Default to reflectance if not explicitly set as emissivity
    if meta['measurement'] == 'unknown':
        meta['measurement'] = 'reflectance'

    # Handle reflectance vs emissivity: Kirchhoff's law for opaque solids (eps = 1 - rho)
    if meta['measurement'] == 'reflectance':
        emissivity = 1.0 - vals
    elif meta['measurement'] == 'emissivity':
        emissivity = vals
    else:
        # Any unsupported modality
        return None

    # GATE 3: Physical Bound Gate in Thermal Window (8.0 - 12.0 um)
    tir_mask = (wavs >= 8.0) & (wavs <= 12.0)
    if np.any(tir_mask):
        tir_emis = emissivity[tir_mask]
        # Terrestrial opaque solids: min emissivity >= 0.40 and max emissivity >= 0.80
        if np.min(tir_emis) < 0.40 or np.max(tir_emis) < 0.80:
            return None

    # Physical clamping to [0.001, 1.0]
    emissivity = np.clip(emissivity, 0.001, 1.0)

    # Fallback for sample_id from filename if metadata didn't provide one
    if meta['sample_id'] == 'unknown':
        fname_base = os.path.basename(filepath).lower().replace('.spectrum.txt', '').replace('.txt', '')
        tokens = fname_base.split('.')
        meta['sample_id'] = tokens[-2] if len(tokens) >= 2 else tokens[0]

    if meta['particle_size'] == 'unknown':
        fname_lower = os.path.basename(filepath).lower()
        if 'coarse' in fname_lower:
            meta['particle_size'] = 'coarse'
        elif 'fine' in fname_lower:
            meta['particle_size'] = 'fine'
        elif 'solid' in fname_lower:
            meta['particle_size'] = 'solid'
        else:
            meta['particle_size'] = 'natural'

    return wavs, emissivity, meta


def resample_to_grid(wavs: np.ndarray, emis: np.ndarray, grid_wavs: np.ndarray) -> np.ndarray:
    """
    Interpolates spectrum onto uniform high-resolution grid.
    Uses edge-padding for boundary channels outside measured range.
    """
    # Sort ascending
    sort_idx = np.argsort(wavs)
    wavs_s = wavs[sort_idx]
    emis_s = emis[sort_idx]

    # Remove potential duplicate wavelengths
    unique_wavs, unique_idx = np.unique(wavs_s, return_index=True)
    unique_emis = emis_s[unique_idx]

    interp_emis = np.interp(grid_wavs, unique_wavs, unique_emis, left=unique_emis[0], right=unique_emis[-1])
    return np.clip(interp_emis, 0.001, 1.0)


def curate_rebalanced_library(raw_dir: str, output_dir: str, seed: int = 42):
    """
    Curates the rebalanced library (~245 spectra) with complete metadata catalog and quarantine tags.
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("Scanning ECOSTRESS library for pristine spectra...")
    all_files = glob.glob(os.path.join(raw_dir, "**", "*.*"), recursive=True)
    spectrum_files = [f for f in all_files if f.lower().endswith(('.spectrum.txt', '.txt')) and not f.lower().endswith('.ancillary.txt')]
    print(f"Found {len(spectrum_files)} candidates in {raw_dir}")

    target_defs = TARGET_DEFS
    botanical_keywords = BOTANICAL_KEYWORDS

    geo_urban_records = []
    botanical_candidates = []

    for fpath in spectrum_files:
        fn_lower = os.path.basename(fpath).lower()

        # Check geological / urban
        matched_geo = False
        for tdef in target_defs:
            if any(kw in fn_lower for kw in tdef["keywords"]):
                parsed = parse_ecostress_spectrum(fpath)
                if parsed is not None:
                    wavs, emis, meta = parsed
                    grid_emis = resample_to_grid(wavs, emis, GRID_WAVELENGTHS)
                    geo_urban_records.append({
                        'fpath': fpath,
                        'class': tdef["class"],
                        'superclass': tdef["superclass"],
                        'meta': meta,
                        'emissivity': grid_emis
                    })
                matched_geo = True
                break

        if matched_geo:
            continue

        # Check botanical
        if any(kw in fn_lower for kw in botanical_keywords):
            parsed = parse_ecostress_spectrum(fpath)
            if parsed is not None:
                wavs, emis, meta = parsed
                grid_emis = resample_to_grid(wavs, emis, GRID_WAVELENGTHS)
                botanical_candidates.append({
                    'fpath': fpath,
                    'class': 'botanical',
                    'superclass': 'botanical',
                    'meta': meta,
                    'emissivity': grid_emis
                })

    print(f"\nDiscovered Valid Geological/Urban Spectra: {len(geo_urban_records)}")
    geo_counts = pd.Series([r['class'] for r in geo_urban_records]).value_counts()
    for c, cnt in geo_counts.items():
        print(f"  - {c:15s}: {cnt:3d} spectra")

    print(f"\nDiscovered Valid Botanical Spectra: {len(botanical_candidates)}")

    # Botanical Subsampling: Option A (Standard L2 Euclidean Farthest Point Sampling - FPS / Max-Min Diversity)
    # Group by unique physical sample_id to prevent intra-sample duplicates
    seen_sids = set()
    unique_botanical = []
    for r in botanical_candidates:
        sid = r['meta']['sample_id']
        if sid not in seen_sids:
            seen_sids.add(sid)
            unique_botanical.append(r)

    print(f"Unique Botanical Physical Samples/Species: {len(unique_botanical)}")

    # Target: select 100 maximally diverse botanical spectra via L2 Euclidean FPS
    n_botanical_target = min(100, len(unique_botanical))

    # Deterministic Anchor: White-Fir Bark Golden Target (vh311)
    vh311_idx = None
    for i, r in enumerate(unique_botanical):
        if 'vh311' in r['meta']['sample_id'].lower() or 'vh311' in r['fpath'].lower():
            vh311_idx = i
            break
    if vh311_idx is None:
        vh311_idx = 0

    E = np.array([r['emissivity'] for r in unique_botanical])  # shape (N_botanical, 1201)
    selected_indices = [vh311_idx]
    min_dists = np.linalg.norm(E - E[vh311_idx], axis=1)

    for _ in range(1, n_botanical_target):
        min_dists[selected_indices] = -1.0
        next_idx = int(np.argmax(min_dists))
        selected_indices.append(next_idx)
        new_dists = np.linalg.norm(E - E[next_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    selected_botanical_records = [unique_botanical[i] for i in selected_indices]
    print(f"Selected Subsampled Botanical Spectra via L2 FPS: {len(selected_botanical_records)} (anchor: {unique_botanical[vh311_idx]['meta']['sample_id']})")

    # Combine Geological/Urban + Botanical
    combined_records = geo_urban_records + selected_botanical_records
    print(f"\nTotal Rebalanced Library Candidates: {len(combined_records)} spectra")

    # Catalog & Quarantine Assignment
    catalog = []
    emissivity_matrix = []

    for i, rec in enumerate(combined_records):
        fname_lower = os.path.basename(rec['fpath']).lower()
        sid_lower = rec['meta']['sample_id'].lower()
        mat_class = rec['class']

        # Determine if this is one of the 7 Golden Targets
        is_golden_target = False
        golden_target_name = None

        for gt_name, target_file in GOLDEN_TARGET_PRIMARY_FILES.items():
            if fname_lower == target_file.lower():
                is_golden_target = True
                golden_target_name = gt_name
                break

        # Check for same-sample-ID sibling (and PI-approved explicit specimen-series siblings)
        is_sibling = False
        if not is_golden_target:
            if sid_lower in GOLDEN_SAMPLE_IDS or sid_lower in EXPLICIT_SIBLING_SAMPLE_IDS:
                is_sibling = True

        spectrum_id = f"SPEC_{i:04d}_{mat_class}"
        emis = rec['emissivity']
        emissivity_matrix.append(emis)

        # Hash of emissivity array for cryptographic audit
        emis_hash = hashlib.md5(emis.tobytes()).hexdigest().upper()

        catalog.append({
            'spectrum_id': spectrum_id,
            'sample_id': rec['meta']['sample_id'],
            'material_class': mat_class,
            'superclass': rec['superclass'],
            'particle_size': rec['meta'].get('particle_size', 'unknown'),
            'measurement_modality': rec['meta'].get('measurement', 'reflectance'),
            'name': rec['meta']['name'],
            'is_golden_target': is_golden_target,
            'golden_target_name': golden_target_name if is_golden_target else 'none',
            'is_sibling_quarantined': is_sibling,
            'min_emissivity': round(float(np.min(emis)), 4),
            'max_emissivity': round(float(np.max(emis)), 4),
            'mean_emissivity': round(float(np.mean(emis)), 4),
            'emissivity_hash': emis_hash,
            'source_filename': os.path.basename(rec['fpath']),
            'source_filepath': rec['fpath']
        })

    catalog_df = pd.DataFrame(catalog)
    emissivity_matrix = np.array(emissivity_matrix, dtype=np.float32)

    # v1.1.2 (a): drop exact emissivity duplicates published under different sample IDs
    # (keep the first occurrence; spectrum_ids are assigned before dropping, so IDs stay stable).
    dup_mask = catalog_df.duplicated(subset='emissivity_hash', keep='first').values
    for r in catalog_df[dup_mask].itertuples():
        keeper = catalog_df[(catalog_df['emissivity_hash'] == r.emissivity_hash) & ~dup_mask].iloc[0]
        print(f"  [dedup] dropping {r.spectrum_id} ({r.sample_id}): identical emissivity to "
              f"{keeper['spectrum_id']} ({keeper['sample_id']})")
    catalog_df = catalog_df[~dup_mask].reset_index(drop=True)
    emissivity_matrix = emissivity_matrix[~dup_mask]

    # v1.1.2 (b): spectral-sibling quarantine - any non-golden spectrum radiometrically
    # indistinguishable from a golden target (RMSE < SPECTRAL_TWIN_RMSE) is quarantined.
    gold_idx = np.where(catalog_df['is_golden_target'].values)[0]
    E64 = emissivity_matrix.astype(np.float64)
    for gi in gold_idx:
        rmse = np.sqrt(((E64 - E64[gi]) ** 2).mean(axis=1))
        for j in np.where(rmse < SPECTRAL_TWIN_RMSE)[0]:
            if j != gi and not catalog_df.at[j, 'is_golden_target'] and not catalog_df.at[j, 'is_sibling_quarantined']:
                catalog_df.at[j, 'is_sibling_quarantined'] = True
                print(f"  [spectral sibling] {catalog_df.at[j, 'spectrum_id']} RMSE={rmse[j]:.4f} to golden "
                      f"{catalog_df.at[gi, 'spectrum_id']}")

    # Save to disk
    csv_path = os.path.join(output_dir, "rebalanced_spectral_library.csv")
    npz_path = os.path.join(output_dir, "rebalanced_spectral_library.npz")
    catalog_path = os.path.join(output_dir, "library_metadata_catalog.csv")

    catalog_df.to_csv(catalog_path, index=False)
    
    # Save CSV format with wavelengths as columns
    cols = ['spectrum_id', 'sample_id', 'material_class', 'superclass', 'is_golden_target'] + [f"wav_{w:.3f}um" for w in GRID_WAVELENGTHS]
    df_full = catalog_df[['spectrum_id', 'sample_id', 'material_class', 'superclass', 'is_golden_target']].copy()
    for j, w in enumerate(GRID_WAVELENGTHS):
        df_full[f"wav_{w:.3f}um"] = emissivity_matrix[:, j]
    df_full.to_csv(csv_path, index=False)

    np.savez_compressed(
        npz_path,
        wavelengths_um=GRID_WAVELENGTHS,
        wavelengths=GRID_WAVELENGTHS,
        emissivity=emissivity_matrix,
        spectrum_ids=catalog_df['spectrum_id'].values,
        sample_ids=catalog_df['sample_id'].values,
        material_classes=catalog_df['material_class'].values,
        superclasses=catalog_df['superclass'].values,
        is_golden_target=catalog_df['is_golden_target'].values,
        is_sibling_quarantined=catalog_df['is_sibling_quarantined'].values
    )

    print(f"\n================================================================")
    print(f"Library Curation Successfully Completed!")
    print(f"Total Physical Spectra Curated:    {len(catalog_df)}")
    print(f"Golden Targets Quarantined:        {catalog_df['is_golden_target'].sum()}")
    print(f"Golden Siblings Excluded:          {catalog_df['is_sibling_quarantined'].sum()}")
    print(f"Development Set (Train/Val):       {(~catalog_df['is_golden_target'] & ~catalog_df['is_sibling_quarantined']).sum()}")
    print(f"Spectral Channels per Spectrum:    {len(GRID_WAVELENGTHS)} ({WAVELENGTH_MIN} to {WAVELENGTH_MAX} um)")
    print(f"Unique Emissivity Hashes:          {catalog_df['emissivity_hash'].nunique()} / {len(catalog_df)}")
    print(f"Saved artifacts:")
    print(f"  - {catalog_path}")
    print(f"  - {csv_path}")
    print(f"  - {npz_path}")
    print(f"================================================================")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    RAW_ECOSPECLIB = os.path.join(PROJECT_ROOT, "data", "raw", "ecospeclib", "ecospeclib-all")
    OUT_SPECTRAL = os.path.join(PROJECT_ROOT, "data", "processed", "spectral_libraries")
    curate_rebalanced_library(RAW_ECOSPECLIB, OUT_SPECTRAL)
