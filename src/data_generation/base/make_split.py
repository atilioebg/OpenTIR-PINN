"""
Grouped & Stratified 5-Fold Partition Engine for OpenTIR-PINN v1.1.0
===================================================================

Generates a leak-free 5-fold cross-validation partition on the rebalanced
spectral library:
1. Strict Clean-Room Quarantine:
   - Isolates the 7 Primary Golden Targets and 6 Physical Sibling Spectra
   - Permanently locks them in a clean-room test vault (never seen in training or validation)
2. Group-Aware Stratified Partition:
   - Evaluates the remaining 237 physical development spectra
   - Uses StratifiedGroupKFold (K=5, random_state=42)
   - Groups by physical `sample_id` (guaranteeing physical samples never cross folds)
   - Stratifies by `material_class` (ensuring balanced class distributions across all 5 folds)
3. Cryptographic Assertions:
   - Proves zero overlap between Train, Validation, and Golden Quarantined sets
   - Confirms that every development spectrum is validated exactly once (out-of-fold coverage)
   - Saves reproducible manifest to `splits/split_v1.1.json` and `splits/golden_targets.json`
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedGroupKFold
from typing import Dict, List, Any


SPECTRAL_TWIN_RMSE = 0.005  # must match curate_spectral_library.SPECTRAL_TWIN_RMSE


def similarity_groups(dev_df: pd.DataFrame, emissivity_lookup: Dict[str, np.ndarray], tau: float):
    """
    Group labels = connected components of the graph whose edges join spectra that share a physical
    sample_id OR whose emissivity RMSE is below tau (radiometric indistinguishability, ~3x NEdT).
    Grouping by these components guarantees that no validation spectrum has a training twin.
    """
    ids = dev_df['spectrum_id'].tolist()
    sids = dev_df['sample_id'].tolist()
    E = np.stack([emissivity_lookup[s] for s in ids]).astype(np.float64)
    parent = list(range(len(ids)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(ids)):
        d = np.sqrt(((E[i + 1:] - E[i]) ** 2).mean(axis=1))
        for j in np.where(d < tau)[0] + i + 1:
            parent[find(i)] = find(int(j))
        for j in range(i + 1, len(ids)):
            if sids[j] == sids[i]:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(len(ids))]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    labels = np.array([remap[r] for r in roots])
    sizes = np.bincount(labels)
    return labels, len(sizes), int(sizes.max())


def generate_grouped_split(
    catalog_path: str,
    output_dir: str,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes the clean-room quarantine and generates the 5-fold StratifiedGroupKFold split.
    """
    os.makedirs(output_dir, exist_ok=True)
    lib_npz = np.load(os.path.join(os.path.dirname(catalog_path), "rebalanced_spectral_library.npz"), allow_pickle=True)
    emissivity_lookup = {str(s): e for s, e in zip(lib_npz['spectrum_ids'], lib_npz['emissivity'])}

    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Library catalog not found at: {catalog_path}")

    df = pd.read_csv(catalog_path)
    total_spectra = len(df)

    print(f"================================================================")
    print(f"Generating 5-Fold Grouped Partition for OpenTIR-PINN v1.1.0")
    print(f"Input Catalog: {catalog_path} ({total_spectra} total spectra)")
    print(f"================================================================")

    # 1. Clean-Room Quarantine Isolation
    golden_df = df[df['is_golden_target']].copy().reset_index(drop=True)
    sibling_df = df[df['is_sibling_quarantined']].copy().reset_index(drop=True)
    dev_df = df[~df['is_golden_target'] & ~df['is_sibling_quarantined']].copy().reset_index(drop=True)

    print(f"Quarantined Golden Targets:   {len(golden_df)} spectra")
    print(f"Quarantined Physical Siblings: {len(sibling_df)} spectra")
    print(f"Development Set (Train/Val):   {len(dev_df)} spectra")

    if len(golden_df) != 7:
        raise ValueError(f"Expected exactly 7 Golden Targets, found {len(golden_df)}!")
    # v1.1.2: sibling and development counts are data-driven (spectral-sibling quarantine and
    # exact-duplicate removal happen in curate_spectral_library.py); they are reported, not hardcoded.
    if len(sibling_df) < 1:
        raise ValueError("Expected at least the calcite_1 same-sample sibling to be quarantined")

    # Save Golden Targets and Siblings Vault
    golden_targets_record = {
        "metadata": {
            "version": "1.1.2",
            "date": "deterministic build (no timestamp; provenance in data/dataset_facts.json)",
            "purpose": "Immutable benchmark test suite - evaluated strictly once",
            "num_golden_targets": len(golden_df),
            "num_quarantined_siblings": len(sibling_df)
        },
        "golden_targets": golden_df[[
            'spectrum_id', 'sample_id', 'material_class', 'superclass',
            'golden_target_name', 'name', 'source_filename', 'emissivity_hash'
        ]].to_dict(orient="records"),
        "quarantined_siblings": sibling_df[[
            'spectrum_id', 'sample_id', 'material_class', 'superclass',
            'name', 'source_filename', 'emissivity_hash'
        ]].to_dict(orient="records")
    }

    golden_json_path = os.path.join(output_dir, "golden_targets.json")
    with open(golden_json_path, "w", encoding="utf-8") as f:
        json.dump(golden_targets_record, f, indent=2)
    print(f"Saved Golden Vault to: {golden_json_path}")

    # 2. StratifiedGroupKFold on Development Corpus
    X = dev_df['spectrum_id'].values
    y = dev_df['material_class'].values
    groups, n_groups, largest = similarity_groups(dev_df, emissivity_lookup, SPECTRAL_TWIN_RMSE)
    print(f"Similarity-aware groups (sample_id OR emissivity RMSE < {SPECTRAL_TWIN_RMSE}): "
          f"{n_groups} groups, largest = {largest} spectra")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds_data = []
    validated_spectrum_ids = set()

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        train_sub = dev_df.iloc[train_idx]
        val_sub = dev_df.iloc[val_idx]

        train_spec_ids = train_sub['spectrum_id'].tolist()
        val_spec_ids = val_sub['spectrum_id'].tolist()

        train_sample_ids = set(train_sub['sample_id'].unique())
        val_sample_ids = set(val_sub['sample_id'].unique())

        # Leakage Assertion 0 (v1.1.2): no validation spectrum radiometrically indistinguishable from training
        tr_E = np.stack([emissivity_lookup[s] for s in train_sub['spectrum_id']])
        for vs in val_sub['spectrum_id']:
            d = np.sqrt(((tr_E - emissivity_lookup[vs]) ** 2).mean(axis=1)).min()
            if d < SPECTRAL_TWIN_RMSE:
                raise AssertionError(f"CRITICAL LEAK: {vs} has a training twin (RMSE {d:.4f}) in fold {fold_idx}")

        # Leakage Assertion 1: Zero overlap in spectrum IDs
        spec_overlap = set(train_spec_ids).intersection(set(val_spec_ids))
        if spec_overlap:
            raise AssertionError(f"CRITICAL LEAK: Fold {fold_idx} has overlapping spectrum IDs: {spec_overlap}")

        # Leakage Assertion 2: Zero overlap in physical sample IDs (Group Leakage)
        sample_overlap = train_sample_ids.intersection(val_sample_ids)
        if sample_overlap:
            raise AssertionError(f"CRITICAL LEAK: Fold {fold_idx} has overlapping physical sample IDs: {sample_overlap}")

        # Leakage Assertion 3: Zero overlap with Golden Targets or Siblings
        all_quarantine_ids = set(golden_df['spectrum_id']).union(set(sibling_df['spectrum_id']))
        leak_with_golden = (set(train_spec_ids) | set(val_spec_ids)).intersection(all_quarantine_ids)
        if leak_with_golden:
            raise AssertionError(f"CRITICAL LEAK: Fold {fold_idx} contains quarantined targets: {leak_with_golden}")

        validated_spectrum_ids.update(val_spec_ids)

        # Class distributions
        train_dist = train_sub['material_class'].value_counts().to_dict()
        val_dist = val_sub['material_class'].value_counts().to_dict()

        folds_data.append({
            "fold_idx": fold_idx,
            "num_train": len(train_sub),
            "num_val": len(val_sub),
            "num_train_groups": len(train_sample_ids),
            "num_val_groups": len(val_sample_ids),
            "train_spectrum_ids": train_spec_ids,
            "val_spectrum_ids": val_spec_ids,
            "train_sample_ids": sorted(train_sample_ids),
            "val_sample_ids": sorted(val_sample_ids),
            "class_distribution_train": train_dist,
            "class_distribution_val": val_dist
        })

        print(f"Fold {fold_idx}: Train={len(train_sub):3d} ({len(train_sample_ids):3d} groups) | "
              f"Val={len(val_sub):2d} ({len(val_sample_ids):2d} groups) | "
              f"Overlap=0 | Classes in Val={len(val_dist)}")

    # Assertion 4: Complete out-of-fold validation coverage
    if validated_spectrum_ids != set(dev_df['spectrum_id']):
        missing = set(dev_df['spectrum_id']) - validated_spectrum_ids
        raise AssertionError(f"Validation coverage incomplete! Missing spectra: {missing}")

    print(f"\nVerification Check: 100% of Development Spectra validated exactly once (Out-of-Fold).")

    # 3. Compile Master Manifest
    manifest = {
        "metadata": {
            "version": "1.1.2",
            "date": "deterministic build (no timestamp; provenance in data/dataset_facts.json)",
            "random_state": random_state,
            "n_splits": n_splits,
            "group_variable": f"connected components of (same sample_id) OR (emissivity RMSE < {SPECTRAL_TWIN_RMSE})",
            "num_groups": int(n_groups),
            "largest_group_size": int(largest),
            "stratification_variable": "material_class",
            "total_curated_spectra": total_spectra,
            "num_golden_targets": len(golden_df),
            "num_quarantined_siblings": len(sibling_df),
            "num_development_spectra": len(dev_df),
            "golden_targets_vault_file": "splits/golden_targets.json"
        },
        "material_classes": sorted(dev_df['material_class'].unique().tolist()),
        "superclasses": sorted(dev_df['superclass'].unique().tolist()),
        "development_spectrum_ids": dev_df['spectrum_id'].tolist(),
        "folds": folds_data
    }

    split_json_path = os.path.join(output_dir, "split_v1.1.json")
    with open(split_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Compute manifest hash
    with open(split_json_path, "rb") as f:
        manifest_hash = hashlib.md5(f.read()).hexdigest().upper()

    print(f"Saved Split Manifest to: {split_json_path}")
    print(f"Manifest MD5 Hash:       {manifest_hash}")
    print(f"================================================================")
    print(f"5-Fold Grouped Split Generation Complete & 100% Leak-Free!")
    print(f"================================================================")

    return manifest


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    CATALOG_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "spectral_libraries", "library_metadata_catalog.csv")
    SPLITS_DIR = os.path.join(PROJECT_ROOT, "splits")
    generate_grouped_split(CATALOG_FILE, SPLITS_DIR, n_splits=5, random_state=42)
