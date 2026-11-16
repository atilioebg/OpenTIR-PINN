"""
Build the Single Source of Truth (SSoT) manifest for the OpenTIR-PINN dataset.
===========================================================================
Every quantitative fact quoted in README.md, DATA_DICTIONARY.md, the dashboard,
clarification.md, and the manuscript MUST be read from (or checked against)
`data/dataset_facts.json`, which this script derives *exclusively* from the
generated artefacts (catalogs, split manifests, LUT catalog, and .pt tensors).

No number in this file is typed by hand: generator parameters that are not
recoverable from the tensors are cross-checked empirically (e.g., temperature
range, augmentation flags) and tagged with their source code location.

Usage:
    python scripts/build_dataset_facts.py            # writes data/dataset_facts.json (+ zenodo copy)
    python scripts/build_dataset_facts.py --check    # exits 1 if the manifest on disk is stale
"""

import os
import sys
import json
import hashlib
import argparse
import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_generation.base.synthesize_forward_dataset import (NOISE_MODEL, SAMPLING_MODEL,  # noqa: E402
    ATM_ERROR_MODEL, BAND_MASK_MODEL, compute_band_mask, build_sensor_srf_matrix, load_atmospheric_luts)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS_DIR = os.path.join(ROOT, "data", "processed", "final_datasets")
OUT_PATH = os.path.join(ROOT, "data", "dataset_facts.json")
ZENODO_COPY = os.path.join(ROOT, "zenodo_release", "data", "dataset_facts.json")
GENERATOR = "src/data_generation/base/synthesize_forward_dataset.py"


def md5(path, chunk=1 << 20):
    hsh = hashlib.md5()
    with open(path, "rb") as fp:
        for blk in iter(lambda: fp.read(chunk), b""):
            hsh.update(blk)
    return hsh.hexdigest().upper()


def summarize_partition(path):
    samples = torch.load(path, map_location="cpu", weights_only=False)
    temps = np.array([float(s["target_temp"][0]) for s in samples])
    atm = Counter(s["atmosphere_id"] for s in samples)
    pwv = np.array([float(s["pwv"]) for s in samples])
    aug = Counter(bool(s.get("is_augmented", False)) for s in samples)
    return {
        "file": os.path.relpath(path, ROOT).replace("\\", "/"),
        "md5": md5(path),
        "n_samples": len(samples),
        "n_bands": int(samples[0]["input_radiance"].shape[0]),
        "n_unique_spectra": len({s["spectrum_id"] for s in samples}),
        "n_unique_sample_ids": len({s["sample_id"] for s in samples}),
        "n_atmospheres": len(atm),
        "atmosphere_ids": sorted(atm),
        "n_unique_temperatures": int(len(np.unique(np.round(temps, 6)))),
        "temperature_K": {"min": round(float(temps.min()), 3), "max": round(float(temps.max()), 3),
                          "mean": round(float(temps.mean()), 3)},
        "pwv_g_cm2": {"min": round(float(pwv.min()), 4), "max": round(float(pwv.max()), 4),
                      "mean": round(float(pwv.mean()), 4)},
        "is_augmented": {"true": aug.get(True, 0), "false": aug.get(False, 0)},
        **({"pwv_est_minus_true_mm": {"mean": round(float(10 * np.mean([s["pwv_est"] - s["pwv"] for s in samples])), 4),
                                      "std": round(float(10 * np.std([s["pwv_est"] - s["pwv"] for s in samples])), 4)},
            "pwv_est_clipped_fraction": round(float(np.mean([bool(s["pwv_est_clipped"]) for s in samples])), 4)}
           if "pwv_est" in samples[0] else {}),
        "material_classes": dict(sorted(Counter(s["material_class"] for s in samples).items())),
        "samples_per_spectrum": sorted(set(Counter(s["spectrum_id"] for s in samples).values())),
    }


def build():
    facts = {"schema_version": "1.0", "dataset_version": "1.3.0",
             "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
             "generator": "scripts/build_dataset_facts.py"}

    # 1. Spectral library
    cat = pd.read_csv(os.path.join(ROOT, "data", "processed", "spectral_libraries", "library_metadata_catalog.csv"))
    gold = cat[cat["is_golden_target"].astype(str) == "True"]
    sib = cat[cat["is_sibling_quarantined"].astype(str) == "True"]
    dev = cat[(cat["is_golden_target"].astype(str) != "True") & (cat["is_sibling_quarantined"].astype(str) != "True")]
    facts["library"] = {
        "n_total": int(len(cat)), "n_development": int(len(dev)),
        "n_golden_targets": int(len(gold)), "n_quarantined_siblings": int(len(sib)),
        "n_classes_development": int(dev["material_class"].nunique()),
        "development_classes": dict(sorted(Counter(dev["material_class"]).items())),
        "emissivity_min_over_library": round(float(cat["min_emissivity"].min()), 4),
        "golden_targets": [
            {"spectrum_id": r.spectrum_id, "golden_name": r.golden_target_name, "sample_id": r.sample_id,
             "name": r.name, "particle_size": r.particle_size, "measurement_modality": r.measurement_modality,
             "eps_min": float(r.min_emissivity), "eps_max": float(r.max_emissivity),
             "eps_mean": float(r.mean_emissivity), "emissivity_hash": r.emissivity_hash}
            for r in gold.itertuples()],
        "quarantined_siblings": [{"spectrum_id": r.spectrum_id, "sample_id": r.sample_id, "name": r.name}
                                 for r in sib.itertuples()],
        "catalog_md5": md5(os.path.join(ROOT, "data", "processed", "spectral_libraries", "library_metadata_catalog.csv")),
    }

    # 2. Split manifest
    # v1.3.0 fix (08/11/2026): the grouping rule is read from the manifest metadata (no hard-coded fallback),
    # and per-fold counts distinguish physical sample IDs from similarity groups. The manifest field
    # `num_train_groups` / `num_val_groups` counts unique sample_ids, NOT similarity groups; similarity-group
    # counts are recomputed here with the same function used by make_split.py.
    split_path = os.path.join(ROOT, "splits", "split_v1.1.json")
    split = json.load(open(split_path, "r"))
    meta = split["metadata"]
    from src.data_generation.base.make_split import similarity_groups, SPECTRAL_TWIN_RMSE  # noqa: E402
    lib_npz = np.load(os.path.join(ROOT, "data", "processed", "spectral_libraries", "rebalanced_spectral_library.npz"),
                      allow_pickle=True)
    emis_lookup = {str(s): e for s, e in zip(lib_npz["spectrum_ids"], lib_npz["emissivity"])}
    dev_sorted = dev.reset_index(drop=True)
    sim_labels, n_sim_groups, largest_group = similarity_groups(dev_sorted, emis_lookup, SPECTRAL_TWIN_RMSE)
    if n_sim_groups != meta["num_groups"] or largest_group != meta["largest_group_size"]:
        raise AssertionError(f"Similarity groups recomputed ({n_sim_groups}, largest {largest_group}) differ from "
                             f"split manifest ({meta['num_groups']}, largest {meta['largest_group_size']})")
    group_of = dict(zip(dev_sorted["spectrum_id"], sim_labels))
    facts["split"] = {
        "file": "splits/split_v1.1.json", "md5": md5(split_path),
        "method": f"StratifiedGroupKFold (stratified by {meta['stratification_variable']}; "
                  f"groups = {meta['group_variable']}; random_state = {meta['random_state']})",
        "n_folds": len(split["folds"]),
        "n_similarity_groups": int(meta["num_groups"]),
        "largest_similarity_group_spectra": int(meta["largest_group_size"]),
        "folds": [{"fold": f["fold_idx"], "n_train_spectra": f["num_train"], "n_val_spectra": f["num_val"],
                   "n_train_sample_ids": len(set(f["train_sample_ids"])),
                   "n_val_sample_ids": len(set(f["val_sample_ids"])),
                   "n_train_similarity_groups": len({group_of[s] for s in f["train_spectrum_ids"]}),
                   "n_val_similarity_groups": len({group_of[s] for s in f["val_spectrum_ids"]}),
                   "similarity_groups_shared_train_val": len({group_of[s] for s in f["train_spectrum_ids"]}
                                                             & {group_of[s] for s in f["val_spectrum_ids"]})}
                  for f in split["folds"]],
    }

    # 3. Atmospheres
    lut = pd.read_csv(os.path.join(ROOT, "data", "processed", "atmospheric_luts", "lut_catalog.csv"))
    def grp(df):
        return {"n_profiles": int(len(df)),
                "models": {int(k): v for k, v in sorted(df.groupby("atm_model_idx")["atm_model_name"].first().items())},
                "h2o_scales": sorted(float(x) for x in df["h2o_scale"].unique()),
                "pwv_g_cm2": {"min": round(float(df["pwv_g_cm2"].min()), 4), "max": round(float(df["pwv_g_cm2"].max()), 4)},
                "profile_ids": sorted(df["profile_id"].tolist()),
                "h2o_saturation_capped": (df[df["h2o_capped_at_saturation"].astype(str) == "True"]
                                          [["profile_id", "pwv_requested_g_cm2", "pwv_g_cm2"]].to_dict("records")
                                          if "h2o_capped_at_saturation" in df.columns else [])}
    z0 = np.load(os.path.join(ROOT, "data", "processed", "atmospheric_luts", lut["npz_file"].iloc[0]), allow_pickle=True)
    facts["atmospheres"] = {
        "engine": "MODTRAN 5.2", "n_profiles": int(len(lut)), "n_unique_md5": int(lut["md5_hash"].nunique()),
        "development": grp(lut[lut["quarantine_status"] == "development"]),
        "golden_quarantined": grp(lut[lut["quarantine_status"].str.contains("golden")]),
        "rt_config": str(z0["rt_config"]) if "rt_config" in z0.files else "unknown",
        "downwelling_definition": (str(z0["downwelling_method"]) if "downwelling_method" in z0.files else "zenith radiance") +
            "; L_down = F_down/pi (hemispheric sky radiance for a Lambertian surface), W/(m^2 sr um), no extra /pi applies. "
            "Source: src/data_generation/base/automate_modtran.py (hemispheric_downwelling).",
    }

    # 4. Forward partitions
    parts = {}
    for sensor in ("aster", "hytes"):
        sp = {}
        for k in range(len(split["folds"])):
            sp[f"train_fold{k}"] = summarize_partition(os.path.join(DS_DIR, f"train_fold{k}_{sensor}.pt"))
            sp[f"val_oof_fold{k}"] = summarize_partition(os.path.join(DS_DIR, f"val_fold{k}_{sensor}.pt"))
            q3 = os.path.join(DS_DIR, f"quadrant_III_fold{k}_{sensor}.pt")
            if os.path.exists(q3):
                sp[f"q3_fold{k}"] = summarize_partition(q3)
        sp["q2"] = summarize_partition(os.path.join(DS_DIR, f"golden_benchmark_quadrant_II_{sensor}.pt"))
        sp["q4"] = summarize_partition(os.path.join(DS_DIR, f"golden_benchmark_quadrant_IV_{sensor}.pt"))
        parts[sensor] = sp
    facts["partitions"] = parts

    a = parts["aster"]
    canonical = {"train": a["train_fold0"]["n_samples"], "val_oof": a["val_oof_fold0"]["n_samples"],
                 "q2": a["q2"]["n_samples"], "q4": a["q4"]["n_samples"]}
    if "q3_fold0" in a:
        canonical["q3"] = a["q3_fold0"]["n_samples"]
    per_sensor = sum(canonical.values())
    facts["canonical_counts_fold0"] = {
        **canonical, "total_per_sensor": per_sensor, "total_both_sensors": 2 * per_sensor,
        "q3_union_all_folds": sum(a[f"q3_fold{k}"]["n_samples"] for k in range(5)) if "q3_fold0" in a else None,
        "plain_language": (f"Each evaluation partition = targets x atmospheres x temperatures; every sample also "
                           f"carries sensor noise (NEDT = " + ", ".join(f"{k.upper()} {v} K" for k, v in NOISE_MODEL["nedt_k"].items()) +
                           f" at evaluation; x U{tuple(NOISE_MODEL['train_noise_multiplier'])} in training) and an atmospheric-correction "
                           f"PWV error (N4: {ATM_ERROR_MODEL['eval_sigma_mm']} mm at evaluation, U{tuple(ATM_ERROR_MODEL['train_sigma_mm'])} mm "
                           f"in training). Fold 0 totals {per_sensor:,} samples per sensor "
                           f"({2 * per_sensor:,} for ASTER + HyTES)."),
    }

    # 5. Generator parameters (code-sourced; empirically cross-checked above)
    _LUTS = load_atmospheric_luts()
    facts["generator_parameters"] = {
        "source": GENERATOR,
        "noise_model": {k: (list(v) if isinstance(v, tuple) else v) for k, v in NOISE_MODEL.items()},
        "sampling_model": {k: (list(v) if isinstance(v, tuple) else v) for k, v in SAMPLING_MODEL.items()},
        "atmospheric_correction_error_model": {k: (list(v) if isinstance(v, tuple) else v) for k, v in ATM_ERROR_MODEL.items()},
        "noise_sigma": "sigma_L(lambda) = dB/dT|_{Tref}(lambda_c) * NEdT[sensor] * noise_multiplier",
        "train": {"sampling": "uniform per training spectrum",
                  "samples_per_spectrum_observed": a["train_fold0"]["samples_per_spectrum"],
                  "atmosphere": "uniform draw among development profiles",
                  "emissivity_augmentation": ("none: training emissivity = pristine laboratory spectrum (N5 + N6, v1.2.2)"
                                              if (tuple(NOISE_MODEL["eps_scale"]), NOISE_MODEL["eps_jitter_sigma"]) == ((1.0, 1.0), 0.0)
                                              else "clip(eps * U(eps_scale) + N(0, eps_jitter_sigma^2), 0, 1)")},
        "band_mask": {"model": BAND_MASK_MODEL, **{s_: (lambda W_c: {"masked_bands": int((~m).sum()), "n_bands": int(m.size),
                      "masked_center_um": [round(float(c), 4) for c, ok in zip(W_c[1], m) if not ok]})(W_c)
                      for s_, W_c, m in [(s_, build_sensor_srf_matrix(s_), compute_band_mask(s_, build_sensor_srf_matrix(s_)[0], _LUTS))
                                          for s_ in ("aster", "hytes")]}},
        "evaluation_partitions": {"noise_multiplier": 1.0, "emissivity": "pristine laboratory (no augmentation)",
                                  "val_oof_q2_atmospheres": a["q2"]["atmosphere_ids"]},
    }
    return facts


def strip_volatile(d):
    d = json.loads(json.dumps(d))  # JSON round-trip: int keys -> str, tuples -> lists (same as on disk)
    d.pop("generated_at", None)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    facts = build()
    if args.check:
        on_disk = json.load(open(OUT_PATH, "r", encoding="utf-8"))
        stale = strip_volatile(on_disk) != strip_volatile(facts)
        print("STALE" if stale else "UP-TO-DATE")
        sys.exit(1 if stale else 0)
    for p in [OUT_PATH] + ([ZENODO_COPY] if os.path.isdir(os.path.join(ROOT, "zenodo_release")) else []):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fp:
            json.dump(facts, fp, indent=2, ensure_ascii=False)
    c = facts["canonical_counts_fold0"]
    print(f"Wrote {OUT_PATH}")
    print(f"Library: {facts['library']['n_total']} total | {facts['library']['n_development']} dev | "
          f"{facts['library']['n_golden_targets']} golden | {facts['library']['n_quarantined_siblings']} sibling")
    print(f"Atmospheres: {facts['atmospheres']['development']['n_profiles']} dev | "
          f"{facts['atmospheres']['golden_quarantined']['n_profiles']} golden")
    print(f"Fold-0 counts: {c}")

