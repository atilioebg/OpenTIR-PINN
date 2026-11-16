"""
Deep Integrity, Leakage & Statistical Validation of the OpenTIR-PINN Forward Dataset (v1.1.2)
=============================================================================================
Every check is declared once in TEST_CATALOG (purpose, method, pass rule, rationale) and every
result is stored with machine-readable metrics {name, value, op, limit, level}, so the report,
the protocol document and the dashboard all show the obtained value next to its limit.

Sections:
  A. Partition integrity & leakage   B. Physical & numerical integrity
  C. Statistical design & bias       D. Cross-validation structure

Outputs:
  data/validation/dataset_validation_report.json   (catalog + results + metrics)
  data/validation/dataset_validation_report.md     (results: obtained vs limit)
  docs/dataset_validation_protocol.md              (all tests, methods, thresholds, rationale)

Usage:  python scripts/validate_dataset_integrity.py [--sensors aster hytes] [--n_null 30]
"""

import os
import sys
import glob
import json
import argparse
import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from src.data_generation.base.synthesize_forward_dataset import (  # noqa: E402
    WAVELENGTH_GRID, WAV_M, C1, C2, build_sensor_srf_matrix, load_atmospheric_luts,
    load_rebalanced_library, planck_temp_derivative, NOISE_MODEL, BAND_MASK_MODEL, compute_band_mask,
    ATM_ERROR_MODEL, estimated_atmosphere, planck_blackbody_grid)
from src.data_generation.base.make_split import similarity_groups, SPECTRAL_TWIN_RMSE  # noqa: E402

DS = os.path.join(ROOT, "data", "processed", "final_datasets")
OUT_DIR = os.path.join(ROOT, "data", "validation")
PROTOCOL_MD = os.path.join(ROOT, "docs", "dataset_validation_protocol.md")
NEDT_BY_SENSOR = NOISE_MODEL["nedt_k"]
NEDT = max(NEDT_BY_SENSOR.values())  # reported constant
TWIN = SPECTRAL_TWIN_RMSE
_MA, _MB = NOISE_MODEL["train_noise_multiplier"]
TRAIN_NOISE_STD = float(np.sqrt((_MB ** 3 - _MA ** 3) / (3.0 * (_MB - _MA))))  # sqrt(E[m^2]), m ~ U(a, b)
_SA, _SB = NOISE_MODEL["eps_scale"]
_JIT = NOISE_MODEL["eps_jitter_sigma"]
_LDF = NOISE_MODEL["ldown_noise_factor"]
_SIG_A, _SIG_B = ATM_ERROR_MODEL["train_sigma_mm"]
_SIG_EVAL = ATM_ERROR_MODEL["eval_sigma_mm"]
B8_MAX_RATIO = 0.30
N4_CHECK_DIR = os.path.join(ROOT, "data", "validation", "n4_interp_check")

SECTIONS = {"A": "Partition integrity & leakage", "B": "Physical & numerical integrity",
            "C": "Statistical design & bias", "D": "Cross-validation structure"}

# ----------------------------------------------------------------------------------------------
# Test catalog (single declaration of every test)
# ----------------------------------------------------------------------------------------------
TEST_CATALOG = {
    "A0": dict(section="A", name="Library rows aligned with metadata catalog",
               purpose="Emissivity arrays and labels must refer to the same spectrum.",
               method="Assert npz `spectrum_ids` order equals catalog order (load_rebalanced_library).",
               rule="aligned == 1", rationale="A silent row shift would attach wrong materials to every sample."),
    "A1": dict(section="A", name="Train/val disjoint (spectrum & physical sample)",
               purpose="No spectrum or physical specimen is shared between training and validation.",
               method="Set intersection of spectrum_id and sample_id read from the tensors of each fold.",
               rule="overlap_spectra == 0 and overlap_samples == 0", rationale="Classical identity leakage."),
    "A1b": dict(section="A", name="Tensors consistent with split manifest",
                purpose="Generated partitions use exactly the spectra declared in splits/split_v1.1.json.",
                method="Compare spectrum sets of train/val tensors with the manifest.",
                rule="missing_train == 0 and val_mismatch == 0", rationale="Reproducibility of the split."),
    "A2": dict(section="A", name="Golden-target & sibling quarantine",
               purpose="No golden target or quarantined sibling enters any development partition.",
               method="Check spectrum_id, sample_id and emissivity MD5 of train/val/Q-III against the golden vault.",
               rule="leaking_partitions == 0", rationale="Zero-shot claims on Q-II/Q-IV require strict quarantine."),
    "A3a": dict(section="A", name="Library near-duplicates across specimens (descriptive)",
                purpose="Quantify spectral autocorrelation in the library.",
                method=f"Count spectrum pairs with emissivity RMSE < {TWIN} and different sample_id.",
                rule="informative (handled by similarity-aware grouping, see A3c)",
                rationale="Motivates grouping by spectral similarity, not only by specimen ID."),
    "A3b": dict(section="A", name="Golden targets have no development twin",
                purpose="Golden targets must not be radiometrically indistinguishable from a training spectrum.",
                method="Minimum emissivity RMSE from each golden target to every development spectrum.",
                rule=f"min_rmse >= {TWIN}",
                rationale=f"RMSE {TWIN} in emissivity ~ 0.3 K brightness temperature (1-1.5x the sensor NEdT)."),
    "A3c": dict(section="A", name="Validation spectra have no training twin",
                purpose="Out-of-fold validation must not contain near-copies of training spectra.",
                method=f"For each fold, count val spectra whose nearest training spectrum has RMSE < {TWIN}.",
                rule="val_spectra_with_twin == 0", rationale="Spectral-autocorrelation leakage (near-clones)."),
    "A4": dict(section="A", name="Atmosphere isolation",
               purpose="Golden (quarantined) atmospheres never appear in train/val; Q-III uses all of them.",
               method="Atmosphere-ID sets per partition vs lut_catalog quarantine status.",
               rule="golden_atm_in_train_val == 0 and q3_atmospheres == 10",
               rationale="Atmospheric OOD claims (Q-III/Q-IV) require isolation."),
    "A5": dict(section="A", name="Q-III spectra equal fold validation spectra",
               purpose="Q-III(fold k) must test only materials unseen by the fold-k model.",
               method="Set difference between Q-III and val spectrum sets.", rule="mismatch == 0",
               rationale="Keeps Q-III a pure atmospheric-OOD test on unseen specimens."),
    "A5b": dict(section="A", name="Q-III union covers every development spectrum once",
                purpose="Across folds, Q-III evaluates the whole development pool exactly once.",
                method="Multiplicity of each development spectrum over the 5 Q-III partitions.",
                rule="uncovered == 0 and max_multiplicity == 1", rationale="Complete, non-redundant coverage."),
    "A6": dict(section="A", name="Golden vault composition (complete factorial grid)",
               purpose="Q-II/Q-IV contain exactly 7 targets x N atmospheres x 11 temperatures, no duplicates.",
               method="Count unique (target, atmosphere, temperature) cells.",
               rule="targets == 7, atmospheres == N, duplicate_cells == 0", rationale="Balanced benchmark design."),
    "B1": dict(section="B", name="Finite values & physical bounds",
               purpose="No NaN/Inf; emissivity in [0, 1]; no radiance clipping artefacts.",
               method="Element-wise checks on all tensors.",
               rule="non_finite == 0, eps_out_of_bounds == 0, clip_events == 0",
               rationale="Numerical integrity of training targets and inputs."),
    "B2": dict(section="B", name="Forward-model reconstruction & NEdT noise law (at-sensor radiance)",
               purpose="The stored observation equals the noise-free RTE at the TRUE atmosphere plus the declared sensor noise.",
               method="Recompute W[tau (eps B + (1-eps) L_down) + L_up] with the true LUT; z = (L_sensor - clean)/sigma_L.",
               rule="|mean z| <= 0.05 and |std z - expected| <= 0.05 (expected 1.000 eval, 1.041 train)",
               rationale="Proves the physics and the instrument-noise model are exactly what the paper states."),
    "B2b": dict(section="B", name="Normality of injected noise (evaluation partitions)",
                purpose="Noise residuals follow N(0, 1).", method="Kolmogorov-Smirnov test on 20,000 at-sensor z values.",
                rule="KS p-value >= 0.01", rationale="Gaussian NEdT model."),
    "B2c": dict(section="B", name="Per-sample noise multiplier law (HyTES training)",
                purpose="Training noise multiplier m ~ U(0.5, 1.5).",
                method="Per-sample std of z over 256 bands, KS against U(0.5, 1.5).",
                rule="KS D <= 0.05", rationale="Verifies the robustness-to-unknown-NEdT design."),
    "B3": dict(section="B", name="Label consistency (band emissivity = W eps_highres)",
               purpose="Band-level targets are the SRF convolution of the high-resolution targets.",
               method="max |W eps_hr - eps_band|.", rule="max_abs_diff <= 1e-5",
               rationale="Consistent supervision at both resolutions."),
    "B4": dict(section="B", name="Augmentation confinement & law",
               purpose="Training emissivity follows the NOISE_MODEL law (v1.2.2: no perturbation, N5 + N6); evaluation emissivity is pristine.",
               method="Least-squares scale and residual std vs library spectrum (train); max |eps - library| (train when "
                      "the law is the identity, and eval).",
               rule="train: scale within eps_scale +/- 0.0005, |resid_std - eps_jitter_sigma| <= max(0.15 sigma, 2e-4), and "
                    "max_dev <= 1e-6 if no perturbation is configured; eval: max_dev <= 1e-6",
               rationale="Benchmarks evaluate laboratory spectra, not synthetic perturbations."),
    "B6": dict(section="B", name="Opaque-band mask present and correct",
               purpose="Bands opaque in some development atmosphere are flagged so they cannot be used as inputs.",
               method="Recompute the mask (SRF-convolved tau < threshold in any development atmosphere) and compare with the band_mask stored in every sample.",
               rule="samples_without_mask == 0 and samples_with_wrong_mask == 0",
               rationale="Surface radiance cannot be observed through an opaque atmosphere (oracle-input guard, N3)."),
    "B2d": dict(section="B", name="Network inputs = operational correction with the estimated atmosphere (N4)",
                purpose="tau_input, L_up_input, L_down and L_surf are exactly those of a correction made at the recorded PWV_est.",
                method="Recompute the estimated atmosphere at PWV_est (estimated_atmosphere) and "
                       "L_surf = W L_surf_true + (L_obs - L_up_in)/tau_in - (W L_sensor_true - W L_up)/(W tau) on usable bands.",
                rule="max relative difference <= 1e-4 for tau_input, L_up_input, L_down and L_surf",
                rationale="Proves that the only atmospheric information given to the network is the perturbed one."),
    "B7": dict(section="B", name="Atmospheric-correction (PWV) error law (N4)",
               purpose=f"PWV_est - PWV ~ N(0, s^2), s = {_SIG_EVAL} mm (evaluation) or s ~ U({_SIG_A}, {_SIG_B}) mm (training).",
               method="KS test of z = drawn error / s against N(0, 1); KS of s against U(a, b) (training); PWV_est = "
                      "clip(PWV + drawn error, 0, saturation cap); fraction clipped reported.",
               rule="KS p >= 0.01 (WARN); sigma consistency and PWV_est reconstruction exact (FAIL)",
               rationale="Wang et al. 2020 (reanalysis vs GNSS, 1.84-3.51 mm); Seemann et al. 2003 (MODIS, 4.1 mm)."),
    "B8": dict(section="B", name="PWV-interpolation fidelity against MODTRAN (N4)",
               purpose="The estimated atmosphere obtained by interpolation in PWV reproduces MODTRAN at intermediate columns.",
               method="24 MODTRAN check profiles at H2O scales 0.625/0.875/1.25/1.75 (data/validation/n4_interp_check, "
                      "saturation-capped ones excluded); LST error of a correction made with the interpolated instead of the "
                      "exact atmosphere (eps 0.95, 300 K, usable bands) vs the LST effect of an eval-sigma PWV error.",
               rule=f"max RMS ratio <= {B8_MAX_RATIO} (WARN)",
               rationale="Worst-case (mid-interval) interpolation error must stay a small fraction of the modelled error."),
    "B5": dict(section="B", name="Atmospheric LUT radiative definitions",
               purpose="L_down is the hemispheric sky radiance F_down/pi from a single, documented method.",
               method="Ratio hemispheric/zenith (8-12 um) for the 30 LUTs; method and RT config fields.",
               rule="1.0 < ratio_min and ratio_max < 2.0 and n_methods == 1",
               rationale="Lambertian reflection requires F_down/pi; zenith radiance underestimates it 17-41 %."),
    "C1": dict(section="C", name="Uniform per-spectrum training sampling",
               purpose="Every training spectrum contributes the same number of samples.",
               method="Distinct values of samples-per-spectrum in each training fold.", rule="distinct_counts == 1",
               rationale="Avoids repeating singleton-class spectra (memorization) and class-prior shift."),
    "C2": dict(section="C", name="Nuisance factors independent of material class",
               purpose="No shortcut correlation between class and atmosphere/temperature.",
               method="Chi-square (class x atmosphere), Kruskal-Wallis (T across classes), Spearman(T, PWV).",
               rule="chi2 p >= 0.01, Kruskal p >= 0.01, |Spearman| <= 0.05",
               rationale="Shortcut-learning guard."),
    "C3": dict(section="C", name="Training temperature law",
               purpose="T_s ~ U(275, 335) K.", method="KS test against U(275, 335).", rule="KS p >= 0.01",
               rationale="Prior support wider than the 280-330 K test grid avoids boundary shrinkage."),
    "C4": dict(section="C", name="Uniform use of development atmospheres",
               purpose="The 20 development atmospheres are sampled uniformly.",
               method="Chi-square goodness of fit on atmosphere counts.", rule="chi2 p >= 0.01 and min_count > 0",
               rationale="No atmosphere dominates training."),
    "C5": dict(section="C", name="Spectrum repetition (memorization risk)",
               purpose="No single spectrum carries a large share of the training set.",
               method="Max share of samples belonging to one spectrum.", rule="max_share <= 1 %",
               rationale="v1.1.1 had 9 % of training mass on one spectrum."),
    "C7": dict(section="C", name="Adversarial AUC train vs val (descriptive)",
               purpose="Describe input-distribution shift between training and validation.",
               method="Logistic regression (PCA-20 for HyTES) on log L_surf, 5-fold ROC-AUC.",
               rule="informative (group-disjoint CV implies AUC > 0.5 by construction)",
               rationale="Context for C7b."),
    "C7b": dict(section="C", name="Composition shift vs null of group-disjoint splits",
                purpose="Is a fold's train/val composition shift larger than any random group-stratified split?",
                method="Adversarial AUC on the val_oof union (identical generation process) for the actual fold vs "
                       "a null distribution from random StratifiedGroupKFold seeds with the same groups.",
                rule="observed AUC <= 95th percentile of null",
                rationale="Separates unavoidable group-CV shift from an unlucky/biased fold."),
    "C8": dict(section="C", name="Golden-atmosphere PWV vs development envelope (descriptive)",
               purpose="Characterize the nature of the atmospheric OOD shift.", method="PWV ranges.",
               rule="informative", rationale="Manuscript must describe OOD as thermal-profile shift where applicable."),
    "C9": dict(section="C", name="Golden spectral contrast vs development pool (descriptive)",
               purpose="Position golden targets within the development MMD distribution.",
               method="MMD (max-min emissivity, 8-12 um) percentiles.", rule="informative",
               rationale="Context for per-target benchmark interpretation."),
    "D1": dict(section="D", name="Training classes present in validation",
               purpose="Stratification places every class with enough independent groups in each validation fold.",
               method="Classes absent from val; flagged only if the class has >= 5 independent groups.",
               rule="flagged_classes == 0",
               rationale="Classes with < 5 independent groups (e.g. singletons) cannot appear in all folds."),
}

RESULTS = []
SEL = {"folds": None, "partitions": {"train", "val", "q2", "q3", "q4"}}  # smoke-test selection


def metric(name, value, op, limit=None, level="FAIL", unit=""):
    return {"name": name, "value": None if value is None else float(value), "op": op,
            "limit": None if limit is None else float(limit), "level": level, "unit": unit}


def _ok(m):
    v, l, op = m["value"], m["limit"], m["op"]
    if op == "info" or l is None:
        return True
    return {"<=": v <= l, "<": v < l, ">=": v >= l, ">": v > l, "==": abs(v - l) <= 1e-12}[op]


def record(cid, sensor, scope, detail, metrics):
    status = "INFO" if all(m["op"] == "info" for m in metrics) else "PASS"
    for m in metrics:
        if not _ok(m):
            status = "FAIL" if m["level"] == "FAIL" or status == "FAIL" else "WARN"
    cat = TEST_CATALOG[cid]
    RESULTS.append({"id": cid, "section": cat["section"], "sensor": sensor, "scope": scope, "status": status,
                    "title": cat["name"], "detail": detail, "metrics": metrics})
    print(f"[{status:4s}] {cid:4s} {sensor:5s} {scope:14s} {cat['name']}")


def planck_matrix(temps):
    t = np.asarray(temps, dtype=np.float64)[:, None]
    return (C1 / (WAV_M[None, :] ** 5)) / (np.exp(C2 / (WAV_M[None, :] * t)) - 1.0) * 1e-6


def load(name):
    return torch.load(os.path.join(DS, name), map_location="cpu", weights_only=False)


def stack(samples, key):
    return torch.stack([s[key] for s in samples]).double().numpy()


# ----------------------------------------------------------------------------------------------
# B: per-partition physics
# ----------------------------------------------------------------------------------------------
def physics_checks(sensor, pname, samples, W, centers, luts, library, is_train):
    eps_hr = stack(samples, "target_emissivity_highres")
    eps_b = stack(samples, "target_emissivity")
    l_surf = stack(samples, "input_radiance")
    l_sens = stack(samples, "input_radiance_sensor")
    l_down = stack(samples, "downwelling_radiance")
    temps = np.array([float(s["target_temp"][0]) for s in samples])
    atm = [s["atmosphere_id"] for s in samples]

    arrays = {"eps_hr": eps_hr, "eps_band": eps_b, "L_surf": l_surf, "L_sensor": l_sens, "L_down": l_down}
    nonfinite = int(sum((~np.isfinite(v)).sum() for v in arrays.values()))
    eps_oob = int(((eps_hr < 0) | (eps_hr > 1)).sum() + ((eps_b < 0) | (eps_b > 1)).sum())
    usable = compute_band_mask(sensor, W, luts)
    # L_surf in opaque (masked) bands is a corrected radiance through tau < 0.1 and may be non-physical by design
    clipped = int((l_sens <= 1.0001e-6).sum() + (l_down <= 1.0001e-6).sum() + (l_surf[:, usable] <= 1.0001e-6).sum())
    record("B1", sensor, pname, f"{len(samples)} samples",
           [metric("non_finite", nonfinite, "==", 0), metric("eps_out_of_bounds", eps_oob, "==", 0),
            metric("clip_events", clipped, "==", 0, level="WARN")])

    lab_err = float(np.abs(eps_hr @ W.T - eps_b).max())
    record("B3", sensor, pname, "", [metric("max_abs_diff", lab_err, "<=", 1e-5)])

    B = planck_matrix(temps)
    ld_hr = np.stack([luts[a]["L_down_surface"] for a in atm])
    tau_hr = np.stack([luts[a]["tau_aster" if sensor == "aster" else "tau_hytes"] for a in atm])
    lup_hr = np.stack([luts[a]["L_up_aster" if sensor == "aster" else "L_up_hytes"] for a in atm])
    lsurf_hr = eps_hr * B + (1.0 - eps_hr) * ld_hr
    sigma = planck_temp_derivative(centers, NOISE_MODEL["nedt_reference_temperature_k"]) * NEDT_BY_SENSOR[sensor]
    lsens_clean = (tau_hr * lsurf_hr + lup_hr) @ W.T
    zs = {"L_sensor": (l_sens - lsens_clean) / sigma}
    exp_std = TRAIN_NOISE_STD if is_train else 1.0
    max_mean = max(abs(float(v.mean())) for v in zs.values())
    max_std_dev = max(abs(float(v.std()) - exp_std) for v in zs.values())
    record("B2", sensor, pname,
           "; ".join(f"{k}: mean {v.mean():+.4f}, std {v.std():.4f}" for k, v in zs.items()) + f" (expected std {exp_std:.3f})",
           [metric("max_abs_mean_z", max_mean, "<=", 0.05), metric("max_abs_std_dev", max_std_dev, "<=", 0.05)])

    if not is_train:
        sub = np.random.default_rng(0).choice(zs["L_sensor"].ravel(), size=min(20000, zs["L_sensor"].size), replace=False)
        ks = stats.kstest(sub, "norm")
        record("B2b", sensor, pname, f"KS D={ks.statistic:.4f}",
               [metric("ks_pvalue", ks.pvalue, ">=", 0.01, level="WARN")])
    elif sensor == "hytes":
        m_hat = zs["L_sensor"].std(axis=1)
        ks = stats.kstest(m_hat, "uniform", args=(_MA, _MB - _MA))
        record("B2c", sensor, pname, f"estimated multiplier range [{m_hat.min():.3f}, {m_hat.max():.3f}]",
               [metric("ks_D", ks.statistic, "<=", 0.05, level="WARN")])

    n4_checks(sensor, pname, samples, W, luts, l_surf, l_sens, l_down, lsurf_hr, lsens_clean, tau_hr, lup_hr,
              usable, is_train)

    ref_mask = compute_band_mask(sensor, W, luts)
    missing = sum("band_mask" not in s for s in samples)
    wrong = sum("band_mask" in s and not np.array_equal(s["band_mask"].numpy(), ref_mask) for s in samples)
    record("B6", sensor, pname, f"{int((~ref_mask).sum())} of {len(ref_mask)} bands masked (tau < "
           f"{BAND_MASK_MODEL['tau_threshold']} in some development atmosphere)",
           [metric("samples_without_mask", missing, "==", 0), metric("samples_with_wrong_mask", wrong, "==", 0),
            metric("masked_bands", int((~ref_mask).sum()), "info")])

    lib_eps = np.stack([library[s["spectrum_id"]]["emissivity"] for s in samples])
    if is_train:
        scale = (eps_hr * lib_eps).sum(1) / (lib_eps * lib_eps).sum(1)
        resid = eps_hr - scale[:, None] * lib_eps
        r_std = float(resid[(eps_hr > 0) & (eps_hr < 1)].std())
        record("B4", sensor, pname, "training augmentation law",
               [metric("scale_min", scale.min(), ">=", _SA - 0.0005), metric("scale_max", scale.max(), "<=", _SB + 0.0005),
                metric("abs_resid_std_minus_jitter", abs(r_std - _JIT), "<=", max(0.15 * _JIT, 2e-4))] +
               ([metric("max_abs_eps_minus_library", float(np.abs(eps_hr - lib_eps).max()), "<=", 1e-6)]
                if (_SA, _SB, _JIT) == (1.0, 1.0, 0.0) else []))
    else:
        record("B4", sensor, pname, "pristine laboratory emissivity",
               [metric("max_abs_eps_minus_library", float(np.abs(eps_hr - lib_eps).max()), "<=", 1e-6)])


def n4_checks(sensor, pname, samples, W, luts, l_surf, l_sens, l_down, lsurf_hr, lsens_clean, tau_hr, lup_hr,
              usable, is_train):
    if "pwv_est" not in samples[0]:
        record("B7", sensor, pname, "no N4 fields (atmospheric-correction error disabled)", [metric("n4_fields", 0, "info")])
        return
    kt, ku = ("tau_aster", "L_up_aster") if sensor == "aster" else ("tau_hytes", "L_up_hytes")
    tau_in, lup_in = stack(samples, "tau_input"), stack(samples, "l_up_input")
    est = [estimated_atmosphere(luts, luts[s["atmosphere_id"]]["model_idx"], float(s["pwv"]) + float(s["pwv_error_drawn"]))
           for s in samples]
    tau_e = np.stack([e[kt] for e in est]) @ W.T
    lup_e = np.stack([e[ku] for e in est]) @ W.T
    ld_e = np.stack([e["L_down_surface"] for e in est]) @ W.T
    tau_t, lup_t = tau_hr @ W.T, lup_hr @ W.T
    ls_rec = lsurf_hr @ W.T + (l_sens - lup_in) / tau_in - (lsens_clean - lup_t) / tau_t

    def rel(a, b, cols=slice(None)):
        return float(np.max(np.abs(a[:, cols] - b[:, cols]) / np.maximum(np.abs(b[:, cols]), 1e-6)))
    record("B2d", sensor, pname, "inputs recomputed from the recorded PWV_est",
           [metric("max_rel_diff_tau_input", rel(tau_in, tau_e), "<=", 1e-4),
            metric("max_rel_diff_l_up_input", rel(lup_in, lup_e, usable), "<=", 1e-4),
            metric("max_rel_diff_l_down", rel(l_down, ld_e, usable), "<=", 1e-4),
            metric("max_rel_diff_l_surf_usable", rel(l_surf, ls_rec, usable), "<=", 1e-4)])

    drawn = np.array([float(s["pwv_error_drawn"]) for s in samples])
    sig = np.array([float(s["pwv_sigma"]) for s in samples])
    pwv = np.array([float(s["pwv"]) for s in samples])
    pwv_est = np.array([float(s["pwv_est"]) for s in samples])
    rec = np.array([e["pwv_used"] for e in est])
    clipped = float(np.mean([bool(s["pwv_est_clipped"]) for s in samples]))
    ks = stats.kstest(drawn / sig, "norm")
    ms = [metric("ks_pvalue_error_over_sigma", ks.pvalue, ">=", 0.01, level="WARN"),
          metric("max_abs_pwv_est_reconstruction", float(np.max(np.abs(pwv_est - rec))), "<=", 1e-9),
          metric("fraction_clipped_to_bounds", clipped, "info")]
    if is_train:
        ks_s = stats.kstest((10 * sig - _SIG_A) / (_SIG_B - _SIG_A), "uniform")
        ms.append(metric("ks_pvalue_sigma_uniform", ks_s.pvalue, ">=", 0.01, level="WARN"))
    else:
        ms.append(metric("max_abs_sigma_minus_eval_mm", float(np.max(np.abs(10 * sig - _SIG_EVAL))), "<=", 1e-9))
    record("B7", sensor, pname, f"drawn error mean {10 * drawn.mean():+.3f} mm, std {10 * drawn.std():.3f} mm; "
           f"mean |PWV_est - PWV| {10 * np.abs(pwv_est - pwv).mean():.3f} mm", ms)


# ----------------------------------------------------------------------------------------------
# A/C/D per sensor
# ----------------------------------------------------------------------------------------------
def c_checks(sensor, k, tr, va, dev_atm):
    cls = np.array([s["material_class"] for s in tr])
    atm = np.array([s["atmosphere_id"] for s in tr])
    temps = np.array([float(s["target_temp"][0]) for s in tr])
    pwv = np.array([float(s["pwv"]) for s in tr])
    scope = f"fold{k}"

    per_spec = Counter(s["spectrum_id"] for s in tr)
    record("C1", sensor, scope, f"{len(per_spec)} spectra x {sorted(set(per_spec.values()))} samples",
           [metric("distinct_samples_per_spectrum", len(set(per_spec.values())), "==", 1)])

    classes, atms = sorted(set(cls)), sorted(set(atm))
    table = np.array([[np.sum((cls == c) & (atm == a)) for a in atms] for c in classes])
    chi = stats.chi2_contingency(table)
    kw = stats.kruskal(*[temps[cls == c] for c in classes])
    rho = stats.spearmanr(temps, pwv).correlation
    record("C2", sensor, scope, f"{len(classes)} classes x {len(atms)} atmospheres",
           [metric("chi2_class_x_atm_p", chi.pvalue, ">=", 0.01, level="WARN"),
            metric("kruskal_T_by_class_p", kw.pvalue, ">=", 0.01, level="WARN"),
            metric("abs_spearman_T_PWV", abs(rho), "<=", 0.05, level="WARN")])

    ks = stats.kstest((temps - 275.0) / 60.0, "uniform")
    record("C3", sensor, scope, f"T range [{temps.min():.2f}, {temps.max():.2f}] K",
           [metric("ks_pvalue", ks.pvalue, ">=", 0.01, level="WARN")])

    counts = np.array([np.sum(atm == a) for a in sorted(dev_atm)])
    record("C4", sensor, scope, f"counts {counts.min()}-{counts.max()}",
           [metric("chi2_uniform_p", stats.chisquare(counts).pvalue, ">=", 0.01, level="WARN"),
            metric("min_count", counts.min(), ">", 0)])

    record("C5", sensor, scope, f"{len(tr)} training samples",
           [metric("max_share_one_spectrum_pct", 100 * max(per_spec.values()) / len(tr), "<=", 1.0, level="WARN", unit="%")])

    vc = Counter(s["material_class"] for s in va)
    missing = sorted(str(c) for c in set(cls) - set(vc))
    flagged = [c for c in missing if CLASS_GROUPS.get(c, 0) >= 5]
    record("D1", sensor, scope, f"absent from val: {[(c, CLASS_GROUPS.get(c, 0)) for c in missing]} (class, #groups)",
           [metric("flagged_classes", len(flagged), "==", 0, level="WARN"),
            metric("val_classes", len(vc), "info"), metric("train_classes", len(set(cls)), "info")])

    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    um = tr[0]["band_mask"].numpy().astype(bool)
    x_tr = np.log(np.clip(stack(tr, "input_radiance")[:, um], 1e-6, None))
    x_va = np.log(np.clip(stack(va, "input_radiance")[:, um], 1e-6, None))
    rng = np.random.default_rng(k)
    n = min(len(x_tr), len(x_va))
    X = np.vstack([x_tr[rng.choice(len(x_tr), n, replace=False)], x_va[rng.choice(len(x_va), n, replace=False)]])
    y = np.r_[np.zeros(n), np.ones(n)]
    steps = [StandardScaler()] + ([PCA(n_components=20, random_state=0)] if X.shape[1] > 20 else []) \
        + [LogisticRegression(max_iter=2000)]
    auc = cross_val_score(make_pipeline(*steps), X, y, cv=5, scoring="roc_auc").mean()
    record("C7", sensor, scope, "descriptive; see C7b", [metric("adversarial_auc", auc, "info")])


def adversarial_null_test(sensor, split, n_null, n_sub=600):
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pool = [s for k in range(len(split["folds"])) for s in load(f"val_fold{k}_{sensor}.pt")]
    um = pool[0]["band_mask"].numpy().astype(bool)
    X_all = np.log(np.clip(stack(pool, "input_radiance")[:, um], 1e-6, None))
    sid = np.array([s["spectrum_id"] for s in pool])
    dev = np.array(split["development_spectrum_ids"])
    cls = np.array([DEV_CLASS[d] for d in dev])
    grp = np.array([DEV_GROUP[d] for d in dev])

    def auc_for(val_ids, seed):
        rng = np.random.default_rng(seed)
        is_val = np.isin(sid, list(val_ids))
        iv, it = np.where(is_val)[0], np.where(~is_val)[0]
        n = min(n_sub, len(iv), len(it))
        X = np.vstack([X_all[rng.choice(it, n, replace=False)], X_all[rng.choice(iv, n, replace=False)]])
        y = np.r_[np.zeros(n), np.ones(n)]
        steps = [StandardScaler()] + ([PCA(n_components=20, random_state=0)] if X.shape[1] > 20 else []) \
            + [LogisticRegression(max_iter=2000)]
        return cross_val_score(make_pipeline(*steps), X, y, cv=3, scoring="roc_auc").mean()

    null = []
    for seed in range(1, n_null + 1):
        sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1000 + seed)
        for _, vi in sg.split(dev, cls, grp):
            null.append(auc_for(set(dev[vi]), seed))
    null = np.array(null)
    q95 = float(np.quantile(null, 0.95))
    for f in split["folds"]:
        obs = auc_for(set(f["val_spectrum_ids"]), 0)
        record("C7b", sensor, f"fold{f['fold_idx']}",
               f"null: {len(null)} folds from {n_null} random splits, median {np.median(null):.3f}; "
               f"observed at {100 * (null < obs).mean():.0f}th percentile",
               [metric("observed_auc", obs, "<=", q95, level="WARN")])
    return null.tolist()


def run_sensor(sensor, library, luts, split, golden, n_null):
    W, centers = build_sensor_srf_matrix(sensor)
    golden_ids = {g["spectrum_id"] for g in golden["golden_targets"]}
    golden_sids = {g["sample_id"] for g in golden["golden_targets"]}
    sib_ids = {s["spectrum_id"] for s in golden.get("quarantined_siblings", [])}
    golden_hash = {g["emissivity_hash"] for g in golden["golden_targets"]}
    dev_atm = {a for a, v in luts.items() if v["quarantine_status"] == "development"}
    gold_atm = {a for a, v in luts.items() if "golden" in v["quarantine_status"]}
    lib_hash = {sid: v.get("emissivity_hash") for sid, v in library.items()}
    q3_union = Counter()

    for f in split["folds"]:
        k = f["fold_idx"]
        if SEL["folds"] is not None and k not in SEL["folds"]:
            continue
        scope = f"fold{k}"
        tr, va = load(f"train_fold{k}_{sensor}.pt"), load(f"val_fold{k}_{sensor}.pt")
        q3 = load(f"quadrant_III_fold{k}_{sensor}.pt") if "q3" in SEL["partitions"] else []
        tr_s, va_s, q3_s = ({s["spectrum_id"] for s in p} for p in (tr, va, q3))
        tr_g, va_g = {s["sample_id"] for s in tr}, {s["sample_id"] for s in va}

        record("A1", sensor, scope, f"train {len(tr_s)} spectra/{len(tr_g)} specimens | val {len(va_s)}/{len(va_g)}",
               [metric("overlap_spectra", len(tr_s & va_s), "==", 0), metric("overlap_samples", len(tr_g & va_g), "==", 0)])
        record("A1b", sensor, scope, "",
               [metric("missing_train_spectra", len(set(f["train_spectrum_ids"]) ^ tr_s), "==", 0),
                metric("val_mismatch", len(set(f["val_spectrum_ids"]) ^ va_s), "==", 0)])
        if q3:
            record("A5", sensor, scope, f"{len(q3_s)} spectra, {len(q3)} samples",
                   [metric("mismatch", len(q3_s ^ va_s), "==", 0)])
        q3_union.update(q3_s)

        leaked = 0
        for part in [p for p in (tr, va, q3) if p]:
            ids = {s["spectrum_id"] for s in part}
            if ids & (golden_ids | sib_ids) or {s["sample_id"] for s in part} & golden_sids \
                    or {lib_hash.get(i) for i in ids} & golden_hash:
                leaked += 1
        record("A2", sensor, scope, "train/val/Q-III checked by ID, specimen and emissivity hash",
               [metric("leaking_partitions", leaked, "==", 0)])

        tr_a, va_a, q3_a = ({s["atmosphere_id"] for s in p} for p in (tr, va, q3))
        record("A4", sensor, scope, f"train {len(tr_a)} dev atm, val {len(va_a)} dev atm, Q-III {len(q3_a)} golden atm",
               [metric("golden_atm_in_train_val", len((tr_a | va_a) & gold_atm), "==", 0),
                *([metric("q3_non_golden_atm", len(q3_a - gold_atm), "==", 0),
                   metric("q3_atmospheres", len(q3_a), "==", len(gold_atm))] if q3 else [])])

        for pn, part, is_tr in [x for x in ((f"train_fold{k}", tr, True), (f"val_fold{k}", va, False), (f"q3_fold{k}", q3, False)) if x[1]]:
            physics_checks(sensor, pn, part, W, centers, luts, library, is_tr)
        c_checks(sensor, k, tr, va, dev_atm)

    if SEL["folds"] is not None or "q3" not in SEL["partitions"]:
        return []
    dev_ids = set(split["development_spectrum_ids"])
    record("A5b", sensor, "all folds", f"{len(q3_union)} of {len(dev_ids)} development spectra covered",
           [metric("uncovered", len(dev_ids - set(q3_union)), "==", 0),
            metric("max_multiplicity", max(q3_union.values()), "==", 1)])

    for pn, fname, atm_set, n_atm in [x for x in (("q2", f"golden_benchmark_quadrant_II_{sensor}.pt", dev_atm, 4),
                                      ("q4", f"golden_benchmark_quadrant_IV_{sensor}.pt", gold_atm, 10))
                                      if x[0] in SEL["partitions"]]:
        part = load(fname)
        cells = Counter((s["spectrum_id"], s["atmosphere_id"], round(float(s["target_temp"][0]), 3)) for s in part)
        atms = {s["atmosphere_id"] for s in part}
        record("A6", sensor, pn, f"{len(part)} samples",
               [metric("golden_targets", len({s['spectrum_id'] for s in part} & golden_ids), "==", 7),
                metric("atmospheres", len(atms), "==", n_atm),
                metric("atm_outside_allowed_set", len(atms - atm_set), "==", 0),
                metric("duplicate_cells", sum(v > 1 for v in cells.values()), "==", 0),
                metric("samples", len(part), "==", 7 * n_atm * 11)])
        physics_checks(sensor, pn, part, W, centers, luts, library, False)

    if SEL["folds"] is not None:
        return []
    return adversarial_null_test(sensor, split, n_null)


# ----------------------------------------------------------------------------------------------
# Library / LUT level
# ----------------------------------------------------------------------------------------------
def library_checks(library, split, golden, luts):
    ids = sorted(library)
    E = np.stack([library[i]["emissivity"] for i in ids])
    idx = {s: n for n, s in enumerate(ids)}
    dev_ids = split["development_spectrum_ids"]
    R = np.sqrt(((E[:, None, :] - E[None, :, :]) ** 2).mean(-1))
    np.fill_diagonal(R, np.inf)
    out = {}

    pairs = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))
             if R[i, j] < TWIN and library[ids[i]]["sample_id"] != library[ids[j]]["sample_id"]]
    record("A3a", "both", "library", f"pairs with RMSE < {TWIN} across different specimens",
           [metric("near_duplicate_pairs", len(pairs), "info")])

    rows = []
    for g in golden["golden_targets"]:
        r, s = min((R[idx[g["spectrum_id"]], idx[d]], d) for d in dev_ids)
        rows.append({"golden": g["spectrum_id"], "nearest_dev": s, "rmse": round(float(r), 4)})
    record("A3b", "both", "library", "; ".join(f"{x['golden']}->{x['nearest_dev']} {x['rmse']}" for x in rows),
           [metric("min_rmse_golden_to_dev", min(x["rmse"] for x in rows), ">=", TWIN)])
    out["golden_nearest_dev"] = rows

    for f in [f for f in split["folds"] if SEL["folds"] is None or f["fold_idx"] in SEL["folds"]]:
        hits = sum(min(R[idx[v], idx[t]] for t in f["train_spectrum_ids"]) < TWIN for v in f["val_spectrum_ids"])
        record("A3c", "both", f"fold{f['fold_idx']}", f"{len(f['val_spectrum_ids'])} val spectra",
               [metric("val_spectra_with_train_twin", hits, "==", 0)])

    dev_pwv = np.array([v["pwv_g_cm2"] for v in luts.values() if v["quarantine_status"] == "development"])
    gold_pwv = np.array([v["pwv_g_cm2"] for v in luts.values() if "golden" in v["quarantine_status"]])
    inside = int(np.sum((gold_pwv >= dev_pwv.min()) & (gold_pwv <= dev_pwv.max())))
    record("C8", "both", "atmospheres",
           f"dev PWV [{dev_pwv.min():.3f}, {dev_pwv.max():.3f}]; golden PWV [{gold_pwv.min():.3f}, {gold_pwv.max():.3f}] g/cm2",
           [metric("golden_profiles_inside_dev_pwv_range", inside, "info")])

    win = (WAVELENGTH_GRID >= 8.0) & (WAVELENGTH_GRID <= 12.0)
    mmd = {s: float(E[idx[s], win].max() - E[idx[s], win].min()) for s in ids}
    dev_mmd = np.array([mmd[s] for s in dev_ids])
    record("C9", "both", "library", "; ".join(
        f"{g['golden_target_name']}: MMD {mmd[g['spectrum_id']]:.3f} (p{stats.percentileofscore(dev_mmd, mmd[g['spectrum_id']]):.0f})"
        for g in golden["golden_targets"]), [metric("dev_median_mmd", np.median(dev_mmd), "info")])

    ratios, methods, configs = [], set(), set()
    for fpath in sorted(glob.glob(os.path.join(ROOT, "data", "processed", "atmospheric_luts", "profile_*.npz"))):
        z = np.load(fpath, allow_pickle=True)
        if "L_down_zenith" not in z.files:
            ratios.append(np.nan)
            continue
        m = (z["wavelength_um"] >= 8.0) & (z["wavelength_um"] <= 12.0)
        ratios.append(float(np.mean(z["L_down_surface"][m] / z["L_down_zenith"][m])))
        methods.add(str(z["downwelling_method"]))
        configs.add(str(z["rt_config"]))
    ratios = np.array(ratios)
    b8_check(luts)
    record("B5", "both", "30 LUTs", f"method={sorted(methods)}; rt_config={sorted(configs)}",
           [metric("hemi_over_zenith_min", np.nanmin(ratios) if np.isfinite(ratios).any() else 0.0, ">", 1.0),
            metric("hemi_over_zenith_max", np.nanmax(ratios) if np.isfinite(ratios).any() else 9.9, "<", 2.0),
            metric("luts_without_method", int(np.sum(~np.isfinite(ratios))), "==", 0),
            metric("n_methods", len(methods), "==", 1)])
    return out


def b8_check(luts):
    cat_path = os.path.join(N4_CHECK_DIR, "lut_catalog.csv")
    if not os.path.exists(cat_path):
        record("B8", "both", "n4 check grid", "check grid not found", [metric("check_profiles", 0, "info")])
        return
    cat = pd.read_csv(cat_path)
    inv = lambda lam, L: C2 / (lam * 1e-6 * np.log(1 + (C1 * 1e-6) / ((lam * 1e-6) ** 5 * np.clip(L, 1e-9, None))))
    for sensor in ("aster", "hytes"):
        W, c = build_sensor_srf_matrix(sensor)
        um = compute_band_mask(sensor, W, luts)
        kt, ku = ("tau_aster", "L_up_aster") if sensor == "aster" else ("tau_hytes", "L_up_hytes")
        B = planck_blackbody_grid(300.0)

        def dT(true, est, eps=0.95):
            ls = eps * B + (1 - eps) * true["L_down_surface"]
            lobs = W @ (true[kt] * ls + true[ku])
            lsurf = W @ ls + (lobs - W @ est[ku]) / (W @ est[kt]) - (lobs - W @ true[ku]) / (W @ true[kt])
            return (inv(c, (lsurf - (1 - eps) * (W @ est["L_down_surface"])) / eps) - inv(c, W @ B))[um]
        rat, err = [], []
        for r in cat.itertuples():
            if str(r.h2o_capped_at_saturation) == "True":
                continue
            z = np.load(os.path.join(N4_CHECK_DIR, f"profile_{r.profile_id}.npz"))
            true = {k: np.interp(WAVELENGTH_GRID, z["wavelength_um"], z[k]) for k in (kt, ku, "L_down_surface")}
            e_int = dT(true, estimated_atmosphere(luts, int(r.atm_model_idx), float(r.pwv_g_cm2)))
            base = estimated_atmosphere(luts, int(r.atm_model_idx), float(r.pwv_g_cm2))
            e_sig = dT(base, estimated_atmosphere(luts, int(r.atm_model_idx), float(r.pwv_g_cm2) + 0.1 * _SIG_EVAL))
            rat.append(np.sqrt(np.mean(e_int ** 2)) / max(np.sqrt(np.mean(e_sig ** 2)), 1e-9))
            err.append(np.sqrt(np.mean(e_int ** 2)))
        record("B8", sensor, "n4 check grid", f"{len(rat)} MODTRAN check profiles (non-capped)",
               [metric("max_rms_ratio_interp_over_sigma_effect", max(rat), "<=", B8_MAX_RATIO, level="WARN"),
                metric("median_rms_ratio", float(np.median(rat)), "info"),
                metric("max_rms_interp_error_K", max(err), "info", unit="K")])


# ----------------------------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------------------------
OPS_TXT = {"<=": "≤", "<": "<", ">=": "≥", ">": ">", "==": "=", "info": "—"}


def fmt(v):
    if v is None:
        return "—"
    if v == int(v) and abs(v) < 1e7:
        return f"{int(v)}"
    return f"{v:.3g}" if abs(v) < 1e-3 or abs(v) >= 1e4 else f"{v:.4f}"


def write_results_md(report, path):
    c = report["summary"]
    L = ["# Dataset Validation Report: OpenTIR-PINN v" + report["dataset_version"], "",
         f"Generated: {report['generated_at']} by `scripts/validate_dataset_integrity.py`. "
         f"Protocol: [`docs/dataset_validation_protocol.md`](../../docs/dataset_validation_protocol.md)", "",
         f"**Summary:** {c.get('PASS', 0)} PASS | {c.get('WARN', 0)} WARN | {c.get('FAIL', 0)} FAIL | {c.get('INFO', 0)} INFO", ""]
    for sec, title in SECTIONS.items():
        rs = [r for r in RESULTS if r["section"] == sec]
        if not rs:
            continue
        L += [f"## {sec}. {title}", "", "| Status | ID | Sensor | Scope | Metric | Obtained | Rule | Limit |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rs:
            for m in r["metrics"]:
                L.append(f"| {r['status']} | {r['id']} | {r['sensor']} | {r['scope']} | {m['name']} | {fmt(m['value'])} "
                         f"| {OPS_TXT[m['op']]} | {fmt(m['limit'])} |")
        L.append("")
    open(path, "w", encoding="utf-8").write("\n".join(L))


def write_protocol_md(path):
    L = ["# Dataset Validation Protocol: OpenTIR-PINN", "",
         "Auto-generated from `TEST_CATALOG` in `scripts/validate_dataset_integrity.py` (single source of the test "
         "definitions). Results: `data/validation/dataset_validation_report.md` and the dashboard tab "
         "*Dataset Validation Report*.", "",
         f"**Global constants:** NEdT (300 K) = {NEDT_BY_SENSOR} K; spectral-twin radius RMSE = {TWIN} "
         "(about 0.3 K of brightness temperature); expected training noise std = "
         f"{TRAIN_NOISE_STD:.3f} (sqrt E[m^2], m ~ U(0.5, 1.5)); atmospheric-correction PWV error (N4): "
         f"s = {_SIG_EVAL} mm (evaluation), s ~ U({_SIG_A}, {_SIG_B}) mm (training).", "",
         "**Status logic:** a test is FAIL if a FAIL-level limit is violated, WARN if only WARN-level limits are "
         "violated, INFO if it is descriptive, and PASS otherwise. FAIL blocks the dataset release; each WARN "
         "requires a documented PI decision.", ""]
    for sec, title in SECTIONS.items():
        L += [f"## {sec}. {title}", "", "| ID | Test | Purpose | Method | Pass rule | Rationale |", "|---|---|---|---|---|---|"]
        for cid, t in TEST_CATALOG.items():
            if t["section"] == sec:
                L.append(f"| {cid} | {t['name']} | {t['purpose']} | {t['method']} | `{t['rule']}` | {t['rationale']} |")
        L.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write("\n".join(L))


CLASS_GROUPS, DEV_CLASS, DEV_GROUP = {}, {}, {}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensors", nargs="+", default=["aster", "hytes"])
    ap.add_argument("--n_null", type=int, default=30)
    ap.add_argument("--dataset-dir", default=None, help="tensor folder (default data/processed/final_datasets)")
    ap.add_argument("--out-dir", default=None, help="report folder (default data/validation)")
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--partitions", nargs="+", default=None, choices=["train", "val", "q2", "q3", "q4"])
    args = ap.parse_args()
    if args.dataset_dir:
        DS = os.path.abspath(args.dataset_dir)
    if args.out_dir:
        OUT_DIR = os.path.abspath(args.out_dir)
    SEL["folds"] = args.folds
    if args.partitions:
        SEL["partitions"] = set(args.partitions) | {"train", "val"}
    os.makedirs(OUT_DIR, exist_ok=True)

    library = load_rebalanced_library()
    record("A0", "both", "library", f"{len(library)} spectra", [metric("aligned", 1, "==", 1)])
    cat = pd.read_csv(os.path.join(ROOT, "data", "processed", "spectral_libraries", "library_metadata_catalog.csv"))
    for r in cat.itertuples():
        library[r.spectrum_id]["emissivity_hash"] = r.emissivity_hash
    luts = load_atmospheric_luts()
    split = json.load(open(os.path.join(ROOT, "splits", "split_v1.1.json")))
    golden = json.load(open(os.path.join(ROOT, "splits", "golden_targets.json")))

    dev_df = cat[cat.spectrum_id.isin(split["development_spectrum_ids"])].set_index("spectrum_id") \
        .loc[split["development_spectrum_ids"]].reset_index()
    labels, _, _ = similarity_groups(dev_df, {s: library[s]["emissivity"] for s in dev_df.spectrum_id}, TWIN)
    DEV_CLASS.update(dict(zip(dev_df.spectrum_id, dev_df.material_class)))
    DEV_GROUP.update(dict(zip(dev_df.spectrum_id, labels)))
    CLASS_GROUPS.update(pd.Series(labels).groupby(dev_df.material_class.values).nunique().to_dict())

    facts_path = os.path.join(ROOT, "data", "dataset_facts.json")
    version = json.load(open(facts_path)).get("dataset_version", "?") if os.path.exists(facts_path) else "?"
    report = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"), "dataset_version": version,
              "constants": {"nedt_K": NEDT, "nedt_K_by_sensor": NEDT_BY_SENSOR, "spectral_twin_rmse": TWIN, "train_noise_std": TRAIN_NOISE_STD,
                            "atm_error_model": {k: (list(v) if isinstance(v, tuple) else v) for k, v in ATM_ERROR_MODEL.items()}},
              "class_independent_groups": {k: int(v) for k, v in CLASS_GROUPS.items()}}
    report["library"] = library_checks(library, split, golden, luts)
    report["adversarial_null"] = {s: run_sensor(s, library, luts, split, golden, args.n_null) for s in args.sensors}
    report["catalog"] = TEST_CATALOG
    report["sections"] = SECTIONS
    report["results"] = RESULTS
    report["summary"] = dict(Counter(r["status"] for r in RESULTS))
    with open(os.path.join(OUT_DIR, "dataset_validation_report.json"), "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False, default=str)
    write_results_md(report, os.path.join(OUT_DIR, "dataset_validation_report.md"))
    write_protocol_md(PROTOCOL_MD)
    print("\nSUMMARY:", report["summary"])
