#!/usr/bin/env python
"""
OpenTIR-PINN: end-to-end reproducible data pipeline (raw spectra -> train / validation / test tensors)
======================================================================================================

    python run_pipeline.py                      # full pipeline with the shipped MODTRAN LUTs
    python run_pipeline.py --run-modtran        # also re-execute the 330 published MODTRAN decks and verify the
                                                # shipped LUTs are reproduced (requires a licensed MODTRAN 5.2)
    python run_pipeline.py --stages curate split # run selected stages only
    python run_pipeline.py --n-null 30          # size of the adversarial null distribution (validation test C7b)

Stage order and why it is this order
------------------------------------
 0 setup       Create the folder tree; in release mode, unpack `inputs/` (raw spectra, SRFs, LUTs).
 1 atmosphere  Verify the MODTRAN LUT checksums; with --run-modtran, re-execute the published decks
               (inputs/modtran_cards) and compare the rebuilt LUTs with the shipped ones.
 2 curate      Parse raw ECOSTRESS spectra, apply the three validation gates (coverage, modality,
               physical bounds), convert reflectance to emissivity (Kirchhoff), drop exact duplicates,
               quarantine the 7 golden targets and their same-specimen / spectral siblings.
 3 split       Group-aware stratified 5-fold split of the development pool. Groups are connected
               components of (same physical sample) OR (emissivity RMSE < 0.005).
 4 synthesize  Forward radiative transfer per fold. Augmentation and noise are applied only AFTER the
               split and only to the training partition of each fold; validation, Q-II, Q-III and
               Q-IV use pristine laboratory emissivity with nominal sensor noise.
 5 facts       Build the Single-Source-of-Truth manifest data/dataset_facts.json from the tensors.
 6 validate    Run the 32-test integrity / leakage / statistical protocol. Any FAIL makes the
               pipeline exit with code 1.
 7 reproduce   (release mode) Compare the outputs with REFERENCE_CHECKSUMS.json shipped with the data.

Split BEFORE augmentation is mandatory: augmenting first would put perturbed copies of the same
laboratory spectrum on both sides of the split (augmentation leakage) and would inflate validation
scores. The validator enforces this ordering indirectly (tests A1, A3c, B4).
"""
import os
import sys
import json
import time
import glob
import shutil
import hashlib
import zipfile
import argparse
import datetime
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
STAGES = ["setup", "atmosphere", "curate", "split", "synthesize", "facts", "validate", "reproduce"]
SMOKE_STAGES = ["setup", "atmosphere", "curate", "split", "synthesize", "validate", "train_smoke"]
SMOKE_DIR = os.path.join(ROOT, "data", "smoke")

PATHS = {
    "raw_spectra": os.path.join(ROOT, "data", "raw", "ecospeclib", "ecospeclib-all"),
    "library": os.path.join(ROOT, "data", "processed", "spectral_libraries"),
    "luts": os.path.join(ROOT, "data", "processed", "atmospheric_luts"),
    "datasets": os.path.join(ROOT, "data", "processed", "final_datasets"),
    "validation": os.path.join(ROOT, "data", "validation"),
    "splits": os.path.join(ROOT, "splits"),
    "aster_srf": os.path.join(ROOT, "bibliografic_references", "aster"),
    "hytes_params": os.path.join(ROOT, "data", "processed", "hytes_real_parameters.csv"),
    "logs": os.path.join(ROOT, "logs"),
    "docs": os.path.join(ROOT, "docs"),
    "inputs": os.path.join(ROOT, "inputs"),
}
LOG = None


def log(msg=""):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG:
        LOG.write(line + "\n")
        LOG.flush()


def run(cmd, stage):
    log(f"  $ {' '.join(os.path.relpath(c, ROOT) if os.path.isabs(c) and c.startswith(ROOT) else c for c in cmd)}")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore")
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env=env)
    for line in p.stdout:
        line = line.rstrip()
        if line and "%|" not in line:  # drop progress bars
            log(f"    {line}")
    if p.wait() != 0:
        raise RuntimeError(f"stage '{stage}' failed (exit code {p.returncode})")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fp:
        for b in iter(lambda: fp.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()


# ---------------------------------------------------------------------------------------------
def stage_setup(args):
    for k in ("raw_spectra", "library", "luts", "datasets", "validation", "splits", "aster_srf", "logs", "docs"):
        os.makedirs(PATHS[k], exist_ok=True)
    inp = PATHS["inputs"]
    if os.path.isdir(inp):
        log("  release mode: populating working folders from inputs/")
        z = os.path.join(inp, "spectral_library_raw.zip")
        if not glob.glob(os.path.join(PATHS["raw_spectra"], "*.txt")):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(PATHS["raw_spectra"])
            log(f"  extracted {len(glob.glob(os.path.join(PATHS['raw_spectra'], '*.txt')))} raw spectra")
        for f in glob.glob(os.path.join(inp, "srf", "aster", "*.txt")):
            shutil.copy(f, PATHS["aster_srf"])
        shutil.copy(os.path.join(inp, "srf", "hytes", "hytes_real_parameters.csv"), PATHS["hytes_params"])
        # the shipped LUTs are always installed; --run-modtran only adds a replication check against them
        for f in glob.glob(os.path.join(inp, "atmospheric_luts", "*")):
            shutil.copy(f, PATHS["luts"])
        chk = os.path.join(inp, "n4_interp_check")
        if os.path.isdir(chk):  # MODTRAN check grid for validator test B8 (N4 interpolation fidelity)
            shutil.copytree(chk, os.path.join(PATHS["validation"], "n4_interp_check"), dirs_exist_ok=True)
    missing = [k for k in ("raw_spectra", "aster_srf") if not os.listdir(PATHS[k])]
    if not os.path.exists(PATHS["hytes_params"]):
        missing.append("hytes_params")
    if missing:
        raise RuntimeError(f"missing required inputs: {missing}")
    log(f"  raw spectra: {len(os.listdir(PATHS['raw_spectra']))} files | ASTER SRFs: {len(os.listdir(PATHS['aster_srf']))} | "
        f"HyTES parameters: ok")


def stage_atmosphere(args):
    import csv
    cat = os.path.join(PATHS["luts"], "lut_catalog.csv")
    rows = list(csv.DictReader(open(cat, encoding="utf-8")))
    bad = [r["profile_id"] for r in rows if md5(os.path.join(PATHS["luts"], r["csv_file"])) != r["md5_hash"]]
    if bad or len(rows) != 30:
        raise RuntimeError(f"LUT verification failed: n={len(rows)}, checksum mismatch={bad}")
    log(f"  30/30 MODTRAN LUTs verified against lut_catalog.csv (MD5)")
    if args.run_modtran:
        # Replication (validation) run: re-execute every documented deck with the user's licensed MODTRAN,
        # rebuild the 30 LUTs in a separate folder and compare them with the shipped LUTs (never overwritten).
        rep_dir = os.path.join(PATHS["validation"], "lut_replication")
        run([PY, os.path.join(ROOT, "src", "data_generation", "base", "automate_modtran.py"),
             "--out", rep_dir, "--archive-decks", os.path.join(PATHS["logs"], "modtran_decks_executed"),
             "--compare-to", PATHS["luts"]], "atmosphere")
        log(f"  LUT replication report: {os.path.relpath(os.path.join(rep_dir, 'lut_replication_report.csv'), ROOT)}")
        ref_idx = os.path.join(PATHS["inputs"], "modtran_cards", "deck_index.csv")
        new_idx = os.path.join(PATHS["logs"], "modtran_decks_executed", "deck_index.csv")
        if os.path.exists(ref_idx):
            import pandas as pd
            a = pd.read_csv(ref_idx).set_index("tag")["deck_md5"]
            b = pd.read_csv(new_idx).set_index("tag")["deck_md5"]
            same = a.reindex(b.index).eq(b).all() and len(a) == len(b)
            log(f"  executed decks identical to the published decks (MD5): {bool(same)}")
            if not same:
                raise RuntimeError("executed MODTRAN decks differ from the published decks")


def stage_curate(args):
    run([PY, os.path.join(ROOT, "src", "data_generation", "base", "curate_spectral_library.py")], "curate")


def stage_split(args):
    run([PY, os.path.join(ROOT, "src", "data_generation", "base", "make_split.py")], "split")


def stage_synthesize(args):
    if args.smoke:
        # Smoke profile: raw -> split -> TRAINING-SIDE partitions only (train + out-of-fold val) for the selected
        # folds, written to data/smoke/ so the frozen dataset and the sealed benchmark vaults are never touched.
        run([PY, os.path.join(ROOT, "src", "data_generation", "base", "synthesize_forward_dataset.py"),
             "--sensor", args.smoke_sensor, "--seed", str(args.seed), "--folds", *map(str, args.smoke_folds),
             "--partitions", "train", "val", "--samples_per_spectrum", str(args.smoke_samples),
             "--out_dir", os.path.join(SMOKE_DIR, "final_datasets")], "synthesize")
        return
    run([PY, os.path.join(ROOT, "src", "data_generation", "base", "synthesize_forward_dataset.py"),
         "--sensor", "both", "--seed", str(args.seed)], "synthesize")


def _require_tensors():
    if not glob.glob(os.path.join(PATHS["datasets"], "train_fold0_*.pt")):
        raise RuntimeError("no generated tensors in data/processed/final_datasets/: run the full pipeline first "
                           "(python run_pipeline.py, or the dashboard mode 'Full pipeline from raw data'); "
                           "'--stages facts validate' only re-checks existing tensors")


def stage_facts(args):
    _require_tensors()
    run([PY, os.path.join(ROOT, "scripts", "build_dataset_facts.py")], "facts")


def stage_validate(args):
    if args.smoke:
        sensors = ["aster", "hytes"] if args.smoke_sensor == "both" else [args.smoke_sensor]
        out = os.path.join(SMOKE_DIR, "validation")
        run([PY, os.path.join(ROOT, "scripts", "validate_dataset_integrity.py"), "--sensors", *sensors,
             "--dataset-dir", os.path.join(SMOKE_DIR, "final_datasets"), "--out-dir", out,
             "--folds", *map(str, args.smoke_folds), "--partitions", "train", "val"], "validate")
        rep = json.load(open(os.path.join(out, "dataset_validation_report.json"), encoding="utf-8"))
        log(f"  smoke validation summary: {rep['summary']}")
        if rep["summary"].get("FAIL", 0):
            raise RuntimeError(f"smoke validation FAILED: {rep['summary']['FAIL']} failing results")
        return
    _require_tensors()
    run([PY, os.path.join(ROOT, "scripts", "validate_dataset_integrity.py"), "--n_null", str(args.n_null)], "validate")
    rep = json.load(open(os.path.join(PATHS["validation"], "dataset_validation_report.json"), encoding="utf-8"))
    log(f"  validation summary: {rep['summary']}")
    if rep["summary"].get("FAIL", 0):
        raise RuntimeError(f"validation FAILED: {rep['summary']['FAIL']} failing results (see data/validation/)")


def stage_train_smoke(args):
    """Plumbing check of the training code on the smoke tensors (few epochs, isolated outputs)."""
    if args.smoke_epochs <= 0:
        log("  --smoke-epochs 0: training smoke test skipped")
        return
    trainer = os.path.join(ROOT, "src", "training", "train_cross_validation.py")
    if not os.path.exists(trainer):
        log("  training code not present in this folder (release kit): training smoke test skipped")
        return
    if args.smoke_folds != [0]:
        log("  note: the trainer runs folds 0..n-1; only fold 0 is used here")
    env_extra = {"OPENTIR_DATASET_DIR": os.path.join(SMOKE_DIR, "final_datasets"),
                 "OPENTIR_OUTPUT_DIR": os.path.join(SMOKE_DIR, "outputs")}
    os.environ.update(env_extra)
    sensors = ["aster", "hytes"] if args.smoke_sensor == "both" else [args.smoke_sensor]
    for s in sensors:
        run([PY, trainer, "--sensor", s, "--epochs", str(args.smoke_epochs), "--n_folds", "1"], "train_smoke")
        csv_path = os.path.join(SMOKE_DIR, "outputs", "benchmarks", f"cv_benchmark_{s}.csv")
        if os.path.exists(csv_path):
            import pandas as pd
            row = pd.read_csv(csv_path).iloc[0]
            finite = all(pd.notna(row[c]) and abs(float(row[c])) < 1e6 for c in ("temp_rmse", "eps_rmse", "val_loss"))
            log(f"  {s}: fold-0 smoke metrics temp_rmse={row['temp_rmse']:.3f} K eps_rmse={row['eps_rmse']:.4f} "
                f"val_loss={row['val_loss']:.4f} | finite={finite}")
            if not finite:
                raise RuntimeError(f"training smoke test produced non-finite metrics for {s}")


def stage_reproduce(args):
    ref_path = os.path.join(ROOT, "REFERENCE_CHECKSUMS.json")
    if not os.path.exists(ref_path):
        log("  no REFERENCE_CHECKSUMS.json (repository mode) - skipped")
        return
    ref = json.load(open(ref_path, encoding="utf-8"))
    facts = json.load(open(os.path.join(ROOT, "data", "dataset_facts.json"), encoding="utf-8"))
    import pandas as pd
    cat = pd.read_csv(os.path.join(PATHS["library"], "library_metadata_catalog.csv"))
    split = json.load(open(os.path.join(PATHS["splits"], "split_v1.1.json"), encoding="utf-8"))
    checks = {
        "library_emissivity_hashes": dict(zip(cat.spectrum_id, cat.emissivity_hash)) == ref["library_emissivity_hashes"],
        "split_folds": [sorted(f["val_spectrum_ids"]) for f in split["folds"]] == ref["split_val_spectrum_ids"],
        "canonical_counts": {k: v for k, v in facts["canonical_counts_fold0"].items() if k != "plain_language"}
                            == ref["canonical_counts_fold0"],
        "partition_md5": {s: {p: d["md5"] for p, d in parts.items()} for s, parts in facts["partitions"].items()}
                         == ref["partition_md5"],
    }
    for k, v in checks.items():
        log(f"  {'MATCH   ' if v else 'MISMATCH'} {k}")
    json.dump(checks, open(os.path.join(PATHS["logs"], "reproducibility_check.json"), "w"), indent=2)
    if not all(checks.values()):
        raise RuntimeError("outputs differ from the reference release (see logs/reproducibility_check.json)")


# ---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=STAGES + ["train_smoke"], default=None)
    ap.add_argument("--run-modtran", action="store_true", help="regenerate LUTs with a local licensed MODTRAN 5.2")
    ap.add_argument("--n-null", type=int, default=30, help="adversarial null-distribution size (test C7b)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="smoke profile: raw -> curate -> split -> train/val partitions only (selected folds, reduced "
                         "samples) in data/smoke/, smoke validation, then a short training run; benchmark vaults untouched")
    ap.add_argument("--smoke-folds", type=int, nargs="+", default=[0])
    ap.add_argument("--smoke-samples", type=int, default=10, help="training samples per spectrum in smoke mode")
    ap.add_argument("--smoke-sensor", default="both", choices=["aster", "hytes", "both"])
    ap.add_argument("--smoke-epochs", type=int, default=3, help="training epochs in smoke mode (0 = skip training)")
    args = ap.parse_args()
    if args.stages is None:
        args.stages = SMOKE_STAGES if args.smoke else STAGES
    order = SMOKE_STAGES if args.smoke else STAGES

    os.makedirs(PATHS["logs"], exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG = open(os.path.join(PATHS["logs"], f"pipeline_{stamp}.log"), "w", encoding="utf-8")
    summary = {"started": stamp, "stages": {}, "python": sys.version.split()[0],
               "profile": "smoke" if args.smoke else "full"}
    log(f"OpenTIR-PINN data pipeline | root = {ROOT} | profile = {summary['profile']}")
    ok = True
    for st in [s for s in order if s in args.stages]:
        t0 = time.time()
        log(f"=== stage {order.index(st)}: {st} ===")
        try:
            globals()[f"stage_{st}"](args)
            summary["stages"][st] = {"status": "ok", "seconds": round(time.time() - t0, 1)}
        except Exception as exc:
            summary["stages"][st] = {"status": "failed", "error": str(exc), "seconds": round(time.time() - t0, 1)}
            log(f"!!! {exc}")
            ok = False
            break
    summary["finished"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary["result"] = "SUCCESS" if ok else "FAILED"
    json.dump(summary, open(os.path.join(PATHS["logs"], "pipeline_summary.json"), "w"), indent=2)
    log(f"PIPELINE {summary['result']} | " + ", ".join(f"{k}: {v['status']} ({v['seconds']} s)" for k, v in summary["stages"].items()))
    LOG.close()
    sys.exit(0 if ok else 1)

