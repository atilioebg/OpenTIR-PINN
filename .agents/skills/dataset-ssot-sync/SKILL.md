---
name: dataset-ssot-sync
description: Regenerates the Single Source of Truth manifest (data/dataset_facts.json), runs the deep dataset integrity/leakage/statistics validator, and synchronizes every dataset number in README, DATA_DICTIONARY, dashboard, clarification.md, and the manuscript against it. Use after any change to the spectral library, split, atmospheric LUTs, or forward synthesizer, and before quoting dataset numbers anywhere.
---

# Dataset SSoT Sync Skill

## When to activate
- After regenerating the library, the split (`splits/*.json`), the LUTs, or any `.pt` partition.
- Before editing any dataset description in docs, the dashboard, or `manuscript/main.tex`.
- When a reviewer or user questions a dataset number.

## Procedure
1. **Rebuild the manifest:**
   `python scripts/build_dataset_facts.py`
   This writes `data/dataset_facts.json` and `zenodo_release/data/dataset_facts.json`.
2. **Validate:**
   `python scripts/validate_dataset_integrity.py`
   This writes `data/validation/dataset_validation_report.{md,json}`.
   - Any **FAIL** blocks the release. Fix the generator, then go back to step 1.
   - Each **WARN** needs a written PI decision (accept with a documented limitation, or fix).
3. **Drift scan:** grep the documentation targets for every canonical number and identity in the manifest:
   - `canonical_counts_fold0`
   - `library.n_*`
   - `atmospheres.*.n_profiles`
   - PWV ranges
   - golden target `name`/`sample_id`
   - `generator_parameters`

   Also grep for known stale literals. These include 16,204, 17,204, 8,602, 5,000 samples, "25 dev", "Model 5 … Development", Albite, "PWV < 0.3", "PWV < 0.85", −79.3%, "Leave-One-Climatology/Cluster-Out", and "L↓/π".

   Targets: `zenodo_release/README.md`, `zenodo_release/DATA_DICTIONARY.md`, `DATA_DICTIONARY.md`, `zenodo_release/DASHBOARD_USER_GUIDE.md`, `docs/DASHBOARD_USER_GUIDE.md`, both `dataset_explorer_dashboard.py` copies, `scripts/package_zenodo_v1.1.0.py`, `clarification.md`, `manuscript/main.tex`.
4. **Fix by reading, not typing:** dashboard and packager code must load the manifest. Markdown edits must copy values verbatim from the manifest.
5. **Log:** add or update a `clarification.md` entry that lists every corrected statement (old → new, with its source).
6. **Package:** rebuild the Zenodo archive only after steps 1–5 pass. Confirm with `python scripts/build_dataset_facts.py --check` that the result is `UP-TO-DATE`.

## Report format to the PI
A table with columns: Check | Status | Evidence (file:line or manifest key) | Action.

