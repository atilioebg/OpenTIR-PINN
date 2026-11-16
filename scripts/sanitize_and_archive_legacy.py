"""
Phase E.1: Repository Sanitation, Archival & Grounding Centralization
===================================================================
1. Creates local-only directory `archive_legacy/`.
2. Moves defunct preliminary run outputs (Run 0, Runs A/B/C) and obsolete analyses.
3. Updates `.gitignore` to ensure `archive_legacy/` is ignored by Git.
4. Creates `docs/grounding/` containing verified copies of SSoT documents D1-D12.
"""

import os
import shutil
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "archive_legacy")
GROUNDING_DIR = os.path.join(PROJECT_ROOT, "docs", "grounding")

os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(GROUNDING_DIR, exist_ok=True)

# 1. Write README_ARCHIVE.md
readme_archive = """# Archive Legacy & Preliminary Exploratory Iterations
===================================================
This directory archives historical development artifacts, exploratory training runs
(Run 0, Phase B, and Phase C Runs 1–3), and obsolete analyses generated prior to the
official SSoT release (Dataset v1.3.0 and Phase D Production Campaign).

These files are preserved strictly for internal provenance, auditability, and local
record-keeping. They are excluded from version control (.gitignore) and do not form
part of the official OpenTIR-PINN publication release.

Author: Dr. Atilio Grondona & OpenTIR-PINN Team
"""
with open(os.path.join(ARCHIVE_DIR, "README_ARCHIVE.md"), "w", encoding="utf-8") as f:
    f.write(readme_archive)
print("Created archive_legacy/README_ARCHIVE.md")

# 2. Move legacy directories
dirs_to_archive = [
    "outputs_phase_c_run1",
    "outputs_phase_c_run2",
    "outputs_phase_c_run3",
    "outputs_baseline_g6",
    "OLD_PAPERS",
    os.path.join("docs", "analysis_BC")
]

for d in dirs_to_archive:
    src = os.path.join(PROJECT_ROOT, d)
    if os.path.exists(src):
        dst = os.path.join(ARCHIVE_DIR, os.path.basename(d))
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)
        print(f"Moved directory: {d} -> archive_legacy/{os.path.basename(d)}")

# 3. Move legacy files
files_to_archive = [
    "analysis_grounding.md",
    "analysis_grounding_runs_BC.md"
]

for f in files_to_archive:
    src = os.path.join(PROJECT_ROOT, f)
    if os.path.exists(src):
        dst = os.path.join(ARCHIVE_DIR, f)
        shutil.move(src, dst)
        print(f"Moved file: {f} -> archive_legacy/{f}")

# 4. Update .gitignore
gitignore_path = os.path.join(PROJECT_ROOT, ".gitignore")
with open(gitignore_path, "r", encoding="utf-8") as gf:
    gi_content = gf.read()

if "archive_legacy/" not in gi_content:
    with open(gitignore_path, "a", encoding="utf-8") as gf:
        gf.write("\n# Local Legacy Archive (Phase E.1 Quarantine)\narchive_legacy/\n")
    print("Updated .gitignore with archive_legacy/")
else:
    print(".gitignore already contains archive_legacy/")

# 5. Populate docs/grounding/ with verified SSoT documents
grounding_files = [
    ("data/dataset_facts.json", "D1_dataset_facts.json"),
    ("docs/PIPELINE_README.md", "D2_PIPELINE_README.md"),
    ("docs/perturbation_model_error_sources.md", "D3_perturbation_model_error_sources.md"),
    ("docs/modtran_downwelling_decision_v112.md", "D4_modtran_downwelling_decision.md"),
    ("data/processed/modtran_cards/README_MODTRAN_CARDS.md", "D5_README_MODTRAN_CARDS.md"),
    ("data/validation/dataset_validation_report.md", "D6_dataset_validation_report.md"),
    ("pinn_grounding.md", "D8_pinn_grounding.md"),
    ("outputs/best_config_aster.json", "D11_best_config_aster.json"),
    ("outputs/best_config_hytes.json", "D11_best_config_hytes.json"),
    ("analysis_grounding_run_D.md", "D12_analysis_grounding_run_D.md"),
]

for src_rel, dst_name in grounding_files:
    src = os.path.join(PROJECT_ROOT, src_rel)
    dst = os.path.join(GROUNDING_DIR, dst_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied SSoT grounding doc: {src_rel} -> docs/grounding/{dst_name}")

# Write docs/grounding/README_GROUNDING.md
grounding_readme = """# Centralized Single Source of Truth (SSoT) Grounding Register
==============================================================
This directory consolidates the authoritative, float32-audited grounding documents
supporting every equation, number, parameter, and figure presented in the manuscript
and public Zenodo benchmark.

| ID | Grounding Document | Scope & Authority | Feeds Section |
|---|---|---|---|
| **D1** | `D1_dataset_facts.json` | Numerical SSoT for every count, PWV, noise, partition | Section III (all), Tables T1–T4 |
| **D2** | `D2_PIPELINE_README.md` | Pipeline stages, sources, replication protocol | Section III-A, III-C, Pipeline Fig |
| **D3** | `D3_perturbation_model_error_sources.md` | Citation-gated perturbation laws (N4, NEΔT, CO₂) | Section III-D, III-E, Limitations |
| **D4** | `D4_modtran_downwelling_decision.md` | Hemispheric downwelling sky radiance convention ($F_\downarrow/\pi$) | Section III-C, Section II |
| **D5** | `D5_README_MODTRAN_CARDS.md` | MODTRAN 5.2 DISORT-8 simulation cards and saturation cap | Section III-C, Table T2 |
| **D6** | `D6_dataset_validation_report.md` | 32-test protocol (402 PASS / 1 WARN / 0 FAIL) | Section III-I, Supplement |
| **D8** | `D8_pinn_grounding.md` | Sobolev seminorms ($W^{1,1}, W^{2,1}$), RTE physics loss | Section IV |
| **D11** | `D11_best_config_*.json` | Converged Optuna hyperparameters (SiLU, 384x5 / 768x3) | Section IV-D, Section V |
| **D12** | `D12_analysis_grounding_run_D.md` | Comprehensive Run D analysis vs. Classical TES (28 tables, 19 figures) | Section V, Section VI |

All legacy files (Run 0, Runs A/B/C) have been quarantined locally in `archive_legacy/`
and excluded from version control.
"""

with open(os.path.join(GROUNDING_DIR, "README_GROUNDING.md"), "w", encoding="utf-8") as f:
    f.write(grounding_readme)
print("Created docs/grounding/README_GROUNDING.md")
print("\nPhase E.1 Sanitation, Archival & Grounding Centralization Complete!")

