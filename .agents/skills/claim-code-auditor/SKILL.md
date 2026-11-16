---
name: claim-code-auditor
description: Rigorously audits academic paper claims against the repository's code, parameters, datasets, and execution logs, classifying claims into VERIFIED, CONTRADICTED, or UNIMPLEMENTED to eliminate hallucinations.
---

# Claim-Code Auditor Skill

## Overview
This skill implements an automated, forensic code-grounding auditor for scientific manuscripts in GeoAI, Remote Sensing, and Physics-Informed Machine Learning. It ensures **Strict Zero Hallucination**: every statement made in the paper (equations, optimizer parameters, loss weights, dataset compositions, physical forward models, hardware benchmarks, and error metrics) must map 1:1 to verifiable source code, configuration files, or empirical data logs.

---

## When to Activate
Activate this skill whenever:
* Preparing a manuscript section for submission or review (specifically Sections III, IV, and V).
* Fact-checking claims such as:
  * *"The model was trained with AdamW, learning rate 1e-3, and cosine annealing."*
  * *"Training dataset contains at-sensor TOA radiance simulated with MODTRAN 6."*
  * *"Validation was performed on an 80/20 random split."*
* Detecting discrepancies between the manuscript text and the repository implementation before Reviewer #2 finds them.
* Reconciling manuscript text with actual execution logs following retraining or hyperparameter optimization.

---

## Audit Classification Triad

Every analyzed paragraph or claim is audited against the active codebase and assigned one of three statuses:

1. **`[VERIFIED]`**:
   * Exact match with active repository code, configuration dictionary, or data tensor.
   * *Example:* "Network architecture consists of 4 dense layers with 512 units and GELU activations" $\longrightarrow$ Matches `pinn.py` line 24 (`nn.Linear(512, 512)`, `nn.GELU()`).

2. **`[CONTRADICTED]`**:
   * The manuscript makes a claim, but the codebase implements a different parameter, algorithm, or mathematical formulation.
   * *Example:* "Trained using AdamW with cosine learning rate schedule" $\longrightarrow$ Code uses `torch.optim.Adam` with `ReduceLROnPlateau`.
   * *Action:* Flag immediately with the exact file and line number; rewrite the manuscript to reflect true code facts or update the code if the claim was the intended design.

3. **`[UNIMPLEMENTED]`**:
   * The manuscript describes a feature, loss term, or baseline that does not exist anywhere in the repository.
   * *Example:* "Coupled with forward RTE physics loss layer" or "compared against operational TES baseline" when no such functions or baselines are implemented in `src/`.
   * *Action:* Lock paragraph from final submission until implemented in Phase 2 or strike from text.

---

## Core Audit Domains & Verification Targets

### 1. Optimization & Hyperparameters
* **Paper Claims:** Optimizer, learning rate ($\eta$), weight decay, scheduler (Plateau vs Cosine), batch size, epoch budget, initialization (Kaiming vs PyTorch default).
* **Target Code:** `src/training/base/train_generalist.py`, `src/training/fine_tuning/train_refine_*.py`.

### 2. Loss Functions & Regularization Weights
* **Paper Claims:** Heterogeneous loss weights ($w_{\mathrm{mse\_eps}}$, $w_{\mathrm{mse\_temp}}$, $w_{\mathrm{sam}}$, $w_{\mathrm{grad}}$, $w_{\mathrm{curve}}$, $w_{\mathrm{phys}}$).
* **Target Code:** `src/training/loss/spectral_loss.py`, `src/utils/physics_utils.py`.

### 3. Data Pipeline, Splits & Atmosphere
* **Paper Claims:** Radiance type (surface-leaving vs TOA), atmospheric correction, MODTRAN version and profiles, PWV range, Golden Target quarantine, cross-validation grouping.
* **Target Code:** `src/data_generation/`, `data/processed/`, `.pt` datasets, split index JSONs.

### 4. Metrics & Hardware Telemetry
* **Paper Claims:** Temperature error ($\Delta T_s$, RMSE), spectral angle mapper (SAM), curvature, inference latency, wall-clock training time.
* **Target Logs:** `outputs/*.csv`, `outputs/run_log_*.json`, benchmark scripts.

### 5. Causal-Claim Gate (Mechanism & Ablation Grounding)
* **Paper Claims:** Any causal attribution linking a specific architectural element, input mode, or loss term ($w_{\mathrm{grad}}$, $w_{\mathrm{curv}}$, $w_{\mathrm{phys}}$, $w_{\mathrm{sam}}$, atmospheric inputs) to a performance outcome using causal verbs (*"demonstrates"*, *"causes"*, *"because of"*, *"due to"*, *"eliminates"*).
* **Verification Rule:** A causal verb is **`[VERIFIED]`** only if a controlled ablation table in `docs/analysis_D/tables/` or `outputs/benchmarks/` isolates that specific component and shows a statistically significant effect. Otherwise, the causal verb is classified as **`[CONTRADICTED — UNGROUNDED CAUSALITY]`** and must be rewritten as a physical hypothesis (*"is consistent with"*, *"we hypothesise that"*).

---

## Execution Protocol

Run the automated Python auditor script:
```bash
python .agents/skills/claim-code-auditor/scripts/audit_code_grounding.py --docx <path_to_manuscript>
```
Or execute targeted audits using `grep_search` and `view_file` to verify candidate claims before proposing text updates.
