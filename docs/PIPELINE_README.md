# OpenTIR-PINN Data Pipeline: From Raw Laboratory Spectra to Train/Validation/Test Tensors

This folder contains everything needed to rebuild the benchmark dataset from the raw inputs:
- the generation code (verbatim copy of the repository, traceable through `CODE_MANIFEST.json`);
- the raw inputs (`inputs/`);
- the validation protocol;
- the reference checksums of the published build.

## 1. Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py                                  # full pipeline (about 10 min on a laptop CPU)
```
`run_pipeline.py` exits with code **0** only if every stage succeeds. That requires zero FAIL results in the validation protocol and, in release mode, outputs identical to `REFERENCE_CHECKSUMS.json`.

| Option | Effect |
|---|---|
| `--stages curate split ...` | Run selected stages only |
| `--run-modtran` | Regenerate the 30 atmospheric LUTs. Needs a **licensed MODTRAN 5.2** (`OPENTIR_MODTRAN_DIR` = PcModWin5 `bin` folder). By default the shipped LUTs are verified and used |
| `--n-null 30` | Size of the null distribution used by validation test C7b |
| `--seed 42` | Master seed (the reference build uses 42) |

## 2. Stages and their order
```
0 setup ─► 1 atmosphere ─► 2 curate ─► 3 split ─► 4 synthesize ─► 5 facts ─► 6 validate ─► 7 reproduce
```
| # | Stage | Code | Output |
|---|---|---|---|
| 0 | Create the folder tree and unpack `inputs/` | `run_pipeline.py` | `data/raw/...`, `bibliografic_references/aster/`, `data/processed/hytes_real_parameters.csv` |
| 1 | Verify (or regenerate) MODTRAN LUTs: thermal radiance, DISORT 8-stream, hemispheric L↓ = F↓/π | `src/data_generation/base/automate_modtran.py` | `data/processed/atmospheric_luts/` |
| 2 | Curate the library: three gates (spectral coverage, transmission-modality rejection, physical bounds); Kirchhoff conversion; exact-duplicate removal; golden-target and sibling quarantine | `curate_spectral_library.py` | `data/processed/spectral_libraries/` |
| 3 | Group-aware stratified 5-fold split. Groups = (same specimen) OR (emissivity RMSE < 0.005) | `make_split.py` | `splits/split_v1.1.json`, `splits/golden_targets.json` |
| 4 | Forward RTE synthesis per fold: training augmentation + sensor noise; pristine validation and benchmark partitions (Q-II, Q-III, Q-IV) | `synthesize_forward_dataset.py` | `data/processed/final_datasets/*.pt` |
| 5 | Single-Source-of-Truth manifest (counts, ranges and checksums read from the tensors) | `scripts/build_dataset_facts.py` | `data/dataset_facts.json` |
| 6 | Validation protocol: 32 tests of leakage, physics, statistics and CV structure (`dataset_validation_protocol.md`) | `scripts/validate_dataset_integrity.py` | `data/validation/dataset_validation_report.{json,md}` |
| 7 | Reproducibility check against the published build | `run_pipeline.py` | `logs/reproducibility_check.json` |

**Why the split comes before augmentation.** Augmenting first would place perturbed copies of the same laboratory spectrum in both training and validation (augmentation leakage), which inflates validation scores. The pipeline therefore does three things:
- splits on *spectra* (grouped by specimen and spectral similarity);
- applies augmentation and noise only to the training partition of each fold;
- generates validation and benchmark partitions from pristine spectra.

Tests A1, A3c and B4 verify this.

## 3. Perturbation model
All perturbations are declared once, in `NOISE_MODEL` / `ATM_ERROR_MODEL` / `SAMPLING_MODEL` (`synthesize_forward_dataset.py`), and are echoed into `data/dataset_facts.json`. The design rule is that **every perturbation must correspond to a documented physical error source of the measurement chain** (detector NEΔT, atmospheric correction, laboratory emissivity uncertainty), with a magnitude taken from that source. Labels (kinetic temperature) are never perturbed.

**v1.3.0 perturbations:**
- **Sensor:** NEΔT at specification (ASTER 0.3 K, HyTES 0.2 K) on the at-sensor radiance; × U(0.5, 1.5) in training.
- **Atmospheric correction (N4):** the network never receives the true atmosphere. τ, L↑ and L↓ come from the same MODTRAN model at PWV_est = PWV + δ, with δ ~ N(0, s²) absolute; s = 2.97 mm in evaluation and U(1.84, 4.1) mm in training (Wang et al. 2020; Seemann et al. 2003). The surface radiance input is the operational correction (L_obs − L↑_est)/τ_est, so sensor noise is propagated physically.
- **Not perturbed:** emissivity (N5/N6 removed).

Tests B2d, B7 and B8 verify N4. B8 uses 24 extra MODTRAN check profiles (`n4_interp_check/`). Details: `docs/perturbation_model_error_sources.md` §3b.

## 4. Inputs (`inputs/`)
| File | Content | Source / licence |
|---|---|---|
| `spectral_library_raw.zip` | Raw ECOSTRESS/ASTER-JHU spectra read by the curator (keyword-selected) | ECOSTRESS Spectral Library v1.0 (Meerdink et al., 2019) [citation required in derived work] |
| `srf/aster/*_ch11-15.txt` | ASTER TIR Bands 10–14 spectral response functions (RTTOV format) | NWP-SAF RTTOV coefficient files |
| `srf/hytes/hytes_real_parameters.csv` | HyTES 256-channel centres and Gaussian widths | Derived from JPL HyTES SRF tables |
| `atmospheric_luts/` | 30 MODTRAN 5.2 profiles + `lut_catalog.csv` (MD5-verified; `pwv_g_cm2` = actual column after MODTRAN's saturation cap, `pwv_requested_g_cm2`, `h2o_capped_at_saturation`) | Generated with `automate_modtran.py` |
| `n4_interp_check/` | 24 MODTRAN check profiles at H₂O × 0.625/0.875/1.25/1.75 + their decks (validator test B8) | `automate_modtran.py --scales 0.625 0.875 1.25 1.75` |
| `modtran_cards/` | **All 330 MODTRAN 5.2 input decks actually executed** (`tape5/*.tp5`), `deck_index.csv` (role, geometry, quadrature node, weight, MD5), `tape7_reference.zip` (the corresponding `tape7.scn` outputs), and `README_MODTRAN_CARDS.md` (card-by-card description) | Archived by `automate_modtran.py --archive-decks` |

### 4.1 Replicating the atmospheric LUTs (licensed MODTRAN 5.2 users)
```bash
set OPENTIR_MODTRAN_DIR=C:\PcModWin5\bin
python run_pipeline.py --stages setup atmosphere --run-modtran
```
This re-executes every published deck. It checks that the executed decks are byte-identical (MD5) to `inputs/modtran_cards`, rebuilds the 30 LUTs in `data/validation/lut_replication/`, and compares them with the shipped LUTs (|a − b| ≤ 10⁻⁶ + 10⁻⁴·|a| per spectral point). The shipped LUTs are never overwritten. The reference build reproduced all 210 checks with a maximum relative difference of 0.

## 5. Outputs
| Path | Content |
|---|---|
| `data/processed/final_datasets/train_fold{k}_{sensor}.pt` | Training partition of fold k |
| `.../val_fold{k}_{sensor}.pt` | Out-of-fold validation (Quadrant I) |
| `.../quadrant_III_fold{k}_{sensor}.pt` | Fold-k validation spectra × golden atmospheres |
| `.../golden_benchmark_quadrant_II_{sensor}.pt` / `..._IV_...` | Golden targets × development / golden atmospheres |
| `data/dataset_facts.json` | Every dataset number (counts, ranges, noise model, checksums) |
| `data/validation/dataset_validation_report.{json,md}` | Every test result with obtained value and limit |
| `logs/pipeline_*.log`, `logs/pipeline_summary.json` | Full execution log and stage timings |

