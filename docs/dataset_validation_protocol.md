# Dataset Validation Protocol: OpenTIR-PINN

Auto-generated from `TEST_CATALOG` in `scripts/validate_dataset_integrity.py` (single source of the test definitions). Results: `data/validation/dataset_validation_report.md` and the dashboard tab *Dataset Validation Report*.

**Global constants:** NEdT (300 K) = {'aster': 0.3, 'hytes': 0.2} K; spectral-twin radius RMSE = 0.005 (about 0.3 K of brightness temperature); expected training noise std = 1.041 (sqrt E[m^2], m ~ U(0.5, 1.5)); atmospheric-correction PWV error (N4): s = 2.97 mm (evaluation), s ~ U(1.84, 4.1) mm (training).

**Status logic:** a test is FAIL if a FAIL-level limit is violated, WARN if only WARN-level limits are violated, INFO if it is descriptive, and PASS otherwise. FAIL blocks the dataset release; each WARN requires a documented PI decision.

## A. Partition integrity & leakage

| ID | Test | Purpose | Method | Pass rule | Rationale |
|---|---|---|---|---|---|
| A0 | Library rows aligned with metadata catalog | Emissivity arrays and labels must refer to the same spectrum. | Assert npz `spectrum_ids` order equals catalog order (load_rebalanced_library). | `aligned == 1` | A silent row shift would attach wrong materials to every sample. |
| A1 | Train/val disjoint (spectrum & physical sample) | No spectrum or physical specimen is shared between training and validation. | Set intersection of spectrum_id and sample_id read from the tensors of each fold. | `overlap_spectra == 0 and overlap_samples == 0` | Classical identity leakage. |
| A1b | Tensors consistent with split manifest | Generated partitions use exactly the spectra declared in splits/split_v1.1.json. | Compare spectrum sets of train/val tensors with the manifest. | `missing_train == 0 and val_mismatch == 0` | Reproducibility of the split. |
| A2 | Golden-target & sibling quarantine | No golden target or quarantined sibling enters any development partition. | Check spectrum_id, sample_id and emissivity MD5 of train/val/Q-III against the golden vault. | `leaking_partitions == 0` | Zero-shot claims on Q-II/Q-IV require strict quarantine. |
| A3a | Library near-duplicates across specimens (descriptive) | Quantify spectral autocorrelation in the library. | Count spectrum pairs with emissivity RMSE < 0.005 and different sample_id. | `informative (handled by similarity-aware grouping, see A3c)` | Motivates grouping by spectral similarity, not only by specimen ID. |
| A3b | Golden targets have no development twin | Golden targets must not be radiometrically indistinguishable from a training spectrum. | Minimum emissivity RMSE from each golden target to every development spectrum. | `min_rmse >= 0.005` | RMSE 0.005 in emissivity ~ 0.3 K brightness temperature (1-1.5x the sensor NEdT). |
| A3c | Validation spectra have no training twin | Out-of-fold validation must not contain near-copies of training spectra. | For each fold, count val spectra whose nearest training spectrum has RMSE < 0.005. | `val_spectra_with_twin == 0` | Spectral-autocorrelation leakage (near-clones). |
| A4 | Atmosphere isolation | Golden (quarantined) atmospheres never appear in train/val; Q-III uses all of them. | Atmosphere-ID sets per partition vs lut_catalog quarantine status. | `golden_atm_in_train_val == 0 and q3_atmospheres == 10` | Atmospheric OOD claims (Q-III/Q-IV) require isolation. |
| A5 | Q-III spectra equal fold validation spectra | Q-III(fold k) must test only materials unseen by the fold-k model. | Set difference between Q-III and val spectrum sets. | `mismatch == 0` | Keeps Q-III a pure atmospheric-OOD test on unseen specimens. |
| A5b | Q-III union covers every development spectrum once | Across folds, Q-III evaluates the whole development pool exactly once. | Multiplicity of each development spectrum over the 5 Q-III partitions. | `uncovered == 0 and max_multiplicity == 1` | Complete, non-redundant coverage. |
| A6 | Golden vault composition (complete factorial grid) | Q-II/Q-IV contain exactly 7 targets x N atmospheres x 11 temperatures, no duplicates. | Count unique (target, atmosphere, temperature) cells. | `targets == 7, atmospheres == N, duplicate_cells == 0` | Balanced benchmark design. |

## B. Physical & numerical integrity

| ID | Test | Purpose | Method | Pass rule | Rationale |
|---|---|---|---|---|---|
| B1 | Finite values & physical bounds | No NaN/Inf; emissivity in [0, 1]; no radiance clipping artefacts. | Element-wise checks on all tensors. | `non_finite == 0, eps_out_of_bounds == 0, clip_events == 0` | Numerical integrity of training targets and inputs. |
| B2 | Forward-model reconstruction & NEdT noise law (at-sensor radiance) | The stored observation equals the noise-free RTE at the TRUE atmosphere plus the declared sensor noise. | Recompute W[tau (eps B + (1-eps) L_down) + L_up] with the true LUT; z = (L_sensor - clean)/sigma_L. | `|mean z| <= 0.05 and |std z - expected| <= 0.05 (expected 1.000 eval, 1.041 train)` | Proves the physics and the instrument-noise model are exactly what the paper states. |
| B2b | Normality of injected noise (evaluation partitions) | Noise residuals follow N(0, 1). | Kolmogorov-Smirnov test on 20,000 at-sensor z values. | `KS p-value >= 0.01` | Gaussian NEdT model. |
| B2c | Per-sample noise multiplier law (HyTES training) | Training noise multiplier m ~ U(0.5, 1.5). | Per-sample std of z over 256 bands, KS against U(0.5, 1.5). | `KS D <= 0.05` | Verifies the robustness-to-unknown-NEdT design. |
| B3 | Label consistency (band emissivity = W eps_highres) | Band-level targets are the SRF convolution of the high-resolution targets. | max |W eps_hr - eps_band|. | `max_abs_diff <= 1e-5` | Consistent supervision at both resolutions. |
| B4 | Augmentation confinement & law | Training emissivity follows the NOISE_MODEL law (v1.2.2: no perturbation, N5 + N6); evaluation emissivity is pristine. | Least-squares scale and residual std vs library spectrum (train); max |eps - library| (train when the law is the identity, and eval). | `train: scale within eps_scale +/- 0.0005, |resid_std - eps_jitter_sigma| <= max(0.15 sigma, 2e-4), and max_dev <= 1e-6 if no perturbation is configured; eval: max_dev <= 1e-6` | Benchmarks evaluate laboratory spectra, not synthetic perturbations. |
| B6 | Opaque-band mask present and correct | Bands opaque in some development atmosphere are flagged so they cannot be used as inputs. | Recompute the mask (SRF-convolved tau < threshold in any development atmosphere) and compare with the band_mask stored in every sample. | `samples_without_mask == 0 and samples_with_wrong_mask == 0` | Surface radiance cannot be observed through an opaque atmosphere (oracle-input guard, N3). |
| B2d | Network inputs = operational correction with the estimated atmosphere (N4) | tau_input, L_up_input, L_down and L_surf are exactly those of a correction made at the recorded PWV_est. | Recompute the estimated atmosphere at PWV_est (estimated_atmosphere) and L_surf = W L_surf_true + (L_obs - L_up_in)/tau_in - (W L_sensor_true - W L_up)/(W tau) on usable bands. | `max relative difference <= 1e-4 for tau_input, L_up_input, L_down and L_surf` | Proves that the only atmospheric information given to the network is the perturbed one. |
| B7 | Atmospheric-correction (PWV) error law (N4) | PWV_est - PWV ~ N(0, s^2), s = 2.97 mm (evaluation) or s ~ U(1.84, 4.1) mm (training). | KS test of z = drawn error / s against N(0, 1); KS of s against U(a, b) (training); PWV_est = clip(PWV + drawn error, 0, saturation cap); fraction clipped reported. | `KS p >= 0.01 (WARN); sigma consistency and PWV_est reconstruction exact (FAIL)` | Wang et al. 2020 (reanalysis vs GNSS, 1.84-3.51 mm); Seemann et al. 2003 (MODIS, 4.1 mm). |
| B8 | PWV-interpolation fidelity against MODTRAN (N4) | The estimated atmosphere obtained by interpolation in PWV reproduces MODTRAN at intermediate columns. | 24 MODTRAN check profiles at H2O scales 0.625/0.875/1.25/1.75 (data/validation/n4_interp_check, saturation-capped ones excluded); LST error of a correction made with the interpolated instead of the exact atmosphere (eps 0.95, 300 K, usable bands) vs the LST effect of an eval-sigma PWV error. | `max RMS ratio <= 0.3 (WARN)` | Worst-case (mid-interval) interpolation error must stay a small fraction of the modelled error. |
| B5 | Atmospheric LUT radiative definitions | L_down is the hemispheric sky radiance F_down/pi from a single, documented method. | Ratio hemispheric/zenith (8-12 um) for the 30 LUTs; method and RT config fields. | `1.0 < ratio_min and ratio_max < 2.0 and n_methods == 1` | Lambertian reflection requires F_down/pi; zenith radiance underestimates it 17-41 %. |

## C. Statistical design & bias

| ID | Test | Purpose | Method | Pass rule | Rationale |
|---|---|---|---|---|---|
| C1 | Uniform per-spectrum training sampling | Every training spectrum contributes the same number of samples. | Distinct values of samples-per-spectrum in each training fold. | `distinct_counts == 1` | Avoids repeating singleton-class spectra (memorization) and class-prior shift. |
| C2 | Nuisance factors independent of material class | No shortcut correlation between class and atmosphere/temperature. | Chi-square (class x atmosphere), Kruskal-Wallis (T across classes), Spearman(T, PWV). | `chi2 p >= 0.01, Kruskal p >= 0.01, |Spearman| <= 0.05` | Shortcut-learning guard. |
| C3 | Training temperature law | T_s ~ U(275, 335) K. | KS test against U(275, 335). | `KS p >= 0.01` | Prior support wider than the 280-330 K test grid avoids boundary shrinkage. |
| C4 | Uniform use of development atmospheres | The 20 development atmospheres are sampled uniformly. | Chi-square goodness of fit on atmosphere counts. | `chi2 p >= 0.01 and min_count > 0` | No atmosphere dominates training. |
| C5 | Spectrum repetition (memorization risk) | No single spectrum carries a large share of the training set. | Max share of samples belonging to one spectrum. | `max_share <= 1 %` | v1.1.1 had 9 % of training mass on one spectrum. |
| C7 | Adversarial AUC train vs val (descriptive) | Describe input-distribution shift between training and validation. | Logistic regression (PCA-20 for HyTES) on log L_surf, 5-fold ROC-AUC. | `informative (group-disjoint CV implies AUC > 0.5 by construction)` | Context for C7b. |
| C7b | Composition shift vs null of group-disjoint splits | Is a fold's train/val composition shift larger than any random group-stratified split? | Adversarial AUC on the val_oof union (identical generation process) for the actual fold vs a null distribution from random StratifiedGroupKFold seeds with the same groups. | `observed AUC <= 95th percentile of null` | Separates unavoidable group-CV shift from an unlucky/biased fold. |
| C8 | Golden-atmosphere PWV vs development envelope (descriptive) | Characterize the nature of the atmospheric OOD shift. | PWV ranges. | `informative` | Manuscript must describe OOD as thermal-profile shift where applicable. |
| C9 | Golden spectral contrast vs development pool (descriptive) | Position golden targets within the development MMD distribution. | MMD (max-min emissivity, 8-12 um) percentiles. | `informative` | Context for per-target benchmark interpretation. |

## D. Cross-validation structure

| ID | Test | Purpose | Method | Pass rule | Rationale |
|---|---|---|---|---|---|
| D1 | Training classes present in validation | Stratification places every class with enough independent groups in each validation fold. | Classes absent from val; flagged only if the class has >= 5 independent groups. | `flagged_classes == 0` | Classes with < 5 independent groups (e.g. singletons) cannot appear in all folds. |
