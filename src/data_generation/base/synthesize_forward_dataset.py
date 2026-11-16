"""
Synthesize Dual-Input Forward Radiative Datasets for OpenTIR-PINN v1.1.1
========================================================================
Combines:
  1. Sanitized Spectral Library (241 physical spectra after the 3rd Validation Gate:
     233 development + 7 golden targets + 1 quarantined sibling; 11 classes)
  2. MODTRAN 5.2 Multi-Condition Atmospheric Look-Up Tables (30 profiles)
  3. 5-Fold Grouped Stratified Split Manifest (zero sample_id leakage)
  4. Real Sensor Spectral Response Functions (HyTES 256 bands, ASTER 5 bands)
  5. Realistic Radiometric Noise Modeling (NEDT = 0.1 K at 300 K)

Generates:
  - 5-Fold Training Sets: `train_fold{k}_{sensor}.pt` (uniform: samples_per_spectrum per training spectrum)
  - 5-Fold Validation Sets: `val_fold{k}_{sensor}.pt` (deterministic grid: 11 T x 4 atm)
  - Golden Target Benchmark Vaults: `golden_benchmark_{quadrant}_{sensor}.pt`
  - Quadrant III (Dev OOF x Golden Atmospheres): `quadrant_III_fold{k}_{sensor}.pt`
  - Default symlinks/files: `train_dataset_{sensor}.pt` and `val_dataset_{sensor}.pt`

Author: Dr. Atilio Grondona & OpenTIR-PINN Research Team
Target Journal: IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)
"""

import os
import sys
import glob
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from scipy.constants import h, c, k
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional

# Root setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)

from src.config import SENSOR_CONFIG

# Common high-resolution wavelength grid (7.50 to 13.50 um, 1201 channels)
WAVELENGTH_GRID = np.linspace(7.50, 13.50, 1201)
WAV_M = WAVELENGTH_GRID * 1e-6
C1 = 2.0 * h * c**2
C2 = h * c / k


# -----------------------------------------------------------------------------
# Perturbation & sampling model (single source of truth; read by build_dataset_facts.py
# and validate_dataset_integrity.py). Every entry must map to a documented physical error
# source (clarification.md Entries 37, 40, 43 and 47). Values are those of dataset v1.3.0 (N1 sensor-spec NEdT,
# N4 atmospheric-correction error, N5 jitter removed, N6 emissivity scale removed).
# -----------------------------------------------------------------------------
NOISE_MODEL = {
    "nedt_k": {"aster": 0.3, "hytes": 0.2},      # detector NEdT spec at 300 K: ASTER <= 0.3 K (Yamaguchi et al. 1998); HyTES ~0.2 K (Johnson et al. 2011)
    "nedt_reference_temperature_k": 300.0,
    "train_noise_multiplier": (0.5, 1.5),        # U(a, b) per training sample; evaluation uses 1.0
    "ldown_noise_factor": 0.0,                   # v1.3.0: white noise on L_down removed; the L_down given to the network now
                                                 # carries the physical atmospheric-correction error (ATM_ERROR_MODEL, N4)
    "eps_scale": (1.0, 1.0),                     # N6 (v1.2.2): removed - no verified absolute FTIR/DHR uncertainty
                                                 # supports a +/-0.5 % scale (Entry 43); (1, 1) = no perturbation
    "eps_jitter_sigma": 0.0,                     # N5 (v1.2.1): removed - white per-channel jitter has no instrumental
                                                 # basis and lies below NEdT after SRF convolution (Entry 37)
}
# N4 (v1.3.0): atmospheric-correction error. The network never receives the true atmosphere: tau, L_up and L_down
# are those of the same MODTRAN model at an ESTIMATED water-vapour column PWV_est = PWV_true + delta, delta ~ N(0, s^2)
# (absolute, mm). Documented RMSE of operational PWV sources: 1.84 mm (ERA5) to 3.51 mm (NCEP/DOE) vs global GNSS
# (Wang et al. 2020, Remote Sens. 12(11):1817) and 4.1 mm (MODIS operational TPW vs ARM microwave radiometer;
# Seemann et al. 2003, J. Appl. Meteorol. 42(8):1072-1091). Training: s ~ U(1.84, 4.1) mm per sample (span of the
# documented sources); evaluation: s = 2.97 mm (midpoint). The observation itself (L_sensor) is exact physics + NEdT;
# the surface radiance given to the network is the operational correction L_surf = (L_sensor - L_up_est) / tau_est,
# expressed relative to the exact band-level correction so that only the PWV error and the propagated sensor noise
# enter (band-model error excluded by construction). Air-temperature profile errors are not representable without
# new MODTRAN runs and are not modelled. PWV_est is bounded to [0, saturation cap of the model] (MODTRAN caps the
# column at 100 % RH; automate_modtran.py records the actual column).
ATM_ERROR_MODEL = {
    "enabled": True,
    "train_sigma_mm": (1.84, 4.1),               # U(a, b) per training sample
    "eval_sigma_mm": 2.97,                       # midpoint of the documented range; all evaluation partitions
    "interpolation": "quadratic",                # in actual PWV between H2O-scaled profiles of the same MODTRAN model
                                                 # (optical depth -ln tau, L_up, L_down); see validator test B8
    "node_merge_tol_g_cm2": 0.01,                # nodes closer than this (saturation-capped duplicates) are merged
}
_ATM_NODES: Dict[Any, Any] = {}

# N3 (v1.2.1): static spectral band mask. A band is flagged unusable when its SRF-convolved transmittance is
# below `tau_threshold` in ANY development atmosphere (test atmospheres are not used to define it). Radiance
# values are kept unchanged in the tensors (raw physics); every sample carries `band_mask` (True = usable) and
# training/evaluation code must exclude masked bands from the network inputs.
BAND_MASK_MODEL = {"tau_threshold": 0.1, "reference_atmospheres": "development"}
BAND_MASKS: Dict[str, torch.Tensor] = {}

SAMPLING_MODEL = {
    "train_temperature_k": (275.0, 335.0),       # U(a, b), continuous
    "eval_temperature_grid_k": (280.0, 330.0, 5.0),
    "samples_per_spectrum": 30,
}


def compute_band_mask(sensor: str, W_sensor: np.ndarray, luts: Dict[str, Any]) -> np.ndarray:
    """True = usable band: SRF-convolved transmittance >= tau_threshold in every development atmosphere."""
    key = "tau_aster" if sensor == "aster" else "tau_hytes"
    dev = [a for a, v in luts.items() if v["quarantine_status"] == "development"]
    tau = np.array([W_sensor @ luts[a][key] for a in dev])
    return (tau.min(axis=0) >= BAND_MASK_MODEL["tau_threshold"])


def ensure_band_mask(sensor: str, W_sensor: np.ndarray, luts: Dict[str, Any]) -> torch.Tensor:
    if sensor not in BAND_MASKS:
        BAND_MASKS[sensor] = torch.tensor(compute_band_mask(sensor, W_sensor, luts), dtype=torch.bool)
        n_bad = int((~BAND_MASKS[sensor]).sum())
        print(f"Band mask ({sensor}): {n_bad} of {len(BAND_MASKS[sensor])} bands flagged opaque "
              f"(tau < {BAND_MASK_MODEL['tau_threshold']} in some development atmosphere)")
    return BAND_MASKS[sensor]


def eval_temperature_grid() -> np.ndarray:
    a, b, step = SAMPLING_MODEL["eval_temperature_grid_k"]
    return np.arange(a, b + step / 2.0, step)


def planck_blackbody_grid(temp_k: float) -> np.ndarray:
    """
    Computes blackbody spectral radiance B(lambda, T) on WAVELENGTH_GRID
    in W / (m^2 * sr * um).
    """
    exp_term = np.exp(C2 / (WAV_M * temp_k))
    val = (C1 / (WAV_M**5)) / (exp_term - 1.0)
    return val * 1e-6


def planck_temp_derivative(wav_um: np.ndarray, temp_k: float = 300.0) -> np.ndarray:
    """
    Computes dB(lambda, T)/dT at temp_k in W / (m^2 * sr * um * K).
    Used for physical sensor noise scaling from NEDT.
    """
    w_m = wav_um * 1e-6
    exp_term = np.exp(C2 / (w_m * temp_k))
    dB_dT = (C1 / (w_m**5)) * (exp_term / (exp_term - 1.0)**2) * (C2 / (w_m * temp_k**2))
    return dB_dT * 1e-6


def build_sensor_srf_matrix(sensor: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds the area-normalized SRF convolution matrix W of shape (n_bands, 1201)
    such that L_band = W @ L_highres.

    Returns:
        W (np.ndarray): Shape (n_bands, 1201), sum(W, axis=1) == 1.
        centers (np.ndarray): Effective band center wavelengths in um.
    """
    cfg = SENSOR_CONFIG[sensor]
    n_bands = cfg['n_bands']

    if sensor == 'hytes':
        param_path = cfg['processed_params_file']
        if not os.path.exists(param_path):
            raise FileNotFoundError(f"HyTES parameter file missing: {param_path}")
        df = pd.read_csv(param_path).sort_values('band_idx')
        W = np.zeros((n_bands, len(WAVELENGTH_GRID)), dtype=np.float64)
        centers = df['center_wavelength'].values

        for idx, row in df.iterrows():
            mu = row['center_wavelength']
            sig = row['sigma']
            resp = np.exp(-0.5 * ((WAVELENGTH_GRID - mu) / sig)**2)
            s = np.sum(resp)
            if s > 0:
                resp /= s
            W[int(row['band_idx']) - 1, :] = resp

        return W, centers

    elif sensor == 'aster':
        srf_dir = cfg['srf_dir']
        band_indices = cfg['band_indices']  # [11, 12, 13, 14, 15] for TIR Bands 10-14
        W = np.zeros((n_bands, len(WAVELENGTH_GRID)), dtype=np.float64)
        centers = []

        for b_idx, ch_num in enumerate(band_indices):
            pattern = os.path.join(srf_dir, f"*_ch{ch_num:02d}.txt")
            files = glob.glob(pattern)
            if not files:
                raise FileNotFoundError(f"ASTER SRF file for channel {ch_num} missing in {srf_dir}")
            fpath = files[0]

            with open(fpath, 'r') as fp:
                lines = fp.readlines()[4:]
            raw_pts = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        raw_pts.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
            arr = np.array(raw_pts)
            w_um = 10000.0 / arr[:, 0]  # Wavenumber cm^-1 to um
            resp = arr[:, 1]
            sidx = np.argsort(w_um)
            w_um, resp = w_um[sidx], resp[sidx]

            f_interp = interp1d(w_um, resp, bounds_error=False, fill_value=0.0)
            r_grid = f_interp(WAVELENGTH_GRID)
            s = np.sum(r_grid)
            if s > 0:
                r_grid /= s
            W[b_idx, :] = r_grid
            eff_center = np.sum(WAVELENGTH_GRID * r_grid)
            centers.append(eff_center)

        return W, np.array(centers)

    else:
        raise ValueError(f"Unknown sensor: {sensor}")


def load_atmospheric_luts() -> Dict[str, Dict[str, Any]]:
    """
    Loads all 30 MODTRAN 5.2 profiles from data/processed/atmospheric_luts/
    and interpolates radiative profiles onto WAVELENGTH_GRID.
    """
    lut_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'atmospheric_luts')
    catalog_path = os.path.join(lut_dir, 'lut_catalog.csv')
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Atmospheric LUT catalog not found: {catalog_path}")

    df_catalog = pd.read_csv(catalog_path)
    luts = {}

    for _, row in df_catalog.iterrows():
        npz_name = f"profile_{row['profile_id']}.npz"
        npz_path = os.path.join(lut_dir, npz_name)
        if not os.path.exists(npz_path):
            continue

        raw = np.load(npz_path)
        raw_wav = raw['wavelength_um']

        # Interpolation helper
        def interp_to_grid(y_arr):
            f = interp1d(raw_wav, y_arr, kind='linear', bounds_error=False, fill_value='extrapolate')
            return f(WAVELENGTH_GRID)

        luts[row['profile_id']] = {
            'profile_id': row['profile_id'],
            'model_idx': int(row['atm_model_idx']),
            'model_name': row['atm_model_name'],
            'h2o_scale': float(row['h2o_scale']),
            'pwv_g_cm2': float(row['pwv_g_cm2']),
            'pwv_requested_g_cm2': float(row.get('pwv_requested_g_cm2', row['pwv_g_cm2'])),
            'h2o_capped': bool(str(row.get('h2o_capped_at_saturation', False)) == 'True'),
            'quarantine_status': row['quarantine_status'],
            'tau_aster': interp_to_grid(raw['tau_aster']),
            'L_up_aster': interp_to_grid(raw['L_up_aster']),
            'tau_hytes': interp_to_grid(raw['tau_hytes']),
            'L_up_hytes': interp_to_grid(raw['L_up_hytes']),
            'L_down_surface': interp_to_grid(raw['L_down_surface']),
        }

    return luts


def load_rebalanced_library() -> Dict[str, Dict[str, Any]]:
    """
    Loads the 241 sanitized physical spectra (v1.1.1, after the 3rd Validation Gate)
    from rebalanced_spectral_library.npz and library_metadata_catalog.csv.
    Rows are matched by position, so the npz `spectrum_ids` order is asserted
    against the catalog to rule out silent emissivity/label misalignment.
    """
    lib_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'spectral_libraries')
    npz_path = os.path.join(lib_dir, 'rebalanced_spectral_library.npz')
    csv_path = os.path.join(lib_dir, 'library_metadata_catalog.csv')

    df_meta = pd.read_csv(csv_path)
    raw = np.load(npz_path, allow_pickle=True)
    emis_mat = raw['emissivity']  # (241, 1201)
    if 'spectrum_ids' in raw.files:
        npz_ids = raw['spectrum_ids'].astype(str)
        if len(npz_ids) != len(df_meta) or not np.array_equal(npz_ids, df_meta['spectrum_id'].astype(str).values):
            raise ValueError("Library npz rows are not aligned with library_metadata_catalog.csv")

    library = {}
    for idx, row in df_meta.iterrows():
        library[row['spectrum_id']] = {
            'spectrum_id': row['spectrum_id'],
            'sample_id': row['sample_id'],
            'material_class': row['material_class'],
            'superclass': row['superclass'],
            'is_golden_target': bool(row['is_golden_target']),
            'is_sibling_quarantined': bool(row['is_sibling_quarantined']),
            'emissivity': emis_mat[idx, :].astype(np.float64),
        }

    return library


_ATM_KEYS = ("tau_aster", "L_up_aster", "tau_hytes", "L_up_hytes", "L_down_surface")


def _atm_nodes(luts: Dict[str, Any], model_idx: int):
    """Distinct (actual-PWV) nodes of one MODTRAN model: PWV vector, transformed variables, saturation flag."""
    key = (id(luts), model_idx)
    if key not in _ATM_NODES:
        rows = sorted((v for v in luts.values() if v["model_idx"] == model_idx), key=lambda v: v["pwv_g_cm2"])
        kept = []
        for r in rows:
            if kept and r["pwv_g_cm2"] - kept[-1]["pwv_g_cm2"] < ATM_ERROR_MODEL["node_merge_tol_g_cm2"]:
                continue
            kept.append(r)
        P = np.array([r["pwv_g_cm2"] for r in kept])
        V = {k: np.stack([(-np.log(np.clip(r[k], 1e-12, 1.0)) if k.startswith("tau") else r[k]) for r in kept])
             for k in _ATM_KEYS}
        capped = np.array([bool(r.get("h2o_capped", False)) for r in kept])
        _ATM_NODES[key] = (P, V, capped, bool(rows[-1].get("h2o_capped", False)))
    return _ATM_NODES[key]


def estimated_atmosphere(luts: Dict[str, Any], model_idx: int, pwv_est: float) -> Dict[str, Any]:
    """
    tau, L_up and L_down of MODTRAN model `model_idx` at water-vapour column `pwv_est` (g/cm^2), interpolated in actual
    PWV between the model's H2O-scaled profiles (quadratic Lagrange on the 3 nearest non-capped nodes inside the range
    of non-capped nodes; linear between the last non-capped node and the saturation cap and for extrapolation below). tau is interpolated as optical depth. PWV_est is bounded to [0, cap] when the
    model's wettest profile is saturation-capped, else to [0, inf).
    """
    P, V, node_capped, capped = _atm_nodes(luts, model_idx)
    lo, hi = 0.0, (P[-1] if capped else np.inf)
    x = float(np.clip(pwv_est, lo, hi))
    unc = np.where(~node_capped)[0]  # saturation-capped profiles have a redistributed H2O profile: excluded from the
    #                                  quadratic stencil (they are not members of the scaled-profile family)
    if (ATM_ERROR_MODEL["interpolation"] == "linear" or len(unc) < 3 or x <= P[0] or x >= P[unc].max()):
        i = int(np.clip(np.searchsorted(P, x) - 1, 0, len(P) - 2))
        w = (x - P[i]) / (P[i + 1] - P[i])
        coef = {i: 1.0 - w, i + 1: w}
    else:
        idx = sorted(unc[np.argsort(np.abs(P[unc] - x))[:3]].tolist())
        coef = {}
        for a in idx:
            c = 1.0
            for b in idx:
                if b != a:
                    c *= (x - P[b]) / (P[a] - P[b])
            coef[a] = c
    out = {"pwv_used": x, "pwv_clipped": bool(x != pwv_est)}
    for k in _ATM_KEYS:
        y = sum(c * V[k][a] for a, c in coef.items())
        out[k] = np.exp(-np.clip(y, 0.0, None)) if k.startswith("tau") else np.clip(y, 0.0, None)
    return out


def draw_pwv_error(rng: np.random.Generator, is_train: bool) -> Tuple[float, float]:
    """Returns (sigma, delta) in g/cm^2 (1 mm = 0.1 g/cm^2)."""
    sigma_mm = float(rng.uniform(*ATM_ERROR_MODEL["train_sigma_mm"])) if is_train else ATM_ERROR_MODEL["eval_sigma_mm"]
    return 0.1 * sigma_mm, float(rng.normal(0.0, 0.1 * sigma_mm))


def augment_emissivity(eps: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Physical shape-preserving augmentation:
    eps_aug = eps * U(eps_scale) + N(0, eps_jitter_sigma^2), clipped to [0, 1] (NOISE_MODEL).
    """
    if tuple(NOISE_MODEL["eps_scale"]) == (1.0, 1.0) and NOISE_MODEL["eps_jitter_sigma"] == 0.0:
        return eps.copy()  # v1.2.2: no emissivity perturbation (N5 + N6); no RNG draws consumed
    scale = rng.uniform(*NOISE_MODEL["eps_scale"])
    noise = rng.normal(0.0, NOISE_MODEL["eps_jitter_sigma"], size=eps.shape)
    eps_aug = eps * scale + noise
    return np.clip(eps_aug, 0.0, 1.0)


def synthesize_single_observation(
    emissivity_highres: np.ndarray,
    temp_k: float,
    atm_lut: Dict[str, Any],
    sensor: str,
    W_sensor: np.ndarray,
    sensor_centers: np.ndarray,
    nedt_k: float,
    noise_multiplier: float,
    rng: np.random.Generator,
    apply_noise: bool = True,
    atm_est: Optional[Dict[str, Any]] = None
) -> Dict[str, torch.Tensor]:
    """
    Computes the exact forward radiative transfer and the band quantities seen by a retrieval.
    The at-sensor radiance carries NEdT noise. If `atm_est` (N4) is given, the network inputs are those of an
    operational atmospheric correction made with the estimated atmosphere:
      L_surf_in = W L_surf + (L_obs - W L_up_est) / (W tau_est) - (W L_sensor - W L_up) / (W tau)
      L_down_in = W L_down_est ;  tau_in = W tau_est ;  L_up_in = W L_up_est
    i.e. the exact band-level correction plus the PWV error and the propagated sensor noise.
    """
    # 1. High-resolution blackbody
    b_highres = planck_blackbody_grid(temp_k)
    l_down_highres = atm_lut['L_down_surface']

    # 2. Surface-leaving radiance: L_surf = eps * B(T) + (1 - eps) * L_down
    l_surf_highres = emissivity_highres * b_highres + (1.0 - emissivity_highres) * l_down_highres

    # 3. Path transmission and upwelling radiance
    if sensor == 'aster':
        tau_highres = atm_lut['tau_aster']
        l_up_highres = atm_lut['L_up_aster']
    else:
        tau_highres = atm_lut['tau_hytes']
        l_up_highres = atm_lut['L_up_hytes']

    # 4. At-sensor radiance: L_sensor = tau * L_surf + L_up
    l_sensor_highres = tau_highres * l_surf_highres + l_up_highres

    # 5. Convolution via area-normalized SRF matrix W
    l_surf_bands = W_sensor @ l_surf_highres
    l_down_bands = W_sensor @ l_down_highres
    l_sensor_bands = W_sensor @ l_sensor_highres
    eps_bands = W_sensor @ emissivity_highres

    tau_in = W_sensor @ tau_highres
    l_up_in = W_sensor @ l_up_highres
    if atm_est is not None:
        # 6'. N4: at-sensor observation (exact physics + NEdT), corrected with the estimated atmosphere
        if apply_noise and nedt_k > 0.0:
            db_dt = planck_temp_derivative(sensor_centers, temp_k=NOISE_MODEL["nedt_reference_temperature_k"])
            l_obs = l_sensor_bands + rng.normal(0.0, db_dt * nedt_k * noise_multiplier, size=l_sensor_bands.shape)
        else:
            l_obs = l_sensor_bands.copy()
        k_tau, k_up = ("tau_aster", "L_up_aster") if sensor == 'aster' else ("tau_hytes", "L_up_hytes")
        tau_est_b = W_sensor @ atm_est[k_tau]
        l_up_est_b = W_sensor @ atm_est[k_up]
        exact_corr = (l_sensor_bands - l_up_in) / tau_in
        l_surf_bands = l_surf_bands + (l_obs - l_up_est_b) / tau_est_b - exact_corr
        l_sensor_bands = l_obs
        l_down_bands = W_sensor @ atm_est["L_down_surface"]
        tau_in, l_up_in = tau_est_b, l_up_est_b
        apply_noise = False  # noise already propagated

    # 6. Radiometric noise injection based on dB/dT * NEDT (legacy path, atm_est is None)
    if apply_noise and nedt_k > 0.0:
        db_dt = planck_temp_derivative(sensor_centers, temp_k=NOISE_MODEL["nedt_reference_temperature_k"])
        sigma_l = db_dt * nedt_k * noise_multiplier
        noise_surf = rng.normal(0.0, sigma_l, size=l_surf_bands.shape)
        noise_sensor = rng.normal(0.0, sigma_l, size=l_sensor_bands.shape)
        noise_down = (rng.normal(0.0, sigma_l * NOISE_MODEL["ldown_noise_factor"], size=l_down_bands.shape)
                      if NOISE_MODEL["ldown_noise_factor"] > 0 else 0.0)

        l_surf_bands = np.clip(l_surf_bands + noise_surf, 1e-6, None)
        l_sensor_bands = np.clip(l_sensor_bands + noise_sensor, 1e-6, None)
        l_down_bands = np.clip(l_down_bands + noise_down, 1e-6, None)

    return {
        'input_radiance': torch.tensor(l_surf_bands, dtype=torch.float32),
        'input_radiance_sensor': torch.tensor(l_sensor_bands, dtype=torch.float32),
        'downwelling_radiance': torch.tensor(l_down_bands, dtype=torch.float32),
        'tau_input': torch.tensor(tau_in, dtype=torch.float32),
        'l_up_input': torch.tensor(l_up_in, dtype=torch.float32),
        'target_temp': torch.tensor([temp_k], dtype=torch.float32),
        'target_emissivity': torch.tensor(eps_bands, dtype=torch.float32),
        'target_emissivity_highres': torch.tensor(emissivity_highres, dtype=torch.float32),
        'band_mask': BAND_MASKS[sensor],
    }


def make_observation(eps_highres: np.ndarray, temp_k: float, atm_id: str, luts: Dict[str, Any], sensor: str,
                     W_sensor: np.ndarray, sensor_centers: np.ndarray, noise_multiplier: float,
                     rng: np.random.Generator, is_train: bool) -> Dict[str, Any]:
    """One sample: true atmosphere for the physics, estimated atmosphere (N4) for the network inputs."""
    atm_lut = luts[atm_id]
    atm_est, meta = None, {}
    if ATM_ERROR_MODEL["enabled"]:
        sigma, delta = draw_pwv_error(rng, is_train)
        atm_est = estimated_atmosphere(luts, atm_lut["model_idx"], atm_lut["pwv_g_cm2"] + delta)
        meta = {"pwv_est": atm_est["pwv_used"], "pwv_error_drawn": delta, "pwv_sigma": sigma,
                "pwv_est_clipped": atm_est["pwv_clipped"]}
    tensors = synthesize_single_observation(
        emissivity_highres=eps_highres, temp_k=float(temp_k), atm_lut=atm_lut, sensor=sensor, W_sensor=W_sensor,
        sensor_centers=sensor_centers, nedt_k=NOISE_MODEL["nedt_k"][sensor], noise_multiplier=noise_multiplier,
        rng=rng, apply_noise=True, atm_est=atm_est)
    tensors.update(meta)
    tensors["pwv"] = atm_lut["pwv_g_cm2"]
    return tensors


def synthesize_all_datasets(sensor: str, samples_per_spectrum: int = SAMPLING_MODEL["samples_per_spectrum"], base_seed: int = 42,
                            folds: Optional[List[int]] = None, partitions: Optional[List[str]] = None,
                            output_dir: Optional[str] = None) -> None:
    """
    Main orchestration routine for forward dataset synthesis.
    """
    print(f"\n{'='*70}")
    print(f"Synthesizing Forward Datasets for Sensor: {sensor.upper()}")
    print(f"{'='*70}")

    output_dir = output_dir or os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load resources
    W_sensor, sensor_centers = build_sensor_srf_matrix(sensor)
    luts = load_atmospheric_luts()
    ensure_band_mask(sensor, W_sensor, luts)
    library = load_rebalanced_library()

    split_manifest_path = os.path.join(PROJECT_ROOT, 'splits', 'split_v1.1.json')
    with open(split_manifest_path, 'r') as fp:
        split_data = json.load(fp)

    golden_vault_path = os.path.join(PROJECT_ROOT, 'splits', 'golden_targets.json')
    with open(golden_vault_path, 'r') as fp:
        golden_vault = json.load(fp)

    # Segregate development atmospheres from golden quarantined atmospheres
    dev_atm_ids = [aid for aid, a in luts.items() if a['quarantine_status'] == 'development']
    golden_atm_ids = [aid for aid, a in luts.items() if 'golden' in a['quarantine_status']]
    print(f"Loaded {len(luts)} total MODTRAN LUTs ({len(dev_atm_ids)} Dev, {len(golden_atm_ids)} Golden Quarantined).")
    print(f"Loaded {len(library)} total spectra from Rebalanced Library.")

    # 4 Representative Development Atmospheres for deterministic validation/testing
    rep_dev_atm_ids = ['atm1_h2o1.00', 'atm2_h2o1.00', 'atm4_h2o1.00', 'atm6_h2o1.00']
    for ra in rep_dev_atm_ids:
        if ra not in luts:
            rep_dev_atm_ids = dev_atm_ids[:4]
            break

    # 2. Iterate across 5 Folds
    for fold in split_data['folds']:
        if folds is not None and fold['fold_idx'] not in folds:
            continue
        fold_idx = fold['fold_idx']
        train_spec_ids = fold['train_spectrum_ids']
        val_spec_ids = fold['val_spectrum_ids']

        rng_train = np.random.default_rng(base_seed + fold_idx * 100)
        rng_val = np.random.default_rng(base_seed + fold_idx * 100 + 50)

        print(f"\n--- Processing Fold {fold_idx} ({len(train_spec_ids)} train spectra, {len(val_spec_ids)} val spectra) ---")

        # Group train spectra by material_class
        class_to_train_specs = {}
        for sid in train_spec_ids:
            cls = library[sid]['material_class']
            class_to_train_specs.setdefault(cls, []).append(sid)

        # 2A. Generate Training Set with uniform per-SPECTRUM sampling (v1.1.2).
        # The v1.1.1 class-balanced sampler (500 draws per class) repeated singleton-class spectra
        # 500x (~9% of the training mass on one spectrum) and created a train/val class-prior shift
        # (adversarial AUC up to 0.86). Every training spectrum now contributes exactly
        # `samples_per_spectrum` augmented observations; class imbalance, if it must be corrected,
        # is a training-time loss-weighting decision, not a data-generation one.
        train_samples = []
        for sid in sorted(train_spec_ids):
            cls = library[sid]['material_class']
            for _ in range(samples_per_spectrum):
                spec_meta = library[sid]
                eps_aug = augment_emissivity(spec_meta['emissivity'], rng_train)

                temp_k = float(rng_train.uniform(*SAMPLING_MODEL["train_temperature_k"]))
                atm_id = rng_train.choice(dev_atm_ids)
                atm_lut = luts[atm_id]
                noise_mult = float(rng_train.uniform(*NOISE_MODEL["train_noise_multiplier"]))

                tensors = make_observation(eps_aug, temp_k, atm_id, luts, sensor, W_sensor, sensor_centers,
                    noise_multiplier=noise_mult, rng=rng_train, is_train=True)
                tensors['spectrum_id'] = sid
                tensors['sample_id'] = spec_meta['sample_id']
                tensors['material_class'] = cls
                tensors['superclass'] = spec_meta['superclass']
                tensors['atmosphere_id'] = atm_id
                tensors['pwv'] = atm_lut['pwv_g_cm2']
                tensors['is_augmented'] = True
                train_samples.append(tensors)

        rng_train.shuffle(train_samples)
        train_path = os.path.join(output_dir, f"train_fold{fold_idx}_{sensor}.pt")
        torch.save(train_samples, train_path)
        print(f"Saved Fold {fold_idx} Train: {train_path} ({len(train_samples)} samples across {len(class_to_train_specs)} classes)")

        # Save Fold 0 as standard default for legacy compatibility
        if fold_idx == 0:
            default_train_path = os.path.join(output_dir, f"train_dataset_{sensor}.pt")
            torch.save(train_samples, default_train_path)
            print(f"Saved Default Train (Fold 0): {default_train_path}")

        # 2B. Generate Deterministic Validation Set (11 Temperatures x 4 Atmospheres)
        val_samples = []
        val_temps = eval_temperature_grid()  # 11 temperatures, 280-330 K

        for sid in val_spec_ids:
            spec_meta = library[sid]
            eps_pure = spec_meta['emissivity']

            for temp_k in val_temps:
                for atm_id in rep_dev_atm_ids:
                    atm_lut = luts[atm_id]
                    tensors = make_observation(eps_pure, float(temp_k), atm_id, luts, sensor, W_sensor, sensor_centers,
                        noise_multiplier=1.0, rng=rng_val, is_train=False)
                    tensors['spectrum_id'] = sid
                    tensors['sample_id'] = spec_meta['sample_id']
                    tensors['material_class'] = spec_meta['material_class']
                    tensors['superclass'] = spec_meta['superclass']
                    tensors['atmosphere_id'] = atm_id
                    tensors['pwv'] = atm_lut['pwv_g_cm2']
                    tensors['is_augmented'] = False
                    val_samples.append(tensors)

        val_path = os.path.join(output_dir, f"val_fold{fold_idx}_{sensor}.pt")
        torch.save(val_samples, val_path)
        print(f"Saved Fold {fold_idx} Val:   {val_path} ({len(val_samples)} samples, 100% deterministic grid)")

        if fold_idx == 0:
            default_val_path = os.path.join(output_dir, f"val_dataset_{sensor}.pt")
            torch.save(val_samples, default_val_path)
            print(f"Saved Default Val (Fold 0):   {default_val_path}")

    parts = set(partitions) if partitions is not None else {"train", "val", "q2", "q3", "q4"}
    if not ({"q2", "q3", "q4"} & parts):
        print("Benchmark partitions (Q-II, Q-III, Q-IV) not requested: golden vault left untouched.")
        return

    # 3. Generate Golden Benchmark Vaults (The 2x2 Clean-Room Matrix)
    print(f"\n--- Generating Golden Target Clean-Room Benchmark Vaults ---")
    rng_golden = np.random.default_rng(base_seed + 999)
    golden_targets = golden_vault['golden_targets']
    val_temps = eval_temperature_grid()  # 11 temperatures, 280-330 K

    # Quadrant II: Golden Targets under Development Atmospheres
    quadrant_II_samples = []
    for g_target in golden_targets:
        sid = g_target['spectrum_id']
        spec_meta = library[sid]
        eps_pure = spec_meta['emissivity']

        for temp_k in val_temps:
            for atm_id in rep_dev_atm_ids:
                atm_lut = luts[atm_id]
                tensors = make_observation(eps_pure, float(temp_k), atm_id, luts, sensor, W_sensor, sensor_centers,
                    noise_multiplier=1.0, rng=rng_golden, is_train=False)
                tensors['spectrum_id'] = sid
                tensors['sample_id'] = spec_meta['sample_id']
                tensors['material_class'] = spec_meta['material_class']
                tensors['superclass'] = spec_meta['superclass']
                tensors['golden_name'] = g_target['golden_target_name']
                tensors['atmosphere_id'] = atm_id
                tensors['pwv'] = atm_lut['pwv_g_cm2']
                tensors['is_augmented'] = False
                quadrant_II_samples.append(tensors)

    quad_II_path = os.path.join(output_dir, f"golden_benchmark_quadrant_II_{sensor}.pt")
    torch.save(quadrant_II_samples, quad_II_path)
    print(f"Saved Quadrant II Benchmark (Golden Materials x Dev Atm): {quad_II_path} ({len(quadrant_II_samples)} samples)")

    # Quadrant IV: Golden Targets under Golden Quarantined Atmospheres (Full Double-Blind OOD)
    quadrant_IV_samples = []
    for g_target in golden_targets:
        sid = g_target['spectrum_id']
        spec_meta = library[sid]
        eps_pure = spec_meta['emissivity']

        for temp_k in val_temps:
            for atm_id in golden_atm_ids:
                atm_lut = luts[atm_id]
                tensors = make_observation(eps_pure, float(temp_k), atm_id, luts, sensor, W_sensor, sensor_centers,
                    noise_multiplier=1.0, rng=rng_golden, is_train=False)
                tensors['spectrum_id'] = sid
                tensors['sample_id'] = spec_meta['sample_id']
                tensors['material_class'] = spec_meta['material_class']
                tensors['superclass'] = spec_meta['superclass']
                tensors['golden_name'] = g_target['golden_target_name']
                tensors['atmosphere_id'] = atm_id
                tensors['pwv'] = atm_lut['pwv_g_cm2']
                tensors['is_augmented'] = False
                quadrant_IV_samples.append(tensors)

    quad_IV_path = os.path.join(output_dir, f"golden_benchmark_quadrant_IV_{sensor}.pt")
    torch.save(quadrant_IV_samples, quad_IV_path)
    print(f"Saved Quadrant IV Benchmark (Golden Materials x Golden Atm OOD): {quad_IV_path} ({len(quadrant_IV_samples)} samples)")

    # Quadrant III: Development materials (out-of-fold) under Golden Quarantined Atmospheres
    if "q3" in parts:
        synthesize_quadrant_III(sensor=sensor, base_seed=base_seed, folds=folds, output_dir=output_dir)

    print(f"\n==================================================================")
    print(f"Step 2.4 Forward Synthesis Complete for {sensor.upper()}!")
    print(f"==================================================================")


def synthesize_quadrant_III(sensor: str, base_seed: int = 42, folds: Optional[List[int]] = None,
                            output_dir: Optional[str] = None) -> None:
    """
    Quadrant III (Pure Atmospheric OOD): out-of-fold development spectra coupled
    exclusively with the 10 golden-quarantined atmospheres (MODTRAN Models 3 & 5).

    Generated per fold so that the model trained on fold k is evaluated only on
    the validation spectra of fold k (materials unseen at the specimen level,
    atmospheres never seen at all). The union over the 5 folds covers each of the
    233 development spectra exactly once.

    Grid per fold: N_val_spectra x 10 golden atmospheres x 11 temperatures (280-330 K, 5 K),
    pure laboratory emissivity (no augmentation), nominal sensor NEdT (multiplier 1.0) and nominal
    atmospheric-correction error (ATM_ERROR_MODEL eval_sigma_mm).
    Uses a dedicated RNG stream (base_seed + 777 + fold) so existing partitions are untouched.
    """
    print(f"\n--- Generating Quadrant III (Dev OOF Materials x Golden Atmospheres) for {sensor.upper()} ---")
    output_dir = output_dir or os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets')
    os.makedirs(output_dir, exist_ok=True)

    W_sensor, sensor_centers = build_sensor_srf_matrix(sensor)
    luts = load_atmospheric_luts()
    ensure_band_mask(sensor, W_sensor, luts)
    library = load_rebalanced_library()
    with open(os.path.join(PROJECT_ROOT, 'splits', 'split_v1.1.json'), 'r') as fp:
        split_data = json.load(fp)

    golden_atm_ids = sorted(aid for aid, a in luts.items() if 'golden' in a['quarantine_status'])
    val_temps = eval_temperature_grid()  # 11 temperatures, 280-330 K

    for fold in split_data['folds']:
        if folds is not None and fold['fold_idx'] not in folds:
            continue
        fold_idx = fold['fold_idx']
        rng_q3 = np.random.default_rng(base_seed + 777 + fold_idx)
        q3_samples = []
        for sid in fold['val_spectrum_ids']:
            spec_meta = library[sid]
            for temp_k in val_temps:
                for atm_id in golden_atm_ids:
                    atm_lut = luts[atm_id]
                    tensors = make_observation(spec_meta['emissivity'], float(temp_k), atm_id, luts, sensor, W_sensor, sensor_centers,
                        noise_multiplier=1.0, rng=rng_q3, is_train=False)
                    tensors['spectrum_id'] = sid
                    tensors['sample_id'] = spec_meta['sample_id']
                    tensors['material_class'] = spec_meta['material_class']
                    tensors['superclass'] = spec_meta['superclass']
                    tensors['atmosphere_id'] = atm_id
                    tensors['pwv'] = atm_lut['pwv_g_cm2']
                    tensors['is_augmented'] = False
                    q3_samples.append(tensors)

        q3_path = os.path.join(output_dir, f"quadrant_III_fold{fold_idx}_{sensor}.pt")
        torch.save(q3_samples, q3_path)
        print(f"Saved Quadrant III Fold {fold_idx}: {q3_path} ({len(q3_samples)} samples = "
              f"{len(fold['val_spectrum_ids'])} spectra x {len(golden_atm_ids)} atm x {len(val_temps)} T)")
        if fold_idx == 0:
            torch.save(q3_samples, os.path.join(output_dir, f"golden_benchmark_quadrant_III_{sensor}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forward Radiative Dataset Synthesizer for OpenTIR-PINN v1.3.0")
    parser.add_argument("--sensor", type=str, default="both", choices=["hytes", "aster", "both"], help="Target sensor")
    parser.add_argument("--samples_per_spectrum", type=int, default=SAMPLING_MODEL["samples_per_spectrum"], help="Training samples generated per training spectrum")
    parser.add_argument("--seed", type=int, default=42, help="Master pseudorandom seed")
    parser.add_argument("--only_q3", action="store_true", help="Generate only Quadrant III without touching other partitions")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="Subset of folds (default: all)")
    parser.add_argument("--partitions", nargs="+", default=None, choices=["train", "val", "q2", "q3", "q4"],
                        help="Subset of partitions (default: all). Train/val are always generated together.")
    parser.add_argument("--out_dir", default=None, help="Output folder (default: data/processed/final_datasets)")
    args = parser.parse_args()

    sensors = ["hytes", "aster"] if args.sensor == "both" else [args.sensor]
    for s in sensors:
        if args.only_q3:
            synthesize_quadrant_III(sensor=s, base_seed=args.seed, folds=args.folds, output_dir=args.out_dir)
        else:
            synthesize_all_datasets(sensor=s, samples_per_spectrum=args.samples_per_spectrum, base_seed=args.seed,
                                    folds=args.folds, partitions=args.partitions, output_dir=args.out_dir)
