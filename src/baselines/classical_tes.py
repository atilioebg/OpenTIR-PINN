"""
Classical Temperature and Emissivity Separation (TES) Baseline Algorithm
========================================================================
Implements the seminal Gillespie et al. (1998) operational TES algorithm,
comprising:
  1. Normalized Emissivity Method (NEM) initial temperature retrieval
  2. Ratio Module (Beta-emissivity normalization)
  3. Min-Max Difference (MMD) spectral contrast calculation
  4. Empirical Power-Law regression (MMD -> eps_min)
  5. Scaled Emissivity and Final Kinetic Temperature Determination

Standard Operational References:
  - Gillespie et al. (1998): "A temperature and emissivity separation algorithm
    for Advanced Spaceborne Thermal Emission and Reflection Radiometer (ASTER) images."
    IEEE TGRS, 36(4), 1113-1126. doi:10.1109/36.700995
  - Hulley, G., Hook, S., Johnson, W., Guillevic, P., Malakar, N. (2016): "Hyperspectral
    Thermal Emission Spectrometer (HyTES) Level-2 Land Surface Temperature and Emissivity
    Algorithm Theoretical Basis Document." JPL Publication, Jet Propulsion Laboratory,
    Pasadena, CA (local copy: bibliografic_references/hytes/HyTES_L2_ATBD.pdf).

TES constants (v2, 08/11/2026 -- verified against the HyTES L2 ATBD):
  - NEM eps_max = 0.99 (ATBD p. 24 and Fig. 13). v1 used 0.983, which is the TEWG graybody
    eps_min assigned when MMD < 0.03 (ATBD p. 35), not the NEM eps_max.
  - HyTES MMD calibration: a=0.9961, b=0.7929, c=0.8234 (ATBD sec. 5.8, p. 41; 150 lab
    spectra, R2 = 0.987). v1 used unsourced values 0.994/0.730/0.820.
  - ASTER MMD calibration: a=0.994, b=0.687, c=0.737 (Gillespie et al. 1998; primary PDF
    check pending).
Simplifications w.r.t. the operational algorithm (to be disclosed): no eps_max refinement
(variance test / eps_max nodes / reset to 0.96), NEM run for a fixed two passes instead of
iterating to convergence, no graybody threshold classifier.

Author: Dr. Atilio Grondona & OpenTIR-PINN Research Team
Target Journal: IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.constants import h, c, k
from typing import Dict, Tuple, Optional, Any, List

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.config import SENSOR_CONFIG

# Physical Radiation Constants in SI / Thermal Remote Sensing Units
# c1 = 2 * h * c^2 in W * um^4 / (m^2 * sr) = 1.19104297e8
# c2 = h * c / k in um * K = 14387.77
C1 = 1.1910429723971887e8
C2 = 14387.768775039338


def planck_radiance(wav_um: np.ndarray, temp_k: np.ndarray) -> np.ndarray:
    """
    Computes spectral blackbody radiance B(lambda, T) in W / (m^2 * sr * um).
    Supports broadcasting across (Batch, N_bands).
    """
    exp_val = C2 / (wav_um * temp_k)
    exp_val = np.clip(exp_val, 1e-4, 80.0)
    return C1 / (wav_um**5 * (np.exp(exp_val) - 1.0))


def invert_planck_radiance(wav_um: np.ndarray, rad: np.ndarray) -> np.ndarray:
    """
    Analytically inverts Planck's law to recover Brightness Temperature in Kelvin:
    T = c2 / (lambda * ln(1 + c1 / (lambda^5 * rad)))
    """
    rad_safe = np.maximum(rad, 1e-6)
    ratio = C1 / (wav_um**5 * rad_safe)
    val = np.log1p(ratio)
    val = np.maximum(val, 1e-8)
    return C2 / (wav_um * val)


def compute_spectral_angle(p: np.ndarray, t: np.ndarray) -> float:
    """
    Computes Spectral Angle Mapper (SAM) in degrees between two emissivity vectors.
    """
    dot = np.sum(p * t)
    norm_p = np.linalg.norm(p)
    norm_t = np.linalg.norm(t)
    if norm_p == 0 or norm_t == 0:
        return 0.0
    cos_val = np.clip(dot / (norm_p * norm_t), -1.0, 1.0)
    return float(np.arccos(cos_val) * (180.0 / np.pi))


def compute_curvature_error(p: np.ndarray, t: np.ndarray) -> float:
    """
    Computes discrete Sobolev 2nd derivative L1 error (curvature error).
    """
    if len(p) < 3:
        return 0.0
    d2_p = p[2:] - 2.0 * p[1:-1] + p[:-2]
    d2_t = t[2:] - 2.0 * t[1:-1] + t[:-2]
    return float(np.mean(np.abs(d2_p - d2_t)))


class ClassicalTES:
    """
    Operational Classical Temperature and Emissivity Separation (TES) Processor.
    """
    def __init__(self, sensor: str = 'aster', eps_max: float = 0.99):
        self.sensor = sensor.lower()
        self.eps_max = eps_max

        if self.sensor == 'aster':
            # 5 discrete NASA ASTER TIR bands: Bands 10-14
            self.wavelengths = np.array([8.281, 8.633, 9.079, 10.662, 11.293])
            self.window_mask = np.ones(5, dtype=bool)  # All 5 bands are within the 8-12 um window
            # Canonical Gillespie et al. (1998) ASTER regression coefficients (Table 2 / Eq. 11)
            # eps_min = 0.994 - 0.687 * MMD^0.737
            self.mmd_a = 0.994
            self.mmd_b = 0.687
            self.mmd_c = 0.737

        elif self.sensor == 'hytes':
            # 256 contiguous HyTES airborne channels
            csv_path = SENSOR_CONFIG['hytes']['processed_params_file']
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"HyTES parameter file not found: {csv_path}")
            df = pd.read_csv(csv_path).sort_values('band_idx')
            self.wavelengths = df['center_wavelength'].values

            # Operational hyperspectral window: 8.2 to 11.5 um, masking out the 9.4-9.8 um ozone notch
            self.window_mask = (self.wavelengths >= 8.20) & (self.wavelengths <= 11.50) & \
                               ~((self.wavelengths >= 9.40) & (self.wavelengths <= 9.80))

            # HyTES MMD calibration curve: Hulley et al. (2016), HyTES L2 ATBD, sec. 5.8 (p. 41)
            self.mmd_a = 0.9961
            self.mmd_b = 0.7929
            self.mmd_c = 0.8234
        else:
            raise ValueError(f"Unsupported sensor for Classical TES: {sensor}")

    def invert_single(
        self,
        l_surf: np.ndarray,
        l_down: np.ndarray,
        max_iterations: int = 2
    ) -> Tuple[float, np.ndarray, float]:
        """
        Executes Classical TES on a single observation vector.

        Args:
            l_surf (np.ndarray): Surface-leaving band radiance in W/(m^2 sr um).
            l_down (np.ndarray): Surface downwelling sky radiance in W/(m^2 sr um).
            max_iterations (int): Number of reflected sky radiance refinement loops.

        Returns:
            Tuple[float, np.ndarray, float]:
                - T_ret (float): Retrieved surface kinetic temperature (Kelvin).
                - eps_ret (np.ndarray): Retrieved spectral emissivity vector (N_bands,).
                - mmd_val (float): Maximum-Minimum Difference spectral contrast.
        """
        cur_eps_max = self.eps_max

        for iteration in range(max_iterations):
            # -------------------------------------------------------------
            # Step 1: Normalized Emissivity Method (NEM)
            # -------------------------------------------------------------
            # Remove reflected downwelling radiance assuming cur_eps_max
            r_surf_nem = (l_surf[self.window_mask] - (1.0 - cur_eps_max) * l_down[self.window_mask]) / cur_eps_max
            r_surf_nem = np.maximum(r_surf_nem, 1e-5)
            t_bands_nem = invert_planck_radiance(self.wavelengths[self.window_mask], r_surf_nem)
            t_nem = float(np.max(t_bands_nem))

            # -------------------------------------------------------------
            # Step 2: Ratio Module (Beta Emissivity)
            # -------------------------------------------------------------
            b_nem_full = planck_radiance(self.wavelengths, t_nem)
            diff_denom = b_nem_full - l_down
            # Numerical safeguard against low radiometric contrast
            diff_denom = np.where(np.abs(diff_denom) < 1e-4, np.sign(diff_denom + 1e-8) * 1e-4, diff_denom)
            eps_est = (l_surf - l_down) / diff_denom

            # Compute beta spectrum normalized by window mean
            mean_eps_win = np.mean(eps_est[self.window_mask])
            if mean_eps_win <= 0.0 or np.isnan(mean_eps_win):
                mean_eps_win = 0.95
            beta_win = eps_est[self.window_mask] / mean_eps_win

            # -------------------------------------------------------------
            # Step 3: MMD Spectral Contrast
            # -------------------------------------------------------------
            beta_max = np.max(beta_win)
            beta_min = np.min(beta_win)
            mmd_val = float(np.clip(beta_max - beta_min, 0.0, 1.5))

            # -------------------------------------------------------------
            # Step 4: Empirical Power-Law Relationship
            # -------------------------------------------------------------
            eps_min_est = self.mmd_a - self.mmd_b * (mmd_val**self.mmd_c)
            eps_min_est = float(np.clip(eps_min_est, 0.40, 0.99))

            # -------------------------------------------------------------
            # Step 5: Emissivity Scaling
            # -------------------------------------------------------------
            safe_min_beta = max(beta_min, 0.10)
            alpha = eps_min_est / safe_min_beta

            beta_all = eps_est / mean_eps_win
            eps_ret = np.clip(alpha * beta_all, 0.01, 1.0)

            # -------------------------------------------------------------
            # Step 6: Final Kinetic Surface Temperature
            # -------------------------------------------------------------
            # Select band with highest emissivity within the clean window
            win_indices = np.where(self.window_mask)[0]
            best_band_idx = win_indices[np.argmax(eps_ret[win_indices])]

            r_best = (l_surf[best_band_idx] - (1.0 - eps_ret[best_band_idx]) * l_down[best_band_idx]) / eps_ret[best_band_idx]
            r_best = max(r_best, 1e-5)
            t_ret = float(invert_planck_radiance(self.wavelengths[best_band_idx], r_best))

            # Update eps_max for next iteration
            cur_eps_max = float(np.max(eps_ret[self.window_mask]))

        return t_ret, eps_ret, mmd_val

    def evaluate_dataset(self, samples: List[Dict[str, Any]], dataset_name: str = "") -> pd.DataFrame:
        """
        Executes Classical TES across an entire dataset and returns detailed per-sample metrics.
        """
        records = []
        for s in samples:
            l_surf = s['input_radiance'].numpy()
            l_down = s['downwelling_radiance'].numpy()
            t_true = float(s['target_temp'].item())
            eps_true = s['target_emissivity'].numpy()

            t_pred, eps_pred, mmd = self.invert_single(l_surf, l_down)

            dt = t_pred - t_true
            abs_dt = abs(dt)

            # Window-restricted emissivity metrics
            eps_true_win = eps_true[self.window_mask]
            eps_pred_win = eps_pred[self.window_mask]

            rmse_eps = float(np.sqrt(np.mean((eps_pred_win - eps_true_win)**2)))
            sam_deg = compute_spectral_angle(eps_pred_win, eps_true_win)
            curv_err = compute_curvature_error(eps_pred, eps_true) if self.sensor == 'hytes' else 0.0

            records.append({
                'dataset': dataset_name,
                'spectrum_id': s.get('spectrum_id', 'unknown'),
                'sample_id': s.get('sample_id', 'unknown'),
                'material_class': s.get('material_class', 'unknown'),
                'superclass': s.get('superclass', 'unknown'),
                'atmosphere_id': s.get('atmosphere_id', 'unknown'),
                'pwv': s.get('pwv', 0.0),
                'true_temp_k': t_true,
                'pred_temp_k': t_pred,
                'delta_t_k': dt,
                'abs_delta_t_k': abs_dt,
                'rmse_emissivity': rmse_eps,
                'sam_deg': sam_deg,
                'curvature_error': curv_err,
                'mmd': mmd
            })

        return pd.DataFrame(records)


def run_benchmark_suite(sensor: str, output_dir: str) -> None:
    """
    Runs Classical TES across Validation Fold 0, Quadrant II, and Quadrant IV.
    """
    print(f"\n{'='*75}")
    print(f"Executing Classical TES Benchmark Evaluation: {sensor.upper()}")
    print(f"{'='*75}")

    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets')
    os.makedirs(output_dir, exist_ok=True)

    tes = ClassicalTES(sensor=sensor)

    target_suites = [
        ('val_fold0', f'val_dataset_{sensor}.pt'),
        ('golden_quadrant_II', f'golden_benchmark_quadrant_II_{sensor}.pt'),
        ('golden_quadrant_IV_ood', f'golden_benchmark_quadrant_IV_{sensor}.pt'),
    ]

    all_dfs = []

    for suite_name, fname in target_suites:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"Warning: Dataset not found: {fpath}")
            continue

        print(f"\n--- Inverting Suite: {suite_name} ({fname}) ---")
        samples = torch.load(fpath, weights_only=False)
        df_suite = tes.evaluate_dataset(samples, dataset_name=suite_name)

        # Print summary statistics
        mae_t = df_suite['abs_delta_t_k'].mean()
        rmse_t = np.sqrt((df_suite['delta_t_k']**2).mean())
        mean_rmse_e = df_suite['rmse_emissivity'].mean()
        mean_sam = df_suite['sam_deg'].mean()
        mean_curv = df_suite['curvature_error'].mean() if sensor == 'hytes' else 0.0

        print(f"  Samples:     {len(df_suite)}")
        print(f"  ΔT MAE:      {mae_t:.2f} K")
        print(f"  ΔT RMSE:     {rmse_t:.2f} K")
        print(f"  Emis RMSE:   {mean_rmse_e:.4f}")
        print(f"  SAM:         {mean_sam:.2f}°")
        if sensor == 'hytes':
            print(f"  Curvature:   {mean_curv:.5f}")

        # Per-class summary
        print("\n  Per-Class Performance:")
        class_summary = df_suite.groupby('material_class').agg(
            N=('abs_delta_t_k', 'count'),
            dT_MAE=('abs_delta_t_k', 'mean'),
            RMSE_eps=('rmse_emissivity', 'mean'),
            SAM=('sam_deg', 'mean')
        ).round({'dT_MAE': 2, 'RMSE_eps': 4, 'SAM': 2})
        print(class_summary.to_string())

        all_dfs.append(df_suite)

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        out_csv = os.path.join(output_dir, f"classical_tes_benchmark_{sensor}.csv")
        full_df.to_csv(out_csv, index=False)
        print(f"\nSaved Full Benchmark Results to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classical TES Baseline Inversion Benchmark")
    parser.add_argument("--sensor", type=str, default="both", choices=["hytes", "aster", "both"], help="Target sensor")
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "baselines"), help="Output directory")
    args = parser.parse_args()

    sensors = ["hytes", "aster"] if args.sensor == "both" else [args.sensor]
    for s in sensors:
        run_benchmark_suite(sensor=s, output_dir=args.output_dir)
