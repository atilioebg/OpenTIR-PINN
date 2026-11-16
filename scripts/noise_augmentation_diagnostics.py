"""Quantitative diagnostics supporting the statistical analysis of noise & augmentation (Entry 37)."""
import os, sys
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from src.data_generation.base.synthesize_forward_dataset import (  # noqa
    WAVELENGTH_GRID, build_sensor_srf_matrix, load_atmospheric_luts, planck_temp_derivative,
    planck_blackbody_grid, load_rebalanced_library)

luts = load_atmospheric_luts()
lib = load_rebalanced_library()
dev = [a for a, v in luts.items() if v["quarantine_status"] == "development"]
gold = [a for a, v in luts.items() if "golden" in v["quarantine_status"]]
eps_mean = np.mean([v["emissivity"] for v in lib.values()], axis=0)

for sensor in ("aster", "hytes"):
    W, c = build_sensor_srf_matrix(sensor)
    print(f"\n===== {sensor.upper()} ({len(c)} bands) =====")
    # 1. band-level std of the per-channel N(0, 0.002^2) jitter after SRF convolution
    neff = 1.0 / (W ** 2).sum(1)
    sj = 0.002 / np.sqrt(neff)
    print(f"[1] SRF effective #grid points per band: median {np.median(neff):.1f} (min {neff.min():.1f}, max {neff.max():.1f})")
    print(f"    band-level std of the N(0,0.002^2) jitter: median {np.median(sj):.5f} (range {sj.min():.5f}-{sj.max():.5f})")
    # scale component std: eps * U(0.995,1.005)
    eb = W @ eps_mean
    print(f"    band-level std of the scale term (eps~{eb.mean():.3f}): {eb.mean() * 0.01 / np.sqrt(12):.5f}")
    # 2. equivalent brightness-temperature noise at other scene temperatures (fixed radiance sigma)
    s300 = planck_temp_derivative(c, 300.0)
    for T in (275.0, 280.0, 330.0, 335.0):
        r = s300 / planck_temp_derivative(c, T)
        print(f"[2] effective NEdT at {T:.0f} K (radiance sigma fixed at 300 K reference): {0.1 * r.mean():.3f} K (band range {0.1 * r.min():.3f}-{0.1 * r.max():.3f})")
    # 3. equivalent delta-T of the +-0.5% emissivity scale (eps*B(T) ~ eps*(1+a)*B(T+dT))
    lam = c * 1e-6; T = 300.0; C2 = 1.438777e-2
    x = C2 / (lam * T)
    dT_per_rel_eps = T / x * (1 - np.exp(-x))      # dT for d(eps)/eps = 1
    print(f"[3] |dT| equivalent to a 0.5% emissivity scale at 300 K: {0.005 * dT_per_rel_eps.mean():.3f} K (band range {0.005 * dT_per_rel_eps.min():.3f}-{0.005 * dT_per_rel_eps.max():.3f})")
    print(f"    |dT| equivalent to the median band jitter: {np.median(sj) / 0.95 * dT_per_rel_eps.mean():.3f} K")
    # 4. L_surf noise vs physically propagated noise sigma/tau (atmospheric correction amplification)
    key = "tau_aster" if sensor == "aster" else "tau_hytes"
    tau_dev = np.array([W @ luts[a][key] for a in dev])
    tau_gold = np.array([W @ luts[a][key] for a in gold])
    for name, t in (("dev", tau_dev), ("golden", tau_gold)):
        amp = 1.0 / t
        print(f"[4] {name} atmospheres: band tau median {np.median(t):.3f} (min {t.min():.3f}); "
              f"physical L_surf noise = sigma/tau -> amplification median {np.median(amp):.2f}x, 95th pct {np.percentile(amp, 95):.2f}x, max {amp.max():.2f}x")
    # 5. sigma of L_down noise relative to L_down signal and to its dev-atmosphere variability
    ld = np.array([W @ luts[a]["L_down_surface"] for a in dev])
    s = planck_temp_derivative(c, 300.0) * 0.1
    print(f"[5] 0.5*sigma / mean(L_down): median {np.median(0.5 * s / ld.mean(0)):.4%}; "
          f"between-atmosphere std of L_down / 0.5*sigma: median {np.median(ld.std(0) / (0.5 * s)):.0f}x")
    # 6. SNR of L_surf at 300 K for eps=0.95
    bb = W @ planck_blackbody_grid(300.0)
    print(f"[6] SNR of L_surf (eps 0.95, 300 K): median {np.median(0.95 * bb / s):.0f}")

