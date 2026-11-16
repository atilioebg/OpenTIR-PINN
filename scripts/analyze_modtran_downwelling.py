"""Analyse modtran_downwelling_experiment.npz: L_down definition and L_up purity, in sensor-band and BT terms."""
import os, sys
import numpy as np
from scipy.interpolate import interp1d
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from src.data_generation.base.synthesize_forward_dataset import (  # noqa
    WAVELENGTH_GRID, build_sensor_srf_matrix, planck_blackbody_grid, planck_temp_derivative)

z = np.load(os.path.join(ROOT, "data", "validation", "modtran_downwelling_experiment.npz"))
keys = sorted({k.split("__")[0] for k in z.files})
W = {s: build_sensor_srf_matrix(s) for s in ("aster", "hytes")}
win = (WAVELENGTH_GRID >= 8.0) & (WAVELENGTH_GRID <= 12.0)


def to_grid(nu, y):
    lam = 1e4 / nu
    o = np.argsort(lam)
    return interp1d(lam[o], y[o], bounds_error=False, fill_value=0.0)(WAVELENGTH_GRID)


for k in keys:
    g = lambda n: z[f"{k}__{n}"]
    nu = g("nu")
    hemi, zen, l53 = (to_grid(nu, g(n)) for n in ("hemi_quad", "zenith", "l53"))
    tau = to_grid(nu, g("nadir1.0_TRAN"))
    print(f"\n================ {k}  (PWV = {float(g('pwv')):.3f} g/cm2) ================")
    # ---- E1: purity of PTH_THRML vs surface albedo
    p0, p4, p1 = (to_grid(nu, g(f"nadir{a}_PTH_THRML")) for a in ("0.0", "0.4", "1.0"))
    gr0, gr4, gr1 = (to_grid(nu, g(f"nadir{a}_GRND_RFLT")) for a in ("0.0", "0.4", "1.0"))
    se0, se4, se1 = (to_grid(nu, g(f"nadir{a}_SURF_EMIS")) for a in ("0.0", "0.4", "1.0"))
    print("[E1] 8-12 um mean radiance W/(m2 sr um):   albedo 0.0 | 0.4 | 1.0")
    print(f"     PTH_THRML : {p0[win].mean():.4f} | {p4[win].mean():.4f} | {p1[win].mean():.4f}")
    print(f"     GRND_RFLT : {gr0[win].mean():.4f} | {gr4[win].mean():.4f} | {gr1[win].mean():.4f}")
    print(f"     SURF_EMIS : {se0[win].mean():.4f} | {se4[win].mean():.4f} | {se1[win].mean():.4f}")
    # ground-reflected sky contained in PTH_THRML (difference vs black surface), expressed as tau*F/pi
    refl_in_path = p1 - p0
    est_hemi_from_path = np.where(tau > 0.05, refl_in_path / np.maximum(tau, 1e-6), np.nan)
    # ---- E2/E3: downwelling definitions
    rel = lambda a, b: np.nanmean(((a - b) / b)[win & (tau > 0.05)])
    print(f"[E3] hemispheric (8-pt Gauss) vs zenith: zenith underestimates by {-100 * rel(zen, hemi):.1f}% (8-12 um mean)")
    print(f"     diffusivity L(53 deg) vs hemispheric: {100 * rel(l53, hemi):+.1f}%")
    print(f"[E2] MODTRAN Lambertian reflection [(PTH_THRML(a=1) - PTH_THRML(a=0)) / tau] vs quadrature: {100 * rel(est_hemi_from_path, hemi):+.1f}%")
    print(f"     PTH_THRML(a=0.4) - PTH_THRML(a=0) = {(p4 - p0)[win].mean():.4f} W/(m2 sr um)  vs expected 0.4*tau*F/pi = {(0.4 * tau * hemi)[win].mean():.4f}")
    # ---- impact in sensor bands / brightness temperature
    for s, (Wm, c) in W.items():
        sig = planck_temp_derivative(c, 300.0)
        for eps in (0.95, 0.80, 0.50):
            d_ref = (1 - eps) * (Wm @ hemi - Wm @ zen)          # missing reflected radiance in L_surf
            dT = d_ref / (eps * planck_temp_derivative(c, 300.0))
            print(f"[I] {s:5s} eps={eps:.2f}: L_surf error from zenith-L_down -> dBT median {np.median(dT):.3f} K, max {dT.max():.3f} K "
                  f"(= {np.median(dT) / 0.1:.1f}x NEdT 0.1 K)")
        d_up = Wm @ (p4 - p0)
        print(f"[U] {s:5s} contamination of L_up by ground-reflected sky (albedo 0.4) -> dBT-equivalent median "
              f"{np.median(d_up / planck_temp_derivative(c, 300.0)):.3f} K, max {np.max(d_up / planck_temp_derivative(c, 300.0)):.3f} K")

