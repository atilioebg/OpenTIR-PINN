"""Post-regeneration validation of the v1.1.2 MODTRAN LUTs (decision item 5)."""
import os, sys, glob
import numpy as np
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from src.data_generation.base.automate_modtran import (build_tape5_deck, run_modtran_deck, hemispheric_downwelling,  # noqa
                                                       TEMPLATE_PATH, RT_IEMSCT, RT_IMULT, RT_DIS, RT_NSTR)
NEW = os.path.join(ROOT, "data", "processed", "atmospheric_luts")
OLD = os.path.join(ROOT, "data", "archive", "v1.1.1_snapshot", "atmospheric_luts")
tpl = open(TEMPLATE_PATH, "r", encoding="ascii").readlines()
C2 = 1.438777e4  # um K


def dBdT(lam_um, T):
    c1 = 1.191042e8  # W um^4 m^-2 sr^-1
    x = C2 / (lam_um * T)
    return c1 / lam_um ** 5 * np.exp(x) / (np.exp(x) - 1) ** 2 * x / T


def planck(lam_um, T):
    return 1.191042e8 / lam_um ** 5 / (np.exp(C2 / (lam_um * T)) - 1)


cat = pd.read_csv(os.path.join(NEW, "lut_catalog.csv"))
rows = []
for pid in cat.profile_id:
    a = np.load(os.path.join(NEW, f"profile_{pid}.npz"), allow_pickle=True)
    b = np.load(os.path.join(OLD, f"profile_{pid}.npz"), allow_pickle=True)
    lam = a["wavelength_um"]; m = (lam >= 8) & (lam <= 12)
    dT_eps08 = np.median((0.2 * (a["L_down_surface"] - b["L_down_surface"]))[m] / (0.8 * dBdT(lam[m], 300.0)))
    rows.append(dict(id=pid, max_dtau=np.abs(a["tau_aster"] - b["tau_aster"]).max(),
                     hemi_over_zen=np.mean(a["L_down_surface"][m] / a["L_down_zenith"][m]),
                     new_over_old_Ldown=np.mean(a["L_down_surface"][m] / b["L_down_surface"][m]),
                     new_over_old_Lup=np.mean(a["L_up_aster"][m] / b["L_up_aster"][m]),
                     dBT_eps080_K=dT_eps08,
                     min_Ldown=float(a["L_down_surface"].min()),
                     sky_colder_than_300K=bool(np.all(a["L_down_surface"][m] < planck(lam[m], 300.0)))))
d = pd.DataFrame(rows)
print("V1 transmittance unchanged vs v1.1.1 (MS/albedo do not alter TRAN): max |dtau| =", d.max_dtau.max())
print("V3 per-LUT summary (8-12 um means):")
print(d[["hemi_over_zen", "new_over_old_Ldown", "new_over_old_Lup", "dBT_eps080_K"]].describe().loc[["min", "mean", "max"]].round(3).to_string())
print("V5 physical sanity: min L_down =", round(d.min_Ldown.min(), 6), "| sky radiance < B(300 K) in window for all LUTs:", d.sky_colder_than_300K.all())

lam_nu = None
_, h8, _, _ = hemispheric_downwelling(tpl, 5, 0.5, n_quad=8)
nu, h16, _, _ = hemispheric_downwelling(tpl, 5, 0.5, n_quad=16)
w = (1e4 / nu >= 8) & (1e4 / nu <= 12)
print(f"V2 quadrature convergence (final config, atm5_h2o0.50): max |8pt-16pt|/16pt = {100 * np.max(np.abs(h8 - h16)[w] / h16[w]):.3f}%")

for alb in (0.06, 0.50):
    df, _ = run_modtran_deck(build_tape5_deck(tpl, 5, 0.5, 100.0, 0.0, 180.0, iemsct=RT_IEMSCT, imult=RT_IMULT,
                                              dis=RT_DIS, nstr=RT_NSTR, surref=alb))
    nu2 = df["FREQ(CM-1)"].values; lam2 = 1e4 / nu2; m2 = (lam2 >= 8) & (lam2 <= 12)
    if alb == 0.06:
        base = df["PTH_THRML"].values * nu2 ** 2
    else:
        diff = (df["PTH_THRML"].values * nu2 ** 2 - base)[m2]
        print(f"V4 residual surface coupling of L_up (atm5_h2o0.50, albedo 0.50 vs 0.06): "
              f"median {np.median(diff / dBdT(lam2[m2], 300.0)):+.3f} K-equivalent (limitation for eps~0.5 pixels)")
d.to_csv(os.path.join(ROOT, "data", "validation", "modtran_v112_lut_validation.csv"), index=False)

