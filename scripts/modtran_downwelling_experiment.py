"""
MODTRAN 5.2 experiment: definition of the downwelling sky term and purity of the upwelling path term.
=======================================================================================================
E1  Nadir (ASTER geometry) runs with SURREF = 0.0 / 0.4 / 1.0: does PTH_THRML contain ground-reflected
    thermal radiance (PcModWin manual p.523: with multiple scattering, Path Thermal "will also include
    thermal radiation ... reflected by the ground")?
E2  Hemispheric downwelling radiance from MODTRAN's own Lambertian reflection: with albedo 1 (emissivity 0)
    the reflected component equals tau * F_down/pi.
E3  Hemispheric downwelling radiance by angular quadrature of sky radiance:
    F_down/pi = 2 * int_0^1 L(mu) mu dmu  (Gauss-Legendre in mu), vs zenith L(0 deg) and diffusivity L(53 deg).
Outputs: data/validation/modtran_downwelling_experiment.npz + printed summary.
"""
import os, sys, re, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from src.data_generation.base.automate_modtran import build_tape5_deck, run_modtran_deck, TEMPLATE_PATH  # noqa

PROFILES = [(1, 2.0), (6, 1.0), (5, 0.5), (3, 1.0)]
N_GAUSS = 8


def set_surref(lines, albedo):
    l0 = lines[0].rstrip("\n")
    l0 = re.sub(r"\s*[0-9.]+\s*$", "", l0)          # strip trailing SURREF token
    l0 = l0.ljust(73)[:73] + f"{albedo:7.2f}"        # SURREF occupies cols 74-80 (A7)
    lines[0] = l0 + "\n"
    return lines


def lam(df):
    nu = df["FREQ(CM-1)"].values
    return nu, (lambda col: df[col].values * nu ** 2)


tpl = open(TEMPLATE_PATH, "r", encoding="ascii").readlines()
res = {}
t0 = time.time()
for model, h2o in PROFILES:
    key = f"atm{model}_h2o{h2o:.2f}"
    print(f"\n=== {key} ===")
    # E1/E2: nadir from 100 km with three albedos
    nadir = {}
    for alb in (0.0, 0.4, 1.0):
        d = set_surref(build_tape5_deck(tpl, model, h2o, h1=100.0, h2=0.0, angle=180.0), alb)
        df, _ = run_modtran_deck(d)
        nu, conv = lam(df)
        nadir[alb] = {c: conv(c) for c in ("PTH_THRML", "THRML_SCT", "SURF_EMIS", "GRND_RFLT", "TOTAL_RAD")}
        nadir[alb]["TRAN"] = df["TRAN"].values
    # E3: sky radiance at Gauss nodes + zenith + 53 deg
    x, w = np.polynomial.legendre.leggauss(N_GAUSS)
    mu = 0.5 * (x + 1.0); wq = 0.5 * w
    ang = np.degrees(np.arccos(mu))
    Ls = []
    for a in list(ang) + [0.0, 53.0]:
        d = build_tape5_deck(tpl, model, h2o, h1=0.02, h2=100.0, angle=float(a))
        df, pwv = run_modtran_deck(d)
        Ls.append(lam(df)[1]("TOTAL_RAD"))
    Ls = np.array(Ls)
    hemi_quad = 2.0 * np.sum((wq * mu)[:, None] * Ls[:N_GAUSS], axis=0)
    zen, l53 = Ls[N_GAUSS], Ls[N_GAUSS + 1]
    res[key] = dict(nu=nu, pwv=pwv, hemi_quad=hemi_quad, zenith=zen, l53=l53, angles=ang, L_angles=Ls[:N_GAUSS],
                    **{f"nadir{a}_{k}": v for a, dd in nadir.items() for k, v in dd.items()})
    print(f"  done ({time.time() - t0:.0f}s elapsed)")

out = os.path.join(ROOT, "data", "validation", "modtran_downwelling_experiment.npz")
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez_compressed(out, **{f"{k}__{kk}": vv for k, d in res.items() for kk, vv in d.items()})
print("saved", out)

